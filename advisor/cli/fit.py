"""Fit the clustering model and write the artifacts the guidance step loads.

    python -m advisor.cli.fit [--data data/raw/sleep_health.csv] [--out artifacts]

Writes three files:
- model.joblib: features -> scaler -> KMeans, one sklearn Pipeline; predict() takes raw rows
- profiles.json: what each cluster looks like, plus population mean and std for deltas
- metadata.json: when, on which data (sha256), with which versions

See docs/CLUSTERING_NOTES.md for the choices behind k, the features and the profiles.
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from advisor.features import FEATURES, WellnessFeatures, clean
from advisor.log import configure, logger
from advisor.models import MEASURES

K = 5
N_INIT = 200
RANDOM_STATE = 42

PROFILE_COLUMNS = list(MEASURES)  # raw columns, original units
SHARE_COLUMNS = ["BMI Category", "Sleep Disorder", "Gender"]

# Derived from the group means. Cluster ids depend on data, seed and sklearn
# version, so main() checks the group sizes before attaching these.
SUMMARIES = {
    0: "Late thirties. Short and poor sleep, high stress, the least active group. About half overweight.",
    1: "Early fifties, normal BMI. Longest and best sleep, lowest stress, little activity.",
    2: "Late thirties, normal BMI. Sleep well, active, lowest blood pressure.",
    3: "Around 52, overweight or obese. Sleep well and report low stress, but blood pressure almost as high as the most stressed group. Most have a sleep disorder.",
    4: "Around 50, overweight, the most active. Shortest sleep, highest stress, highest blood pressure.",
}
EXPECTED_SIZES = {0: 101, 1: 34, 2: 145, 3: 62, 4: 32}

# What guidance for each group should be about. Not advice: the perimeter given to the model.
FOCUS = {
    0: [
        "sleep duration and sleep quality, both about one standard deviation below the population",
        "stress, which goes with the short sleep",
        "activity, the lowest of all groups",
    ],
    1: [
        "physical activity, the one thing below average in an otherwise healthy group",
        "keeping the good sleep and low stress while getting older",
    ],
    2: [
        "maintenance: sleep, stress, activity and blood pressure are all at or better than average",
        "small refinements rather than changes",
    ],
    3: [
        "blood pressure, more than one standard deviation above the population despite good habits",
        "a sleep disorder in 92% of the group, mostly apnea: point to a professional, do not advise on it",
        "weight, through movement rather than diet: almost everyone is overweight, and food advice is not given here",
    ],
    4: [
        "stress and short sleep, the highest and lowest of all groups even though activity is the highest",
        "blood pressure and heart rate, both high: point to a professional",
        "recovery rather than more exercise",
    ],
}


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("features", WellnessFeatures()),
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=K, n_init=N_INIT, random_state=RANDOM_STATE)),
        ]
    )


def profiles(df: pd.DataFrame, labels: pd.Series) -> dict:
    mean, std = df[PROFILE_COLUMNS].mean(), df[PROFILE_COLUMNS].std()
    out = {}
    for cluster in sorted(labels.unique()):
        group = df[labels == cluster]
        group_mean = group[PROFILE_COLUMNS].mean()
        out[str(cluster)] = {
            "summary": SUMMARIES[cluster],
            "focus": FOCUS[cluster],
            "size": len(group),
            "distinct_rows": int(group[FEATURES].drop_duplicates().shape[0]),
            "mean": group_mean.round(1).to_dict(),
            "deviation_from_population_sd": ((group_mean - mean) / std).round(2).to_dict(),
            "share": {
                c: group[c].value_counts(normalize=True).round(2).to_dict() for c in SHARE_COLUMNS
            },
        }
    return {
        "clusters": out,
        "population": {"mean": mean.round(2).to_dict(), "std": std.round(2).to_dict()},
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/raw/sleep_health.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    configure()

    if not args.data.exists():
        logger.error("%s not found. Run `make data` first.", args.data)
        return 1
    raw = pd.read_csv(args.data)

    try:
        pipeline = build_pipeline().fit(raw)
    except ValueError as e:  # a value clean() does not know
        logger.error("%s", e)
        return 1
    labels = pd.Series(pipeline["kmeans"].labels_, index=raw.index, name="cluster")

    sizes = labels.value_counts().to_dict()
    if sizes != EXPECTED_SIZES:
        logger.error(
            "cluster sizes %s differ from %s: the summaries in advisor/cli/fit.py no longer match "
            "the clusters, check them against the group means before writing artifacts",
            dict(sorted(sizes.items())),
            EXPECTED_SIZES,
        )
        return 1

    cleaned = clean(raw)
    cleaned[FEATURES] = pipeline["features"].transform(raw)
    scaled = pipeline[:-1].transform(raw)

    args.out.mkdir(exist_ok=True)
    joblib.dump(pipeline, args.out / "model.joblib")
    (args.out / "profiles.json").write_text(json.dumps(profiles(cleaned, labels), indent=2))
    metadata = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {"path": str(args.data), "sha256": sha256(args.data), "rows": len(raw)},
        "model": {
            "k": K,
            "n_init": N_INIT,
            "random_state": RANDOM_STATE,
            "features": FEATURES,
            "inertia": round(float(pipeline["kmeans"].inertia_), 1),
            "silhouette": round(float(silhouette_score(scaled, labels)), 3),
        },
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info(
        "fit on %d rows, k=%d, n_init=%d, silhouette %.3f",
        len(raw),
        K,
        N_INIT,
        metadata["model"]["silhouette"],
    )
    logger.info("wrote model.joblib, profiles.json, metadata.json to %s/", args.out)
    print(labels.value_counts().sort_index().rename("people").to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
