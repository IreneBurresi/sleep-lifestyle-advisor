"""Pick the users the guidance is evaluated on and write them to eval/testset.json.

    python -m advisor.cli.build_testset

Nine real rows:
- typical: one per cluster, closest to the centre
- divergent: two more members of the cluster with the most internal variety, far from the
  centre on different features
- boundary: the two rows with the smallest gap between their own centre and the next one
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from advisor.features import FEATURES
from advisor.models import UserInput
from advisor.profiling import Artifacts

DIVERGENT = 2
BOUNDARY = 2


def pick(raw: pd.DataFrame, artifacts: Artifacts) -> list[dict]:
    kmeans = artifacts.pipeline["kmeans"]
    labels = kmeans.labels_
    scaled = artifacts.scale(raw)
    distance = artifacts.distances_to_centres(scaled)
    own = distance[np.arange(len(raw)), labels]
    margin = np.sort(distance, axis=1)[:, 1] - own

    def case(row: int, role: str, note: str) -> dict:
        return {
            "id": f"c{labels[row]}-{role}",
            "cluster": int(labels[row]),
            "role": role,
            "row": int(row),
            "note": note,
            "distance_to_centre": round(float(own[row]), 2),
            "margin_to_next_cluster": round(float(margin[row]), 2),
            "user": UserInput.from_row(raw.iloc[row].to_dict()).model_dump(),
        }

    cases = []
    for cluster in range(kmeans.n_clusters):
        members = np.flatnonzero(labels == cluster)
        cases.append(case(members[np.argmin(own[members])], "typical", "closest to the centre"))

    offsets = scaled - kmeans.cluster_centers_[labels]
    far = np.abs(offsets).max(axis=1) > 1
    variety = {
        c: len(np.unique(scaled[(labels == c) & far], axis=0)) for c in range(kmeans.n_clusters)
    }
    cluster = max(variety, key=lambda c: variety[c])
    members = np.flatnonzero((labels == cluster) & far)
    chosen: list[int] = []
    for _ in range(DIVERGENT):
        # farthest from the centre on a feature not already covered, skipping duplicates
        used = {int(np.abs(offsets[r]).argmax()) for r in chosen}
        score = np.abs(offsets[members]).copy()
        score[:, list(used)] = 0
        for r in chosen:
            score[np.all(scaled[members] == scaled[r], axis=1)] = 0
        row = int(members[score.max(axis=1).argmax()])
        chosen.append(row)
        feature = int(np.abs(offsets[row]).argmax())
        cases.append(
            case(
                row,
                f"divergent-{len(chosen)}",
                f"{FEATURES[feature]} {offsets[row, feature]:+.1f} SD from the cluster centre",
            )
        )

    for n, row in enumerate(np.argsort(margin)[:BOUNDARY], start=1):
        second = int(np.argsort(distance[row])[1])
        cases.append(case(int(row), f"boundary-{n}", f"almost as close to cluster {second}"))
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=Path("data/raw/sleep_health.csv"))
    parser.add_argument("--out", type=Path, default=Path("eval/testset.json"))
    args = parser.parse_args(argv)

    try:
        cases = pick(pd.read_csv(args.data), Artifacts.load())
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(cases, indent=2))
    print(f"wrote {len(cases)} cases to {args.out}")
    columns = ["id", "row", "distance_to_centre", "margin_to_next_cluster", "note"]
    print(pd.DataFrame(cases)[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
