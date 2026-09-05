"""Assign a user to a cluster and compute their deltas. No network calls."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from advisor.models import MEASURES, ArtifactMetadata, ClusterProfile, Profiles, UserInput

ARTIFACTS = Path("artifacts")

# Referral thresholds.
SYSTOLIC_REFERRAL = 140
DIASTOLIC_REFERRAL = 90
RESTING_HEART_RATE_REFERRAL = 100


@dataclass(frozen=True)
class Artifacts:
    pipeline: Pipeline
    profiles: Profiles
    metadata: ArtifactMetadata

    @classmethod
    def load(cls, directory: Path = ARTIFACTS) -> Self:
        try:
            return cls(
                pipeline=joblib.load(directory / "model.joblib"),
                profiles=Profiles.load(directory / "profiles.json"),
                metadata=ArtifactMetadata.load(directory / "metadata.json"),
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{e.filename} not found. Run `make fit` first.") from e

    def scale(self, raw: pd.DataFrame) -> np.ndarray:
        """Rows in the feature space the clustering works in."""
        return np.asarray(self.pipeline[:-1].transform(raw))

    def distances_to_centres(self, scaled: np.ndarray) -> np.ndarray:
        """Rows x clusters."""
        centres = self.pipeline["kmeans"].cluster_centers_
        return np.linalg.norm(scaled[:, None, :] - centres[None], axis=2)


@dataclass(frozen=True)
class Assignment:
    user: UserInput
    cluster: int
    profile: ClusterProfile
    population_mean: dict[str, float]
    delta_from_cluster: dict[str, float]  # in population SD
    delta_from_population: dict[str, float]
    referral_reasons: list[str]


def referral_reasons(user: UserInput) -> list[str]:
    reasons = []
    if user.sleep_disorder != "No disorder":
        reasons.append(f"reports {user.sleep_disorder.lower()}")
    if user.systolic >= SYSTOLIC_REFERRAL or user.diastolic >= DIASTOLIC_REFERRAL:
        reasons.append(f"blood pressure {user.systolic}/{user.diastolic}")
    if user.heart_rate >= RESTING_HEART_RATE_REFERRAL:
        reasons.append(f"resting heart rate {user.heart_rate}")
    return reasons


def assign(user: UserInput, artifacts: Artifacts) -> Assignment:
    cluster = int(artifacts.pipeline.predict(user.to_frame())[0])
    profile = artifacts.profiles.clusters[cluster]
    population = artifacts.profiles.population

    from_cluster, from_population = {}, {}
    for column, value in user.measures().items():
        std = population.std[column]
        from_cluster[column] = round((value - profile.mean[column]) / std, 2)
        from_population[column] = round((value - population.mean[column]) / std, 2)

    return Assignment(
        user=user,
        cluster=cluster,
        profile=profile,
        population_mean={c: round(population.mean[c], 1) for c in MEASURES},
        delta_from_cluster=from_cluster,
        delta_from_population=from_population,
        referral_reasons=referral_reasons(user),
    )
