"""Feature engineering for the clustering model.

Reproduces what notebooks/features.ipynb does, so that the same
transformation runs on the training data and on a new user.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

RAW_COLUMNS = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Daily Steps",
    "Stress Level",
    "Heart Rate",
    "Blood Pressure",
    "BMI Category",
]

FEATURES = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Activity Index",
    "Stress Level",
    "Heart Rate",
    "Mean Arterial Pressure",
    "BMI",
]

BMI_LEVELS = {"Normal": 0, "Overweight": 1, "Obese": 2}

ACTIVITY_PAIR = ["Physical Activity Level", "Daily Steps"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning decided in the EDA. No fitted state."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    out = df.copy()
    out["BMI Category"] = out["BMI Category"].replace({"Normal Weight": "Normal"})
    unknown = sorted(set(out["BMI Category"]) - set(BMI_LEVELS), key=str)
    if unknown:
        raise ValueError(f"unknown BMI category {unknown}, expected one of {list(BMI_LEVELS)}")
    out["BMI"] = out["BMI Category"].map(BMI_LEVELS)

    pressure = out["Blood Pressure"].astype(str).str.strip()
    valid = pressure.str.fullmatch(r"\d{2,3}/\d{2,3}")
    if not valid.all():
        bad = out.loc[~valid, "Blood Pressure"].head().tolist()
        raise ValueError(f"Blood Pressure must look like '120/80', got {bad}")
    parts = pressure.str.split("/", expand=True).astype(int)
    out["Systolic"] = parts[0]
    out["Diastolic"] = parts[1]
    out["Mean Arterial Pressure"] = out["Diastolic"] + (out["Systolic"] - out["Diastolic"]) / 3

    if "Sleep Disorder" in out.columns:
        out["Sleep Disorder"] = out["Sleep Disorder"].fillna("No disorder")
    return out


class WellnessFeatures(BaseEstimator, TransformerMixin):
    """Raw rows in, the eight clustering features out."""

    def fit(self, X: pd.DataFrame, y: object = None) -> WellnessFeatures:
        cleaned = clean(X)
        self.means_ = cleaned[ACTIVITY_PAIR].mean()
        self.stds_ = cleaned[ACTIVITY_PAIR].std()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["means_", "stds_"])
        cleaned = clean(X)
        z = (cleaned[ACTIVITY_PAIR] - self.means_) / self.stds_
        cleaned["Activity Index"] = z.mean(axis=1)
        return cleaned[FEATURES]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(FEATURES)
