"""Data models shared by the fit, guidance and evaluation steps."""

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from advisor.features import BMI_ALIASES

BMICategory = Literal["Normal", "Overweight", "Obese"]
SleepDisorder = Literal["No disorder", "Insomnia", "Sleep Apnea"]

# The numeric measures, dataset column -> UserInput field. Profiles, deltas and the prompt
# table are all described on these.
MEASURES = {
    "Age": "age",
    "Sleep Duration": "sleep_duration",
    "Quality of Sleep": "quality_of_sleep",
    "Stress Level": "stress_level",
    "Physical Activity Level": "physical_activity_level",
    "Daily Steps": "daily_steps",
    "Heart Rate": "heart_rate",
    "Systolic": "systolic",
    "Diastolic": "diastolic",
}


class UserInput(BaseModel):
    """One user's raw data, in the units of the dataset. Ranges are plausibility bounds,
    not the training range."""

    model_config = ConfigDict(frozen=True)

    age: int = Field(ge=18, le=100)
    sleep_duration: float = Field(ge=0, le=24, description="hours per night")
    quality_of_sleep: int = Field(ge=1, le=10, description="self reported, 1 to 10")
    physical_activity_level: int = Field(ge=0, le=1440, description="minutes per day")
    daily_steps: int = Field(ge=0, le=100_000)
    stress_level: int = Field(ge=1, le=10, description="self reported, 1 to 10")
    heart_rate: int = Field(ge=30, le=220, description="resting, beats per minute")
    systolic: int = Field(ge=70, le=250)
    diastolic: int = Field(ge=40, le=150)
    bmi_category: BMICategory
    sleep_disorder: SleepDisorder = "No disorder"
    gender: str | None = Field(default=None, description="context for the model, not a feature")
    occupation: str | None = Field(default=None, description="context for the model, not a feature")

    @field_validator("bmi_category", mode="before")
    @classmethod
    def _normal_weight_is_normal(cls, value: object) -> object:
        return BMI_ALIASES.get(value, value) if isinstance(value, str) else value

    @field_validator("sleep_disorder", mode="before")
    @classmethod
    def _missing_means_none(cls, value: object) -> object:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "No disorder"
        return value

    @model_validator(mode="after")
    def _pressure_order(self) -> Self:
        if self.systolic <= self.diastolic:
            raise ValueError("systolic must be higher than diastolic")
        return self

    @classmethod
    def from_row(cls, row: dict) -> Self:
        """From a dataset row, where blood pressure is one text column like '126/83'."""
        systolic, diastolic = str(row["Blood Pressure"]).split("/")
        row = {**row, "Systolic": int(systolic), "Diastolic": int(diastolic)}
        return cls.model_validate(
            {
                **{field: row[column] for column, field in MEASURES.items()},
                "bmi_category": row["BMI Category"],
                "sleep_disorder": row.get("Sleep Disorder"),
                "gender": row.get("Gender"),
                "occupation": row.get("Occupation"),
            }
        )

    def measures(self) -> dict[str, int | float]:
        return {column: getattr(self, field) for column, field in MEASURES.items()}

    def to_frame(self) -> pd.DataFrame:
        """One dataset-shaped row, the input the fitted pipeline expects."""
        measures = self.measures()
        row = {
            **measures,
            "Blood Pressure": f"{measures.pop('Systolic')}/{measures.pop('Diastolic')}",
            "BMI Category": self.bmi_category,
            "Sleep Disorder": self.sleep_disorder,
        }
        return pd.DataFrame([row])


class ClusterProfile(BaseModel):
    summary: str
    focus: list[str] = Field(
        description="What guidance for this group should be about. Not advice."
    )
    size: int
    distinct_rows: int
    mean: dict[str, float]
    deviation_from_population_sd: dict[str, float]
    share: dict[str, dict[str, float]]


class Population(BaseModel):
    mean: dict[str, float]
    std: dict[str, float]


class Profiles(BaseModel):
    clusters: dict[int, ClusterProfile]
    population: Population

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_text())


class ArtifactMetadata(BaseModel):
    created_at: datetime
    dataset: dict
    model: dict
    versions: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate(json.loads(path.read_text()))


class GuidanceRecord(BaseModel):
    """One generated guidance with everything needed to evaluate and to account for it."""

    case: str
    cluster: int
    user: "UserInput"
    referral_reasons: list[str]
    guidance: "Guidance"
    model: str
    prompt_version: str
    reasoning: bool
    rag: bool = False
    tool_calls: list[dict] = []  # {"query": ..., "results": [citation, ...]} per search
    latency_seconds: float
    input_tokens: int
    output_tokens: int


class Guidance(BaseModel):
    summary: str = Field(
        max_length=400, description="Two or three sentences on where this user stands."
    )
    recommendations: list[str] = Field(
        min_length=3, max_length=4, description="Concrete habit changes, one sentence each."
    )

    @field_validator("summary", "recommendations")
    @classmethod
    def _no_markup(cls, value: object) -> object:
        # minimax leaks tool-call markup into the text; a validation error makes it retry
        texts = value if isinstance(value, list) else [value]
        for text in texts:
            if isinstance(text, str) and re.search(r"</?[A-Za-z_]+>", text):
                raise ValueError(f"contains markup tags, write plain text: {text[-40:]!r}")
        return value
