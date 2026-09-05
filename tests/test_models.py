import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from advisor.models import Guidance, GuidanceRecord, UserInput

USER = {
    "age": 30,
    "sleep_duration": 7,
    "quality_of_sleep": 7,
    "physical_activity_level": 60,
    "daily_steps": 7000,
    "stress_level": 5,
    "heart_rate": 70,
    "systolic": 120,
    "diastolic": 80,
    "bmi_category": "Normal",
}


def test_every_dataset_row_is_a_valid_user(raw):
    users = [UserInput.from_row(r) for r in raw.to_dict("records")]
    assert len(users) == len(raw)
    assert {u.sleep_disorder for u in users} == {"No disorder", "Insomnia", "Sleep Apnea"}


def test_normal_weight_is_normal():
    user = UserInput.model_validate({**USER, "bmi_category": "Normal Weight"})
    assert user.bmi_category == "Normal"


@pytest.mark.parametrize(
    "bad",
    [
        {"systolic": 80, "diastolic": 90},
        {"bmi_category": "Underweight"},
        {"sleep_duration": 30},
        {"age": 12},
    ],
)
def test_implausible_users_are_rejected(bad):
    with pytest.raises(ValidationError):
        UserInput.model_validate({**USER, **bad})


def test_guidance_rejects_markup_and_wrong_counts():
    with pytest.raises(ValidationError, match="markup"):
        Guidance(summary="ok</summary>", recommendations=["a", "b", "c"])
    with pytest.raises(ValidationError):
        Guidance(summary="ok", recommendations=["a", "b"])


def test_saved_runs_still_load_as_records():
    # runs made before the markup validator existed are kept as evidence and skipped here
    paths = [p for p in Path("eval/runs").rglob("*.json") if "novalidator" not in p.name]
    assert paths
    for path in paths:
        for item in json.loads(path.read_text()):
            GuidanceRecord.model_validate(item)
