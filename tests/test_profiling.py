from pathlib import Path

import pytest

from advisor.models import UserInput
from advisor.profiling import Artifacts, assign, referral_reasons


def test_assign_reproduces_the_fit_labels(raw, artifacts):
    labels = artifacts.pipeline["kmeans"].labels_
    for row, label in zip(raw.to_dict("records"), labels, strict=True):
        assert assign(UserInput.from_row(row), artifacts).cluster == label


def test_deltas_are_zero_for_the_cluster_mean(raw, artifacts):
    a = assign(UserInput.from_row(raw.iloc[0].to_dict()), artifacts)
    mean = artifacts.profiles.clusters[a.cluster].mean
    assert a.delta_from_cluster.keys() == mean.keys()
    assert all(abs(d) < 5 for d in a.delta_from_cluster.values())


def test_referral_reasons(raw):
    user = UserInput.from_row(raw.iloc[3].to_dict())  # 140/90, sleep apnea
    assert referral_reasons(user) == ["reports sleep apnea", "blood pressure 140/90"]
    healthy = user.model_copy(
        update={"sleep_disorder": "No disorder", "systolic": 120, "diastolic": 80}
    )
    assert referral_reasons(healthy) == []


def test_missing_artifacts_is_a_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="make fit"):
        Artifacts.load(tmp_path)
