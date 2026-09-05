import pandas as pd
import pytest

from advisor.features import FEATURES, WellnessFeatures, clean


def test_transform_gives_the_eight_features(raw):
    out = WellnessFeatures().fit_transform(raw)
    assert list(out.columns) == FEATURES
    assert len(out) == len(raw)
    assert out.notna().all().all()


def test_one_row_alone_equals_the_row_in_a_batch(raw):
    fitted = WellnessFeatures().fit(raw.iloc[:300])
    alone = fitted.transform(raw.iloc[[350]]).iloc[0]
    in_batch = fitted.transform(raw.iloc[300:]).loc[350]
    pd.testing.assert_series_equal(alone, in_batch, check_names=False)


def test_clean_rejects_unknown_bmi(raw):
    row = raw.iloc[[0]].assign(**{"BMI Category": "Underweight"})
    with pytest.raises(ValueError, match="Underweight"):
        clean(row)


def test_clean_rejects_malformed_blood_pressure(raw):
    row = raw.iloc[[0]].assign(**{"Blood Pressure": "high"})
    with pytest.raises(ValueError, match="120/80"):
        clean(row)
