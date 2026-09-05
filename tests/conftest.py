from pathlib import Path

import pandas as pd
import pytest

from advisor.profiling import Artifacts

DATA = Path("data/raw/sleep_health.csv")
ARTIFACTS = Path("artifacts")


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    if not DATA.exists():
        pytest.skip("run `make data` first")
    return pd.read_csv(DATA)


@pytest.fixture(scope="session")
def artifacts() -> Artifacts:
    if not ARTIFACTS.exists():
        pytest.skip("run `make fit` first")
    return Artifacts.load(ARTIFACTS)
