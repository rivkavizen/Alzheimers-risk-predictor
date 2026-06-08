from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "alzheimers_disease_data.csv"
CLEANED_DATA = PROJECT_ROOT / "alzheimers_disease_data_cleaned.csv"


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def raw_data_path():
    return RAW_DATA


@pytest.fixture(scope="session")
def cleaned_data_path():
    return CLEANED_DATA
