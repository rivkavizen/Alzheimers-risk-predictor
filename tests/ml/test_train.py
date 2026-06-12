import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from clean_data import run as clean_run
from train import (
    EVALUATION_REPORT_PATH,
    FEATURES_CSV_PATH,
    FEATURE_NAMES_PATH,
    MODEL_CARD_PATH,
    MODEL_PATH,
    train_and_save,
    load_cleaned_data,
)

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "ml" / "model_artifacts"
SHAP_FEATURE_NAMES_PATH = ARTIFACTS_DIR / "shap_feature_names.json"


@pytest.fixture(scope="module")
def trained_model(raw_data_path, tmp_path_factory):
    work = tmp_path_factory.mktemp("train")
    cleaned = work / "cleaned.csv"
    clean_run(input_path=raw_data_path, output_path=cleaned)
    metrics = train_and_save(cleaned)
    yield metrics


def test_train_auc_above_threshold(trained_model):
    assert trained_model["cv_auc"] > 0.85


def test_model_comparison_has_two_variations(trained_model):
    assert len(trained_model["model_comparison"]) >= 2


def test_artifacts_saved(trained_model):
    assert MODEL_PATH.exists()
    assert FEATURE_NAMES_PATH.exists()
    assert FEATURES_CSV_PATH.exists()
    assert EVALUATION_REPORT_PATH.exists()
    assert MODEL_CARD_PATH.exists()
    assert SHAP_FEATURE_NAMES_PATH.exists()

    pipeline = joblib.load(MODEL_PATH)
    features = json.loads(FEATURE_NAMES_PATH.read_text())
    assert len(features) == trained_model["base_features"]
    assert hasattr(pipeline, "predict_proba")

    df = pd.read_csv(FEATURES_CSV_PATH)
    assert "Diagnosis" in df.columns
    assert len(df.columns) == trained_model["total_features"] + 1

    assert "# Model Evaluation Report" in EVALUATION_REPORT_PATH.read_text(encoding="utf-8")
    assert "# Model Card" in MODEL_CARD_PATH.read_text(encoding="utf-8")


def test_load_cleaned_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="clean_data.py"):
        load_cleaned_data(tmp_path / "missing.csv")
