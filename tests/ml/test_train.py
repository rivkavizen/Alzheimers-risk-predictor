import json
from pathlib import Path

import joblib
import pytest

from clean_data import run as clean_run
from train import MODEL_PATH, FEATURE_NAMES_PATH, train_and_save, load_cleaned_data


@pytest.fixture(scope="module")
def trained_model(raw_data_path, cleaned_data_path):
  clean_run(input_path=raw_data_path, output_path=cleaned_data_path)
  metrics = train_and_save(cleaned_data_path)
  yield metrics
  # keep artifacts for other tests


def test_train_auc_above_threshold(trained_model):
  assert trained_model["cv_auc"] > 0.85


def test_artifacts_saved(trained_model):
  assert MODEL_PATH.exists()
  assert FEATURE_NAMES_PATH.exists()
  pipeline = joblib.load(MODEL_PATH)
  features = json.loads(FEATURE_NAMES_PATH.read_text())
  assert len(features) == trained_model["features"]
  assert hasattr(pipeline, "predict_proba")


def test_load_cleaned_data_missing_file(tmp_path):
  with pytest.raises(FileNotFoundError, match="clean_data.py"):
    load_cleaned_data(tmp_path / "missing.csv")
