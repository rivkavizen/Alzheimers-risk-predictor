import json

import joblib
import pandas as pd
import pytest

from clean_data import run as clean_run
from evaluate import run as evaluate_run
from train import FEATURE_NAMES_PATH, MODEL_PATH, train_and_save


@pytest.fixture(scope="module")
def shap_result(raw_data_path, cleaned_data_path):
  clean_run(input_path=raw_data_path, output_path=cleaned_data_path)
  train_and_save(cleaned_data_path)
  return evaluate_run()


def test_shap_structure(shap_result):
  shap_data = shap_result["shap_explanation"]
  assert "top_risk_factors" in shap_data
  assert "top_protective_factors" in shap_data
  assert "all_shap" in shap_data
  assert 0 <= shap_result["risk_score"] <= 1
  assert shap_result["risk_label"] in ("High", "Low")

  for name, val in shap_data["top_risk_factors"]:
    assert val > 0
  for name, val in shap_data["top_protective_factors"]:
    assert val < 0

  features = json.loads(FEATURE_NAMES_PATH.read_text())
  assert set(shap_data["all_shap"].keys()) == set(features)
