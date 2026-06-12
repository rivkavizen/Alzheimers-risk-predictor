"""Evaluate trained model and verify SHAP explanation structure."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from cleaning_config import CLEANED_DATA_PATH
from feature_engineering import BASE_COLUMNS
from train import FEATURE_NAMES_PATH, MODEL_PATH

ARTIFACTS_DIR = Path(__file__).resolve().parent / "model_artifacts"
SHAP_FEATURE_NAMES_PATH = ARTIFACTS_DIR / "shap_feature_names.json"


def get_top_factors(shap_values: np.ndarray, feature_names: list[str], top_n: int = 5) -> dict:
    pairs = list(zip(feature_names, shap_values.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top_risk = [[name, round(val, 4)] for name, val in pairs if val > 0][:top_n]
    top_protective = [[name, round(val, 4)] for name, val in reversed(pairs) if val < 0][:top_n]
    all_shap = {name: round(val, 4) for name, val in pairs}
    return {
        "top_risk_factors": top_risk,
        "top_protective_factors": top_protective,
        "all_shap": all_shap,
    }


def explain_sample(pipeline, feature_names: list[str], row: pd.DataFrame) -> dict:
    features_step = pipeline.named_steps["features"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_eng = features_step.transform(row[BASE_COLUMNS])
    X_transformed = preprocessor.transform(X_eng)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)[0]
    return get_top_factors(shap_values, feature_names)


def run() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run: python ml/train.py")

    pipeline = joblib.load(MODEL_PATH)
    base_features: list[str] = json.loads(FEATURE_NAMES_PATH.read_text())
    shap_features: list[str] = (
        json.loads(SHAP_FEATURE_NAMES_PATH.read_text())
        if SHAP_FEATURE_NAMES_PATH.exists()
        else base_features
    )
    df = pd.read_csv(CLEANED_DATA_PATH)
    sample = df[base_features].iloc[0:1]

    risk_score = float(pipeline.predict_proba(sample)[0][1])
    risk_label = "High" if risk_score >= 0.5 else "Low"
    shap_explanation = explain_sample(pipeline, shap_features, sample)

    result = {
        "risk_score": round(risk_score, 4),
        "risk_label": risk_label,
        "shap_explanation": shap_explanation,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    run()


if __name__ == "__main__":
    main()
