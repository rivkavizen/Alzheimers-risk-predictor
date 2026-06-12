from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from config import FEATURE_NAMES_PATH, MODEL_PATH, PROJECT_ROOT, SHAP_FEATURE_NAMES_PATH


def _ensure_ml_import_path() -> None:
    """model.pkl references ml/feature_engineering.py classes at unpickle time."""
    ml_dir = str(PROJECT_ROOT / "ml")
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)

# API snake_case -> model PascalCase
FEATURE_MAP = {
    "age": "Age",
    "gender": "Gender",
    "ethnicity": "Ethnicity",
    "education_level": "EducationLevel",
    "bmi": "BMI",
    "smoking": "Smoking",
    "alcohol_consumption": "AlcoholConsumption",
    "physical_activity": "PhysicalActivity",
    "diet_quality": "DietQuality",
    "sleep_quality": "SleepQuality",
    "family_history_alzheimers": "FamilyHistoryAlzheimers",
    "cardiovascular_disease": "CardiovascularDisease",
    "diabetes": "Diabetes",
    "depression": "Depression",
    "head_injury": "HeadInjury",
    "hypertension": "Hypertension",
    "systolic_bp": "SystolicBP",
    "diastolic_bp": "DiastolicBP",
    "cholesterol_total": "CholesterolTotal",
    "cholesterol_ldl": "CholesterolLDL",
    "cholesterol_hdl": "CholesterolHDL",
    "cholesterol_triglycerides": "CholesterolTriglycerides",
    "mmse": "MMSE",
    "functional_assessment": "FunctionalAssessment",
    "memory_complaints": "MemoryComplaints",
    "behavioral_problems": "BehavioralProblems",
    "adl": "ADL",
    "confusion": "Confusion",
    "disorientation": "Disorientation",
    "personality_changes": "PersonalityChanges",
    "difficulty_completing_tasks": "DifficultyCompletingTasks",
    "forgetfulness": "Forgetfulness",
}

MODEL_TO_SNAKE = {v: k for k, v in FEATURE_MAP.items()}

_pipeline = None
_feature_names: list[str] = []
_shap_feature_names: list[str] = []


def load_model() -> None:
    global _pipeline, _feature_names, _shap_feature_names
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    _ensure_ml_import_path()
    _pipeline = joblib.load(MODEL_PATH)
    _feature_names = json.loads(FEATURE_NAMES_PATH.read_text())
    if SHAP_FEATURE_NAMES_PATH.exists():
        _shap_feature_names = json.loads(SHAP_FEATURE_NAMES_PATH.read_text())
    else:
        _shap_feature_names = _feature_names


def _normalize_input(patient_data: dict) -> dict:
    normalized = {}
    for key, value in patient_data.items():
        if key in FEATURE_MAP:
            normalized[FEATURE_MAP[key]] = value
        elif key in _feature_names:
            normalized[key] = value
        else:
            snake = key.lower()
            if snake in FEATURE_MAP:
                normalized[FEATURE_MAP[snake]] = value
    missing = [f for f in _feature_names if f not in normalized]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return normalized


def predict(patient_data: dict) -> dict:
    if _pipeline is None:
        load_model()

    normalized = _normalize_input(patient_data)
    row = pd.DataFrame([normalized])[_feature_names]
    risk_score = float(_pipeline.predict_proba(row)[0][1])
    risk_label = "High" if risk_score >= 0.5 else "Low"

    features_step = _pipeline.named_steps["features"]
    preprocessor = _pipeline.named_steps["preprocessor"]
    classifier = _pipeline.named_steps["classifier"]
    X_eng = features_step.transform(row)
    X_transformed = preprocessor.transform(X_eng)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)[0]

    pairs = list(zip(_shap_feature_names, shap_values.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top_risk = [[name, round(val, 4)] for name, val in pairs if val > 0][:5]
    top_protective = [[name, round(val, 4)] for name, val in reversed(pairs) if val < 0][:5]
    all_shap = {name: round(val, 4) for name, val in pairs}

    return {
        "risk_score": round(risk_score, 4),
        "risk_label": risk_label,
        "shap_explanation": {
            "top_risk_factors": top_risk,
            "top_protective_factors": top_protective,
            "all_shap": all_shap,
        },
        "features": {MODEL_TO_SNAKE.get(k, k): normalized[k] for k in _feature_names},
    }
