from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from config import FEATURE_NAMES_PATH, MODEL_PATH

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


def load_model() -> None:
    global _pipeline, _feature_names
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    _pipeline = joblib.load(MODEL_PATH)
    _feature_names = json.loads(FEATURE_NAMES_PATH.read_text())


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


def _build_shap_explanation(shap_values: np.ndarray) -> dict:
    pairs = list(zip(_feature_names, shap_values.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top_risk = [[name, round(val, 4)] for name, val in pairs if val > 0][:5]
    top_protective = [[name, round(val, 4)] for name, val in reversed(pairs) if val < 0][:5]
    all_shap = {name: round(val, 4) for name, val in pairs}
    return {
        "top_risk_factors": top_risk,
        "top_protective_factors": top_protective,
        "all_shap": all_shap,
    }


def predict(patient_data: dict) -> dict:
    if _pipeline is None:
        load_model()

    normalized = _normalize_input(patient_data)
    row = pd.DataFrame([normalized])[_feature_names]
    risk_score = float(_pipeline.predict_proba(row)[0][1])
    risk_label = "High" if risk_score >= 0.5 else "Low"

    preprocessor = _pipeline.named_steps["preprocessor"]
    classifier = _pipeline.named_steps["classifier"]
    X_transformed = preprocessor.transform(row)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)[0]

    return {
        "risk_score": round(risk_score, 4),
        "risk_label": risk_label,
        "shap_explanation": _build_shap_explanation(shap_values),
        "features": {MODEL_TO_SNAKE.get(k, k): normalized[k] for k in _feature_names},
    }
