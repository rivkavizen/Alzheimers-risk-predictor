"""Build dataset_contract.json for the Data Scientist crew."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from cleaning_config import (
    BINARY_COLUMNS,
    CLEANED_DATA_PATH,
    CONTINUOUS_RANGES,
    DROP_COLUMNS,
    ORDINAL_COLUMNS,
    PROJECT_ROOT,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)

CONTRACT_PATH = PROJECT_ROOT / "ml" / "dataset_contract.json"
CONTRACT_VERSION = "1.0"

# Model PascalCase -> API snake_case (aligned with backend/services/ml_service.py)
API_FIELD_MAP = {
    "Age": "age",
    "Gender": "gender",
    "Ethnicity": "ethnicity",
    "EducationLevel": "education_level",
    "BMI": "bmi",
    "Smoking": "smoking",
    "AlcoholConsumption": "alcohol_consumption",
    "PhysicalActivity": "physical_activity",
    "DietQuality": "diet_quality",
    "SleepQuality": "sleep_quality",
    "FamilyHistoryAlzheimers": "family_history_alzheimers",
    "CardiovascularDisease": "cardiovascular_disease",
    "Diabetes": "diabetes",
    "Depression": "depression",
    "HeadInjury": "head_injury",
    "Hypertension": "hypertension",
    "SystolicBP": "systolic_bp",
    "DiastolicBP": "diastolic_bp",
    "CholesterolTotal": "cholesterol_total",
    "CholesterolLDL": "cholesterol_ldl",
    "CholesterolHDL": "cholesterol_hdl",
    "CholesterolTriglycerides": "cholesterol_triglycerides",
    "MMSE": "mmse",
    "FunctionalAssessment": "functional_assessment",
    "MemoryComplaints": "memory_complaints",
    "BehavioralProblems": "behavioral_problems",
    "ADL": "adl",
    "Confusion": "confusion",
    "Disorientation": "disorientation",
    "PersonalityChanges": "personality_changes",
    "DifficultyCompletingTasks": "difficulty_completing_tasks",
    "Forgetfulness": "forgetfulness",
}

ASSUMPTIONS = [
    "Each row represents one patient record with a binary Alzheimer's diagnosis label.",
    "Cleaned data excludes identifier and free-text columns (PatientID, DoctorInCharge).",
    "Allowed value ranges reflect clinical plausibility filters applied during cleaning.",
    "Ordinal fields (Ethnicity, EducationLevel) are coded integers, not raw categories.",
    "Binary symptom and lifestyle fields are encoded as 0 (no/absent) or 1 (yes/present).",
    "Training uses only features listed in model_features; Diagnosis is the sole target.",
    "API submissions use snake_case field names mapped to model PascalCase columns.",
]

CONSTRAINTS = [
    "No null values in any feature or target column after cleaning.",
    "Binary columns must equal exactly 0 or 1.",
    "Ordinal columns must be integers within their documented min–max range.",
    "Continuous columns must fall within documented min–max bounds.",
    "Target Diagnosis must be 0 or 1 only.",
    "PatientID and DoctorInCharge must never be used as model inputs.",
    "Feature order at inference must match ml/model_artifacts/feature_names.json.",
    "Retraining must read from alzheimers_disease_data_cleaned.csv, not the raw file.",
]


def _schema_entry(column: str) -> dict:
    api_field = API_FIELD_MAP.get(column, column)
    if column in BINARY_COLUMNS:
        return {
            "type": "binary",
            "dtype": "integer",
            "api_field": api_field,
            "allowed_values": [0, 1],
            "nullable": False,
        }
    if column in ORDINAL_COLUMNS:
        lo, hi = ORDINAL_COLUMNS[column]
        return {
            "type": "ordinal",
            "dtype": "integer",
            "api_field": api_field,
            "min": lo,
            "max": hi,
            "nullable": False,
        }
    if column in CONTINUOUS_RANGES:
        lo, hi = CONTINUOUS_RANGES[column]
        return {
            "type": "continuous",
            "dtype": "float",
            "api_field": api_field,
            "min": lo,
            "max": hi,
            "nullable": False,
        }
    if column == TARGET_COLUMN:
        return {
            "type": "binary",
            "dtype": "integer",
            "api_field": "diagnosis",
            "allowed_values": [0, 1],
            "labels": {"0": "no_diagnosis", "1": "diagnosis"},
            "nullable": False,
        }
    return {"type": "unknown", "api_field": api_field, "nullable": False}


def build_contract(
    cleaning_report: dict | None = None,
    row_count: int | None = None,
    model_features: list[str] | None = None,
) -> dict:
    feature_names_path = PROJECT_ROOT / "ml" / "model_artifacts" / "feature_names.json"
    if model_features is None and feature_names_path.exists():
        model_features = json.loads(feature_names_path.read_text(encoding="utf-8"))

    schema_columns = (
        list(CONTINUOUS_RANGES)
        + list(ORDINAL_COLUMNS)
        + BINARY_COLUMNS
        + [TARGET_COLUMN]
    )
    schema = {col: _schema_entry(col) for col in schema_columns}

    return {
        "version": CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Alzheimer's Disease Risk Dataset",
            "raw_source": str(RAW_DATA_PATH.relative_to(PROJECT_ROOT)),
            "cleaned_source": str(CLEANED_DATA_PATH.relative_to(PROJECT_ROOT)),
            "target_column": TARGET_COLUMN,
            "dropped_columns": DROP_COLUMNS,
            "row_count": row_count,
            "cleaning_report": cleaning_report or {},
        },
        "schema": schema,
        "allowed_values": {
            "binary": [0, 1],
            "ordinal": ORDINAL_COLUMNS,
            "continuous": CONTINUOUS_RANGES,
            "target": [0, 1],
        },
        "assumptions": ASSUMPTIONS,
        "constraints": CONSTRAINTS,
        "model_features": model_features or [],
        "for_data_scientist": {
            "must_read_before_training": True,
            "train_script": "ml/train.py",
            "pipeline_script": "ml/data_pipeline.py",
            "pre_scientist_agents": [
                "Data Engineer",
                "Data Analyst",
                "Insights Analyst",
            ],
            "required_artifacts": [
                "ml/model_artifacts/features.csv",
                "ml/model_artifacts/model.pkl",
                "ml/model_artifacts/evaluation_report.md",
                "ml/model_artifacts/model_card.md",
                "ml/model_artifacts/feature_names.json",
                "ml/model_artifacts/shap_feature_names.json",
            ],
            "forbidden_columns": DROP_COLUMNS,
            "validation_reference": "backend/services/validation.py",
        },
    }


def save_contract(
    contract: dict,
    path: Path | str = CONTRACT_PATH,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return path


def load_contract(path: Path | str = CONTRACT_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_training_frame(df, feature_columns: list[str]) -> None:
    """Raise ValueError if training data violates the dataset contract."""
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CONTRACT_PATH.name}. Run: python ml/clean_data.py"
        )

    contract = load_contract()
    forbidden = set(contract["for_data_scientist"]["forbidden_columns"])
    present_forbidden = forbidden & set(df.columns)
    if present_forbidden:
        raise ValueError(f"Forbidden columns in training data: {sorted(present_forbidden)}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column {TARGET_COLUMN} missing from training data")

    schema = contract["schema"]
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"Contract feature {col} missing from training data")
        if col not in schema:
            raise ValueError(f"No schema entry for feature {col} in contract")

    stored_features = contract.get("model_features") or []
    if stored_features and list(stored_features) != list(feature_columns):
        raise ValueError(
            "Feature column order differs from dataset_contract.json model_features"
        )

    if df[feature_columns + [TARGET_COLUMN]].isnull().any().any():
        raise ValueError("Null values found in training features or target")
