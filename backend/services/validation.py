"""Validate assessment input (aligned with ml/cleaning_config.py)."""

from services.ml_service import FEATURE_MAP

BINARY_FIELDS = [
    "gender", "smoking", "family_history_alzheimers", "cardiovascular_disease",
    "diabetes", "depression", "head_injury", "hypertension", "memory_complaints",
    "behavioral_problems", "confusion", "disorientation", "personality_changes",
    "difficulty_completing_tasks", "forgetfulness",
]

ORDINAL_FIELDS = {"ethnicity": (0, 3), "education_level": (0, 3)}

CONTINUOUS_FIELDS = {
    "age": (50, 100),
    "bmi": (10, 60),
    "alcohol_consumption": (0, 30),
    "physical_activity": (0, 15),
    "diet_quality": (0, 10),
    "sleep_quality": (0, 10),
    "systolic_bp": (60, 200),
    "diastolic_bp": (40, 130),
    "cholesterol_total": (50, 400),
    "cholesterol_ldl": (20, 300),
    "cholesterol_hdl": (10, 120),
    "cholesterol_triglycerides": (30, 500),
    "mmse": (0, 30),
    "functional_assessment": (0, 10),
    "adl": (0, 10),
}


def validate_patient_data(data: dict) -> list[str]:
    errors = []

    for field in FEATURE_MAP:
        if field not in data:
            errors.append(f"{field}: required")
            continue
        try:
            value = float(data[field])
        except (TypeError, ValueError):
            errors.append(f"{field}: must be numeric")
            continue

        if field in BINARY_FIELDS:
            if value not in (0, 1):
                errors.append(f"{field}: must be 0 or 1")
        elif field in ORDINAL_FIELDS:
            lo, hi = ORDINAL_FIELDS[field]
            if value < lo or value > hi or value != int(value):
                errors.append(f"{field}: must be integer {lo}–{hi}")
        elif field in CONTINUOUS_FIELDS:
            lo, hi = CONTINUOUS_FIELDS[field]
            if value < lo or value > hi:
                errors.append(f"{field}: must be between {lo} and {hi}")

    return errors
