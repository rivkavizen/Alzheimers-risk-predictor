import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from services.api_key_utils import validate_key_format
from services.validation import validate_patient_data


def test_reject_supabase_key_format():
    assert "Supabase" in validate_key_format("sb_secret_abc123")


def test_accept_anthropic_key_format():
    assert validate_key_format("sk-ant-api03-testkey123456789") is None


def test_validate_patient_data_range():
    data = {k: 0 for k in [
        "age", "gender", "ethnicity", "education_level", "bmi", "smoking",
        "alcohol_consumption", "physical_activity", "diet_quality", "sleep_quality",
        "family_history_alzheimers", "cardiovascular_disease", "diabetes", "depression",
        "head_injury", "hypertension", "systolic_bp", "diastolic_bp",
        "cholesterol_total", "cholesterol_ldl", "cholesterol_hdl", "cholesterol_triglycerides",
        "mmse", "functional_assessment", "memory_complaints", "behavioral_problems",
        "adl", "confusion", "disorientation", "personality_changes",
        "difficulty_completing_tasks", "forgetfulness",
    ]}
    data["age"] = 40
    errors = validate_patient_data(data)
    assert any("age" in e for e in errors)
