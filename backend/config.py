import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Supabase project: https://supabase.com/dashboard/project/puhtwpurgkqylnanortj
SUPABASE_PROJECT_ID = "puhtwpurgkqylnanortj"
SUPABASE_URL = os.environ.get(
    "SUPABASE_URL",
    f"https://{SUPABASE_PROJECT_ID}.supabase.co",
)
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Claude model for AI recommendations + chat (https://platform.claude.com/docs)
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

def _resolve_project_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


MODEL_PATH = _resolve_project_path(
    os.environ.get("MODEL_PATH", "ml/model_artifacts/model.pkl")
)
FEATURE_NAMES_PATH = _resolve_project_path(
    os.environ.get("FEATURE_NAMES_PATH", "ml/model_artifacts/feature_names.json")
)
SHAP_FEATURE_NAMES_PATH = _resolve_project_path(
    os.environ.get("SHAP_FEATURE_NAMES_PATH", "ml/model_artifacts/shap_feature_names.json")
)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
