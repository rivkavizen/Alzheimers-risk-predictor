import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

MODEL_PATH = Path(
    os.environ.get("MODEL_PATH", PROJECT_ROOT / "ml" / "model_artifacts" / "xgb_model.pkl")
)
FEATURE_NAMES_PATH = Path(
    os.environ.get(
        "FEATURE_NAMES_PATH",
        PROJECT_ROOT / "ml" / "model_artifacts" / "feature_names.json",
    )
)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
