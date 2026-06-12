"""Train XGBoost classifier on cleaned Alzheimer's dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cleaning_config import CLEANED_DATA_PATH
from dataset_contract import CONTRACT_PATH, build_contract, save_contract, validate_training_frame

ARTIFACTS_DIR = Path(__file__).resolve().parent / "model_artifacts"
MODEL_PATH = ARTIFACTS_DIR / "xgb_model.pkl"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"

TARGET_COLUMN = "Diagnosis"

CONTINUOUS_COLUMNS = [
    "Age",
    "BMI",
    "AlcoholConsumption",
    "PhysicalActivity",
    "DietQuality",
    "SleepQuality",
    "SystolicBP",
    "DiastolicBP",
    "CholesterolTotal",
    "CholesterolLDL",
    "CholesterolHDL",
    "CholesterolTriglycerides",
    "MMSE",
    "FunctionalAssessment",
    "ADL",
]

PASSTHROUGH_COLUMNS = [
    "Gender",
    "Ethnicity",
    "EducationLevel",
    "Smoking",
    "FamilyHistoryAlzheimers",
    "CardiovascularDisease",
    "Diabetes",
    "Depression",
    "HeadInjury",
    "Hypertension",
    "MemoryComplaints",
    "BehavioralProblems",
    "Confusion",
    "Disorientation",
    "PersonalityChanges",
    "DifficultyCompletingTasks",
    "Forgetfulness",
]


def load_cleaned_data(path: Path = CLEANED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {path}. Run: python ml/clean_data.py"
        )
    return pd.read_csv(path)


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CONTINUOUS_COLUMNS),
            ("pass", "passthrough", PASSTHROUGH_COLUMNS),
        ]
    )
    classifier = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_and_save(data_path: Path = CLEANED_DATA_PATH) -> dict:
    df = load_cleaned_data(data_path)
    feature_columns = CONTINUOUS_COLUMNS + PASSTHROUGH_COLUMNS
    validate_training_frame(df, feature_columns)
    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    scale_pos_weight = neg / pos if pos else 1.0

    pipeline = build_pipeline(scale_pos_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba = cross_val_predict(
        pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    cv_auc = roc_auc_score(y, y_proba)
    print(f"5-fold CV AUC: {cv_auc:.4f}")

    pipeline.fit(X, y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(pipeline, MODEL_PATH)
    FEATURE_NAMES_PATH.write_text(json.dumps(feature_columns, indent=2))

    if CONTRACT_PATH.exists():
        cleaning_report = {}
        cache = Path(__file__).resolve().parent / "last_cleaning_report.json"
        if cache.exists():
            cleaning_report = json.loads(cache.read_text(encoding="utf-8"))
        save_contract(
            build_contract(
                cleaning_report=cleaning_report,
                row_count=len(df),
                model_features=feature_columns,
            )
        )

    metrics = {
        "cv_auc": round(cv_auc, 4),
        "rows": len(df),
        "features": len(feature_columns),
        "scale_pos_weight": round(scale_pos_weight, 4),
        "class_0": neg,
        "class_1": pos,
    }
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    train_and_save()


if __name__ == "__main__":
    main()
