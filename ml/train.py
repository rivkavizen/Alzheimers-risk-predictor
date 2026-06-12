"""Train Alzheimer's risk model: validate, engineer features, compare models, report."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cleaning_config import CLEANED_DATA_PATH, TARGET_COLUMN
from dataset_contract import CONTRACT_PATH, build_contract, load_contract, save_contract, validate_training_frame
from feature_engineering import (
    BASE_COLUMNS,
    ENGINEERED_COLUMNS,
    FEATURE_COLUMNS,
    PASSTHROUGH_COLUMNS,
    SCALE_CONTINUOUS,
    FeatureEngineer,
    save_features_csv,
)
from model_reports import build_evaluation_report, build_model_card, save_markdown

ARTIFACTS_DIR = Path(__file__).resolve().parent / "model_artifacts"
FEATURES_CSV_PATH = ARTIFACTS_DIR / "features.csv"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
MODEL_PATH_LEGACY = ARTIFACTS_DIR / "xgb_model.pkl"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"
EVALUATION_REPORT_PATH = ARTIFACTS_DIR / "evaluation_report.md"
MODEL_CARD_PATH = ARTIFACTS_DIR / "model_card.md"

# Re-export for evaluate.py / tests
CONTINUOUS_COLUMNS = SCALE_CONTINUOUS


def load_cleaned_data(path: Path = CLEANED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {path}. Run: python ml/clean_data.py"
        )
    return pd.read_csv(path)


def validate_dataset_and_contract(df: pd.DataFrame) -> dict:
    """Step 1: read cleaned data and validate against dataset_contract.json."""
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CONTRACT_PATH}. Run: python ml/clean_data.py"
        )
    contract = load_contract()
    validate_training_frame(df, BASE_COLUMNS)
    return contract


def build_model_pipeline(classifier, scale_continuous: list[str], passthrough: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), scale_continuous),
            ("pass", "passthrough", passthrough),
        ]
    )
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def _cv_metrics(pipeline, X: pd.DataFrame, y: pd.Series, cv) -> dict:
    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "cv_auc": float(roc_auc_score(y, y_proba)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }


def compare_models(X: pd.DataFrame, y: pd.Series, scale_pos_weight: float, cv) -> list[dict]:
    """Step 4: evaluate at least two model variations."""
    from feature_engineering import CONTINUOUS_COLUMNS as BASE_CONTINUOUS

    scale_continuous = BASE_CONTINUOUS + ENGINEERED_COLUMNS
    passthrough = PASSTHROUGH_COLUMNS

    candidates = [
        {
            "name": "XGBoost (tuned)",
            "rationale": "Gradient boosting with class-weighting; strong tabular performance.",
            "pipeline": build_model_pipeline(
                xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
                scale_continuous,
                passthrough,
            ),
        },
        {
            "name": "XGBoost (shallow)",
            "rationale": "Lower capacity variant to test overfitting and simpler decision boundaries.",
            "pipeline": build_model_pipeline(
                xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
                scale_continuous,
                passthrough,
            ),
        },
        {
            "name": "Random Forest",
            "rationale": "Bagged trees baseline with different inductive bias than boosting.",
            "pipeline": build_model_pipeline(
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                scale_continuous,
                passthrough,
            ),
        },
    ]

    results = []
    for spec in candidates:
        metrics = _cv_metrics(spec["pipeline"], X, y, cv)
        results.append({"name": spec["name"], "rationale": spec["rationale"], **metrics, "pipeline": spec["pipeline"]})
    results.sort(key=lambda r: r["cv_auc"], reverse=True)
    return results


def train_and_save(data_path: Path = CLEANED_DATA_PATH) -> dict:
    df = load_cleaned_data(data_path)

    # 1. Validate cleaned dataset + contract
    validate_dataset_and_contract(df)

    # 2. Feature engineering → features.csv
    save_features_csv(df, FEATURES_CSV_PATH, include_target=True)

    X = df[BASE_COLUMNS]
    y = df[TARGET_COLUMN]
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    scale_pos_weight = neg / pos if pos else 1.0

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3–4. Train candidates and compare
    comparison = compare_models(X, y, scale_pos_weight, cv)
    winner = comparison[0]
    best_pipeline = winner.pop("pipeline")

    print(f"Best model: {winner['name']} — CV AUC {winner['cv_auc']:.4f}")

    # Fit selected model on full training set
    best_pipeline.fit(X, y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    joblib.dump(best_pipeline, MODEL_PATH_LEGACY)
    FEATURE_NAMES_PATH.write_text(json.dumps(BASE_COLUMNS, indent=2))
    (ARTIFACTS_DIR / "shap_feature_names.json").write_text(
        json.dumps(FEATURE_COLUMNS, indent=2)
    )

    training_summary = {
        "rows": len(df),
        "base_features": len(BASE_COLUMNS),
        "engineered_features": len(ENGINEERED_COLUMNS),
        "total_features": len(FEATURE_COLUMNS),
        "positive_rate": float(y.mean()),
        "class_0": neg,
        "class_1": pos,
        "scale_pos_weight": round(scale_pos_weight, 4),
    }

    # 5. Reports
    comparison_public = [{k: v for k, v in row.items() if k != "pipeline"} for row in comparison]
    save_markdown(
        build_evaluation_report(comparison_public, winner, training_summary),
        EVALUATION_REPORT_PATH,
    )
    save_markdown(
        build_model_card(winner, training_summary, comparison_public),
        MODEL_CARD_PATH,
    )

    if CONTRACT_PATH.exists():
        cleaning_report = {}
        cache = Path(__file__).resolve().parent / "last_cleaning_report.json"
        if cache.exists():
            cleaning_report = json.loads(cache.read_text(encoding="utf-8"))
        save_contract(
            build_contract(
                cleaning_report=cleaning_report,
                row_count=len(df),
                model_features=BASE_COLUMNS,
            )
        )

    metrics = {
        **winner,
        **training_summary,
        "selected_model": winner["name"],
        "artifacts": {
            "features_csv": str(FEATURES_CSV_PATH.name),
            "model_pkl": str(MODEL_PATH.name),
            "evaluation_report": str(EVALUATION_REPORT_PATH.name),
            "model_card": str(MODEL_CARD_PATH.name),
        },
        "model_comparison": comparison_public,
    }
    print(json.dumps({k: v for k, v in metrics.items() if k != "model_comparison"}, indent=2))
    return metrics


def main() -> None:
    train_and_save()


if __name__ == "__main__":
    main()
