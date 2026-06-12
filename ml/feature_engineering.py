"""Feature engineering for Alzheimer's risk modeling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from cleaning_config import TARGET_COLUMN

# Base model inputs (from cleaned dataset, per dataset contract)
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

BASE_COLUMNS = CONTINUOUS_COLUMNS + PASSTHROUGH_COLUMNS

SYMPTOM_COLUMNS = [
    "MemoryComplaints",
    "BehavioralProblems",
    "Confusion",
    "Disorientation",
    "PersonalityChanges",
    "DifficultyCompletingTasks",
    "Forgetfulness",
]

ENGINEERED_COLUMNS = [
    "PulsePressure",
    "CholesterolRatio",
    "SymptomBurden",
    "LifestyleScore",
]

FEATURE_COLUMNS = BASE_COLUMNS + ENGINEERED_COLUMNS

SCALE_CONTINUOUS = CONTINUOUS_COLUMNS + ENGINEERED_COLUMNS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features from base clinical columns."""
    out = df[BASE_COLUMNS].copy()
    out["PulsePressure"] = out["SystolicBP"] - out["DiastolicBP"]
    out["CholesterolRatio"] = out["CholesterolTotal"] / out["CholesterolHDL"].clip(lower=1)
    out["SymptomBurden"] = out[SYMPTOM_COLUMNS].sum(axis=1)
    out["LifestyleScore"] = (
        out["PhysicalActivity"] + out["DietQuality"] + out["SleepQuality"]
    ) / 3.0 - out["Smoking"]
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn transformer wrapping engineer_features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return engineer_features(X)
        return engineer_features(pd.DataFrame(X, columns=BASE_COLUMNS))


def build_feature_matrix(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Engineered features; optionally attach target column."""
    features = engineer_features(df)
    if include_target and TARGET_COLUMN in df.columns:
        features[TARGET_COLUMN] = df[TARGET_COLUMN].values
    return features


def save_features_csv(
    df: pd.DataFrame,
    path: Path | str,
    include_target: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_feature_matrix(df, include_target=include_target).to_csv(path, index=False)
    return path
