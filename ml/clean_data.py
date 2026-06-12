"""Clean raw Alzheimer's dataset before model training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from cleaning_config import (
    BINARY_COLUMNS,
    CLEANED_DATA_PATH,
    CONTINUOUS_RANGES,
    DROP_COLUMNS,
    ORDINAL_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def load_raw_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _count_step(df: pd.DataFrame, mask: pd.Series, report: dict, key: str) -> pd.DataFrame:
    dropped = int(mask.sum())
    report[key] = dropped
    return df.loc[~mask].copy()


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report: dict = {"rows_in": len(df)}

    # Exact duplicate rows
    dup_mask = df.duplicated(keep="first")
    df = _count_step(df, dup_mask, report, "dropped_duplicates")

    # Duplicate PatientID
    if "PatientID" in df.columns:
        pid_mask = df.duplicated(subset=["PatientID"], keep="first")
        df = _count_step(df, pid_mask, report, "dropped_duplicate_patient_id")

    # Missing values in features or target
    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]
    null_mask = df[feature_cols].isnull().any(axis=1)
    df = _count_step(df, null_mask, report, "dropped_nulls")

    # Invalid target
    invalid_target = ~df[TARGET_COLUMN].isin([0, 1])
    df = _count_step(df, invalid_target, report, "dropped_invalid_target")

    # Binary columns must be 0 or 1
    binary_invalid = pd.Series(False, index=df.index)
    for col in BINARY_COLUMNS:
        if col in df.columns:
            binary_invalid |= ~df[col].isin([0, 1])
    df = _count_step(df, binary_invalid, report, "dropped_invalid_binary")

    # Ordinal range checks
    ordinal_invalid = pd.Series(False, index=df.index)
    for col, (lo, hi) in ORDINAL_COLUMNS.items():
        if col in df.columns:
            ordinal_invalid |= (df[col] < lo) | (df[col] > hi)
    df = _count_step(df, ordinal_invalid, report, "dropped_invalid_ordinal")

    # Continuous range checks
    range_invalid = pd.Series(False, index=df.index)
    for col, (lo, hi) in CONTINUOUS_RANGES.items():
        if col in df.columns:
            range_invalid |= (df[col] < lo) | (df[col] > hi)
    df = _count_step(df, range_invalid, report, "dropped_out_of_range")

    # Drop masked / ID columns
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    report["rows_out"] = len(df)
    report["rows_dropped"] = report["rows_in"] - report["rows_out"]
    return df.reset_index(drop=True), report


def save_cleaned(df: pd.DataFrame, path: Path | str = CLEANED_DATA_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run(
    input_path: Path | str = RAW_DATA_PATH,
    output_path: Path | str = CLEANED_DATA_PATH,
) -> dict:
    from data_pipeline import REPORT_CACHE, run_pre_scientist_pipeline

    pipeline = run_pre_scientist_pipeline(input_path=input_path, output_path=output_path)
    report = json.loads(REPORT_CACHE.read_text(encoding="utf-8"))
    report["pipeline"] = pipeline
    return report


def main() -> None:
    report = run()
    print("Data cleaning complete:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
