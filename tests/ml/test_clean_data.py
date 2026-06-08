import pandas as pd
import pytest

from clean_data import clean_data, load_raw_data, run
from cleaning_config import DROP_COLUMNS, TARGET_COLUMN, BINARY_COLUMNS


def test_clean_data_produces_valid_output(raw_data_path, cleaned_data_path, tmp_path):
  report = run(input_path=raw_data_path, output_path=tmp_path / "cleaned.csv")
  df = pd.read_csv(tmp_path / "cleaned.csv")

  assert report["rows_out"] > 2000
  assert report["rows_out"] <= report["rows_in"]
  assert not df.duplicated().any()
  assert df.isnull().sum().sum() == 0
  assert set(DROP_COLUMNS).isdisjoint(df.columns)
  assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})

  for col in BINARY_COLUMNS:
    assert set(df[col].unique()).issubset({0, 1})


def test_clean_data_idempotent(raw_data_path):
  df = load_raw_data(raw_data_path)
  cleaned, report1 = clean_data(df)
  doubled = pd.concat([cleaned, cleaned], ignore_index=True)
  cleaned2, report2 = clean_data(doubled)
  assert report2["dropped_duplicates"] == len(cleaned)
  assert len(cleaned2) == report1["rows_out"]
