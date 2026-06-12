"""Orchestrate pre-modeling pipeline: Engineer → Analyst → Insights Analyst."""

from __future__ import annotations

import json
from pathlib import Path

from clean_data import clean_data, load_raw_data, save_cleaned
from cleaning_config import CLEANED_DATA_PATH, PROJECT_ROOT, RAW_DATA_PATH
from dataset_contract import CONTRACT_PATH, build_contract, save_contract
from eda import EDA_HTML_PATH, INSIGHTS_PATH, run_eda_report, run_insights

REPORT_CACHE = PROJECT_ROOT / "ml" / "last_cleaning_report.json"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_cleaning_report() -> dict | None:
    if not REPORT_CACHE.exists():
        return None
    return json.loads(REPORT_CACHE.read_text(encoding="utf-8"))


def run_engineer(
    input_path: Path | str = RAW_DATA_PATH,
    output_path: Path | str = CLEANED_DATA_PATH,
    contract_path: Path | str = CONTRACT_PATH,
) -> dict:
    """Data Engineer: clean raw data and publish dataset_contract.json."""
    df = load_raw_data(input_path)
    cleaned, report = clean_data(df)
    save_cleaned(cleaned, output_path)
    report["output_path"] = str(output_path)

    REPORT_CACHE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    contract = build_contract(cleaning_report=report, row_count=len(cleaned))
    save_contract(contract, contract_path)

    return {
        "agent": "Data Engineer",
        "cleaned_data": _rel(Path(output_path)),
        "dataset_contract": _rel(Path(contract_path)),
        "rows_in": report["rows_in"],
        "rows_out": report["rows_out"],
        "rows_dropped": report["rows_dropped"],
    }


def run_analyst(
    data_path: Path | str = CLEANED_DATA_PATH,
    html_path: Path | str = EDA_HTML_PATH,
) -> dict:
    """Data Analyst: produce eda_report.html from cleaned data."""
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Cleaned data not found at {data_path}. Run engineer step first.")
    result = run_eda_report(data_path=data_path, html_path=html_path)
    result["agent"] = "Data Analyst"
    return result


def run_insights_analyst(
    data_path: Path | str = CLEANED_DATA_PATH,
    insights_path: Path | str = INSIGHTS_PATH,
    cleaning_report: dict | None = None,
) -> dict:
    """Insights Analyst: produce insights.md for business stakeholders."""
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Cleaned data not found at {data_path}. Run engineer step first.")
    cleaning_report = cleaning_report or _load_cleaning_report()
    result = run_insights(
        data_path=data_path,
        insights_path=insights_path,
        cleaning_report=cleaning_report,
    )
    result["agent"] = "Insights Analyst"
    return result


def run_pre_scientist_pipeline(
    input_path: Path | str = RAW_DATA_PATH,
    output_path: Path | str = CLEANED_DATA_PATH,
) -> dict:
    """Run all three agents before the Data Scientist."""
    engineer = run_engineer(input_path=input_path, output_path=output_path)
    analyst = run_analyst(data_path=output_path)
    insights = run_insights_analyst(data_path=output_path)
    return {
        "engineer": engineer,
        "analyst": analyst,
        "insights_analyst": insights,
    }


def main() -> None:
    result = run_pre_scientist_pipeline()
    print("Pre-scientist pipeline complete:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
