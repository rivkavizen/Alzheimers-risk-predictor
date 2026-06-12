import json
from pathlib import Path

import pytest

from clean_data import run
from data_pipeline import run_analyst, run_engineer, run_insights_analyst, run_pre_scientist_pipeline
from dataset_contract import CONTRACT_PATH, load_contract
from eda import run_eda, run_eda_report, run_insights


def test_run_eda_report_produces_html(cleaned_data_path, tmp_path):
    html = tmp_path / "eda_report.html"
    result = run_eda_report(data_path=cleaned_data_path, html_path=html)

    assert html.exists()
    assert html.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    assert "eda_report.html" in result["eda_report"]


def test_run_insights_produces_markdown(cleaned_data_path, tmp_path):
    insights = tmp_path / "insights.md"
    result = run_insights(
        data_path=cleaned_data_path,
        insights_path=insights,
        cleaning_report={"rows_in": 2200, "rows_out": 2149, "rows_dropped": 51},
    )

    assert insights.exists()
    assert "# Alzheimer's Dataset" in insights.read_text(encoding="utf-8")
    assert "insights.md" in result["insights"]


def test_run_eda_produces_both_deliverables(cleaned_data_path, tmp_path):
    html = tmp_path / "eda_report.html"
    insights = tmp_path / "insights.md"

    result = run_eda(
        data_path=cleaned_data_path,
        cleaning_report={"rows_in": 2200, "rows_out": 2149},
        html_path=html,
        insights_path=insights,
    )

    assert html.exists()
    assert insights.exists()
    assert "eda_report" in result
    assert "insights" in result


def test_engineer_publishes_contract(raw_data_path, tmp_path):
    out = tmp_path / "cleaned.csv"
    contract = tmp_path / "dataset_contract.json"
    result = run_engineer(
        input_path=raw_data_path,
        output_path=out,
        contract_path=contract,
    )

    assert out.exists()
    assert contract.exists()
    assert result["agent"] == "Data Engineer"
    data = json.loads(contract.read_text(encoding="utf-8"))
    assert "schema" in data
    assert "assumptions" in data
    assert "constraints" in data


def test_pre_scientist_pipeline_three_agents(raw_data_path, tmp_path):
    out = tmp_path / "cleaned.csv"
    result = run_pre_scientist_pipeline(input_path=raw_data_path, output_path=out)

    assert result["engineer"]["agent"] == "Data Engineer"
    assert result["analyst"]["agent"] == "Data Analyst"
    assert result["insights_analyst"]["agent"] == "Insights Analyst"
    assert out.exists()


def test_clean_data_run_includes_pipeline(raw_data_path, tmp_path):
    out = tmp_path / "cleaned.csv"
    report = run(input_path=raw_data_path, output_path=out)

    assert "pipeline" in report
    pipeline = report["pipeline"]
    assert "engineer" in pipeline
    assert "analyst" in pipeline
    assert "insights_analyst" in pipeline
    assert report["rows_out"] > 2000


def test_contract_schema_matches_features():
    if not CONTRACT_PATH.exists():
        pytest.skip("Run ml/clean_data.py to generate contract")

    contract = load_contract()
    schema_keys = {k for k in contract["schema"] if k != "Diagnosis"}
    model_features = set(contract.get("model_features") or [])
    if model_features:
        assert model_features == schema_keys
