"""Exploratory data analysis: eda_report.html and insights.md."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cleaning_config import CLEANED_DATA_PATH, PROJECT_ROOT, TARGET_COLUMN
EDA_HTML_PATH = PROJECT_ROOT / "ml" / "eda_report.html"
INSIGHTS_PATH = PROJECT_ROOT / "ml" / "insights.md"

def load_cleaned_data(path: Path | str = CLEANED_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _plot_target_distribution(df: pd.DataFrame) -> str:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["No diagnosis (0)", "Diagnosis (1)"]
    colors = ["#4c78a8", "#e45756"]
    ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors)
    ax.set_title("Diagnosis class balance")
    ax.set_ylabel("Patient count")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v + 15, f"{v:,}", ha="center", fontsize=10)
    return _fig_to_base64(fig)


def _plot_age_by_diagnosis(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in [(0, "#4c78a8"), (1, "#e45756")]:
        subset = df.loc[df[TARGET_COLUMN] == label, "Age"]
        ax.hist(subset, bins=20, alpha=0.65, label=f"Diagnosis={label}", color=color)
    ax.set_title("Age distribution by diagnosis")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.legend()
    return _fig_to_base64(fig)


def _plot_mmse_by_diagnosis(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [
        df.loc[df[TARGET_COLUMN] == 0, "MMSE"],
        df.loc[df[TARGET_COLUMN] == 1, "MMSE"],
    ]
    ax.boxplot(data, tick_labels=["No diagnosis", "Diagnosis"])
    ax.set_title("MMSE score by diagnosis")
    ax.set_ylabel("MMSE (0–30)")
    return _fig_to_base64(fig)


def _plot_correlation_heatmap(df: pd.DataFrame) -> str:
    numeric = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    corr = numeric.corrwith(df[TARGET_COLUMN]).sort_values(key=abs, ascending=False)
    top = corr.head(12).index.tolist()
    matrix = df[top + [TARGET_COLUMN]].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation heatmap — top features vs diagnosis")
    return _fig_to_base64(fig)


def _plot_lifestyle_rates(df: pd.DataFrame) -> str:
    lifestyle = ["Smoking", "PhysicalActivity", "DietQuality", "SleepQuality"]
    rates = []
    labels = []
    for col in lifestyle:
        if col == "Smoking":
            rates.append(df[col].mean() * 100)
            labels.append("Smoking %")
        else:
            rates.append(df[col].mean())
            labels.append(f"{col} (mean)")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, rates, color="#72b7b2")
    ax.set_title("Lifestyle factor summary (cohort averages)")
    ax.set_xlabel("Value")
    return _fig_to_base64(fig)


def build_insights_markdown(
    df: pd.DataFrame,
    cleaning_report: dict | None = None,
) -> str:
    n = len(df)
    pos_rate = df[TARGET_COLUMN].mean()
    neg_rate = 1 - pos_rate
    mean_age = df["Age"].mean()
    mean_mmse = df["MMSE"].mean()
    mean_bmi = df["BMI"].mean()

    top_corr = (
        df.drop(columns=[TARGET_COLUMN])
        .corrwith(df[TARGET_COLUMN])
        .abs()
        .sort_values(ascending=False)
        .head(8)
    )
    corr_lines = "\n".join(
        f"- **{col}**: correlation with diagnosis = {df[col].corr(df[TARGET_COLUMN]):+.3f}"
        for col in top_corr.index
    )

    modifiable = [
        "Smoking",
        "AlcoholConsumption",
        "PhysicalActivity",
        "DietQuality",
        "SleepQuality",
        "BMI",
    ]
    mod_lines = []
    for col in modifiable:
        if col in df.columns:
            high = df[col].quantile(0.75)
            rate_high = df.loc[df[col] >= high, TARGET_COLUMN].mean()
            rate_low = df.loc[df[col] <= df[col].quantile(0.25), TARGET_COLUMN].mean()
            mod_lines.append(
                f"- **{col}**: diagnosis rate in top quartile = {rate_high:.1%} "
                f"vs bottom quartile = {rate_low:.1%}"
            )

    cleaning = cleaning_report or {}
    dropped = cleaning.get("rows_dropped", "N/A")
    rows_in = cleaning.get("rows_in", "N/A")
    rows_out = cleaning.get("rows_out", n)

    return f"""# Alzheimer's Dataset — Business & Analytical Insights

## Executive summary

This cohort contains **{n:,}** cleaned patient records with a **{pos_rate:.1%}** positive diagnosis rate
({neg_rate:.1%} negative). The dataset is suitable for supervised risk modeling with class imbalance
that should be monitored during training and evaluation.

## Dataset overview

| Metric | Value |
|--------|-------|
| Records (cleaned) | {n:,} |
| Raw rows processed | {rows_in} |
| Rows removed in cleaning | {dropped} |
| Mean age | {mean_age:.1f} years |
| Mean BMI | {mean_bmi:.1f} |
| Mean MMSE | {mean_mmse:.1f} / 30 |
| Diagnosis prevalence | {pos_rate:.1%} |

## Key visual insights

See `eda_report.html` for interactive-style charts including:

- **Class balance** — diagnosis vs no diagnosis counts
- **Age by diagnosis** — older patients show higher diagnosis density
- **MMSE by diagnosis** — lower cognitive scores align with positive cases
- **Correlation heatmap** — strongest linear relationships with diagnosis
- **Lifestyle averages** — cohort-level modifiable factor profile

## Strongest statistical signals

{corr_lines}

## Modifiable risk factor patterns

{chr(10).join(mod_lines)}

## Data quality notes

- Cleaning removed duplicate rows, invalid ranges, and nulls per `dataset_contract.json`.
- Identifier columns (`PatientID`, `DoctorInCharge`) are excluded from modeling.
- All binary flags are strictly 0/1; ordinal and continuous fields are range-validated.

## Recommendations for the Data Scientist crew

1. **Read `dataset_contract.json` before training** — schema, allowed values, and constraints are mandatory.
2. **Use stratified cross-validation** — diagnosis prevalence is ~{pos_rate:.0%}, not 50/50.
3. **Prioritize interpretability** — SHAP explanations are a product requirement; avoid black-box-only approaches.
4. **Monitor MMSE, functional, and symptom features** — they dominate correlation with diagnosis.
5. **Do not leak target** — train only on features listed in `model_features` inside the contract.
6. **Re-run `ml/eda.py` after any raw data change** — refresh HTML, insights, and contract together.

## Business implications

- **Screening focus**: Patients with low MMSE and functional scores warrant closer follow-up in the product UX.
- **Prevention messaging**: Lifestyle factors (activity, diet, sleep, smoking) show measurable prevalence gaps across risk quartiles.
- **Fairness review**: Ethnicity and education are ordinal codes — treat carefully in explanations and avoid over-claiming causality.
- **Deployment**: Inference inputs must match API snake_case fields mapped in the dataset contract.
"""


def build_eda_html(df: pd.DataFrame, charts: dict[str, str]) -> str:
    n = len(df)
    pos_rate = df[TARGET_COLUMN].mean()
    feature_count = len([c for c in df.columns if c != TARGET_COLUMN])

    chart_sections = "\n".join(
        f"""
        <section>
          <h2>{title}</h2>
          <img src="data:image/png;base64,{b64}" alt="{title}" />
        </section>"""
        for title, b64 in charts.items()
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Alzheimer's Dataset — EDA Report</title>
  <style>
    body {{ font-family:Segoe UI,Arial,sans-serif; margin:2rem; color:#1a1a1a; line-height:1.5; }}
    h1 {{ color:#2c5282; }}
    h2 {{ color:#2b6cb0; margin-top:2rem; }}
    .meta {{ background:#f7fafc; padding:1rem 1.25rem; border-radius:8px; border:1px solid #e2e8f0; }}
    section {{ margin:2rem 0; }}
    img {{ max-width:100%; border:1px solid #e2e8f0; border-radius:8px; }}
    table {{ border-collapse:collapse; margin-top:1rem; }}
    th, td {{ border:1px solid #e2e8f0; padding:0.5rem 0.75rem; text-align:left; }}
    th {{ background:#edf2f7; }}
  </style>
</head>
<body>
  <h1>Exploratory Data Analysis Report</h1>
  <div class="meta">
    <p><strong>Dataset:</strong> alzheimers_disease_data_cleaned.csv</p>
    <p><strong>Records:</strong> {n:,} &nbsp;|&nbsp; <strong>Features:</strong> {feature_count}
       &nbsp;|&nbsp; <strong>Diagnosis rate:</strong> {pos_rate:.1%}</p>
    <p><strong>Companion files:</strong> insights.md, dataset_contract.json</p>
  </div>

  <h2>Summary statistics</h2>
  <table>
    <tr><th>Column</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
    {
        "".join(
            f"<tr><td>{col}</td><td>{df[col].mean():.2f}</td>"
            f"<td>{df[col].std():.2f}</td><td>{df[col].min():.2f}</td>"
            f"<td>{df[col].max():.2f}</td></tr>"
            for col in ["Age", "BMI", "MMSE", "FunctionalAssessment", "ADL"]
            if col in df.columns
        )
    }
  </table>

  {chart_sections}

  <footer><p>Generated by ml/eda.py — Data Analyst deliverable for the Alzheimer's Risk Predictor project.</p></footer>
</body>
</html>"""


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _build_charts(df: pd.DataFrame) -> dict[str, str]:
    return {
        "Diagnosis class balance": _plot_target_distribution(df),
        "Age distribution by diagnosis": _plot_age_by_diagnosis(df),
        "MMSE by diagnosis": _plot_mmse_by_diagnosis(df),
        "Top feature correlations": _plot_correlation_heatmap(df),
        "Lifestyle factor summary": _plot_lifestyle_rates(df),
    }


def run_eda_report(
    data_path: Path | str = CLEANED_DATA_PATH,
    html_path: Path | str = EDA_HTML_PATH,
) -> dict:
    """Data Analyst deliverable: eda_report.html only."""
    df = load_cleaned_data(data_path)
    charts = _build_charts(df)
    html_path = Path(html_path)
    html_path.write_text(build_eda_html(df, charts), encoding="utf-8")
    return {
        "eda_report": _rel(html_path),
        "rows": len(df),
        "diagnosis_rate": round(float(df[TARGET_COLUMN].mean()), 4),
    }


def run_insights(
    data_path: Path | str = CLEANED_DATA_PATH,
    cleaning_report: dict | None = None,
    insights_path: Path | str = INSIGHTS_PATH,
) -> dict:
    """Insights Analyst deliverable: insights.md only."""
    df = load_cleaned_data(data_path)
    insights_path = Path(insights_path)
    insights_path.write_text(
        build_insights_markdown(df, cleaning_report),
        encoding="utf-8",
    )
    return {
        "insights": _rel(insights_path),
        "rows": len(df),
        "diagnosis_rate": round(float(df[TARGET_COLUMN].mean()), 4),
    }


def run_eda(
    data_path: Path | str = CLEANED_DATA_PATH,
    cleaning_report: dict | None = None,
    html_path: Path | str = EDA_HTML_PATH,
    insights_path: Path | str = INSIGHTS_PATH,
) -> dict:
    """Run analyst + insights analyst steps (eda_report.html and insights.md)."""
    report = run_eda_report(data_path=data_path, html_path=html_path)
    insights = run_insights(
        data_path=data_path,
        cleaning_report=cleaning_report,
        insights_path=insights_path,
    )
    return {**report, **insights}


def main() -> None:
    from data_pipeline import run_pre_scientist_pipeline

    result = run_pre_scientist_pipeline()
    print("EDA pipeline complete:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
