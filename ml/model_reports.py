"""Generate evaluation_report.md and model_card.md."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def build_evaluation_report(
    comparison: list[dict],
    winner: dict,
    training_summary: dict,
) -> str:
    lines = [
        "# Model Evaluation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Training data summary",
        "",
        f"- Rows: **{training_summary['rows']:,}**",
        f"- Base features: **{training_summary['base_features']}**",
        f"- Engineered features: **{training_summary['engineered_features']}**",
        f"- Total model inputs: **{training_summary['total_features']}**",
        f"- Diagnosis prevalence: **{training_summary['positive_rate']:.1%}**",
        f"- Validation: **5-fold stratified cross-validation**",
        "",
        "## Model comparison",
        "",
        "| Model | CV AUC | Accuracy | Precision | Recall | F1 |",
        "|-------|--------|----------|-----------|--------|-----|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['name']} | {row['cv_auc']:.4f} | {row['accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Selected model",
            "",
            f"**{winner['name']}** was selected based on highest cross-validated AUC "
            f"({winner['cv_auc']:.4f}).",
            "",
            "### Rationale",
            "",
            winner.get("rationale", "Best discrimination on held-out CV folds."),
            "",
            "## Notes",
            "",
            "- Metrics are from out-of-fold predictions (no train-set leakage).",
            "- Class imbalance handled via `scale_pos_weight` for tree models.",
            "- See `model_card.md` for limitations and ethical considerations.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_model_card(
    winner: dict,
    training_summary: dict,
    comparison: list[dict],
) -> str:
    return f"""# Model Card — Alzheimer's Disease Risk Predictor

## Model purpose

Binary classifier estimating the probability that a patient record aligns with a positive
Alzheimer's **diagnosis label** (`Diagnosis=1`) from clinical, lifestyle, and cognitive features.
The model supports **risk screening and education** in the web app — it does **not** provide
a clinical diagnosis.

- **Algorithm:** {winner['name']}
- **Artifact:** `ml/model_artifacts/model.pkl`
- **Features:** `ml/model_artifacts/features.csv`
- **Explanations:** SHAP (TreeExplainer) at inference time

## Training data summary

| Item | Value |
|------|--------|
| Source | `alzheimers_disease_data_cleaned.csv` |
| Contract | `ml/dataset_contract.json` |
| Records | {training_summary['rows']:,} |
| Target | `Diagnosis` (0 = no, 1 = yes) |
| Class balance | {training_summary['positive_rate']:.1%} positive |
| Base features | {training_summary['base_features']} |
| Engineered features | {training_summary['engineered_features']} (`PulsePressure`, `CholesterolRatio`, `SymptomBurden`, `LifestyleScore`) |
| Forbidden columns | `PatientID`, `DoctorInCharge` (excluded per contract) |

## Metrics (5-fold CV)

| Metric | Value |
|--------|--------|
| AUC-ROC | {winner['cv_auc']:.4f} |
| Accuracy | {winner['accuracy']:.4f} |
| Precision | {winner['precision']:.4f} |
| Recall | {winner['recall']:.4f} |
| F1 | {winner['f1']:.4f} |

### Compared variations

{chr(10).join(f"- **{m['name']}**: AUC {m['cv_auc']:.4f}" for m in comparison)}

## Limitations

- Trained on a **single static cohort**; performance may not generalize to other populations or geographies.
- Features are **coded ordinals** (e.g. ethnicity, education) — not full demographic representation.
- **Correlation ≠ causation**; SHAP highlights association, not proven causal pathways.
- Threshold default 0.5 for High/Low label may not match clinical operating points.
- No external temporal validation; dataset may not reflect future data drift.
- Model does not incorporate imaging, biomarkers, or physician notes available in real care.

## Ethical considerations

- **Not a diagnostic device** — outputs must be presented as risk estimates with clear disclaimers.
- **Fairness:** Performance across ethnicity/education subgroups was not independently audited;
  deploy with caution and monitor disparate impact if used beyond research demos.
- **Privacy:** Patient identifiers are removed in cleaning; inference should avoid logging sensitive inputs.
- **Autonomy:** Users should retain agency; recommendations are supportive, not prescriptive medical orders.
- **Transparency:** SHAP explanations are provided to improve interpretability, not to imply certainty.
- **Human oversight:** Clinicians or caregivers should interpret results; the app must not replace professional assessment.

---
*Generated by `ml/train.py` — Data Scientist pipeline.*
"""


def save_markdown(content: str, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
