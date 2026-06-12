# Model Evaluation Report

Generated: 2026-06-12 08:42 UTC

## Training data summary

- Rows: **2,149**
- Base features: **32**
- Engineered features: **4**
- Total model inputs: **36**
- Diagnosis prevalence: **35.4%**
- Validation: **5-fold stratified cross-validation**

## Model comparison

| Model | CV AUC | Accuracy | Precision | Recall | F1 |
|-------|--------|----------|-----------|--------|-----|
| XGBoost (tuned) | 0.9517 | 0.9488 | 0.9368 | 0.9171 | 0.9269 |
| Random Forest | 0.9503 | 0.9511 | 0.9396 | 0.9211 | 0.9302 |
| XGBoost (shallow) | 0.9484 | 0.9493 | 0.9323 | 0.9237 | 0.9280 |

## Selected model

**XGBoost (tuned)** was selected based on highest cross-validated AUC (0.9517).

### Rationale

Gradient boosting with class-weighting; strong tabular performance.

## Notes

- Metrics are from out-of-fold predictions (no train-set leakage).
- Class imbalance handled via `scale_pos_weight` for tree models.
- See `model_card.md` for limitations and ethical considerations.
