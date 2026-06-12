# Alzheimer's Dataset — Business & Analytical Insights

## Executive summary

This cohort contains **2,149** cleaned patient records with a **35.4%** positive diagnosis rate
(64.6% negative). The dataset is suitable for supervised risk modeling with class imbalance
that should be monitored during training and evaluation.

## Dataset overview

| Metric | Value |
|--------|-------|
| Records (cleaned) | 2,149 |
| Raw rows processed | 2149 |
| Rows removed in cleaning | 0 |
| Mean age | 74.9 years |
| Mean BMI | 27.7 |
| Mean MMSE | 14.8 / 30 |
| Diagnosis prevalence | 35.4% |

## Key visual insights

See `eda_report.html` for interactive-style charts including:

- **Class balance** — diagnosis vs no diagnosis counts
- **Age by diagnosis** — older patients show higher diagnosis density
- **MMSE by diagnosis** — lower cognitive scores align with positive cases
- **Correlation heatmap** — strongest linear relationships with diagnosis
- **Lifestyle averages** — cohort-level modifiable factor profile

## Strongest statistical signals

- **FunctionalAssessment**: correlation with diagnosis = -0.365
- **ADL**: correlation with diagnosis = -0.332
- **MemoryComplaints**: correlation with diagnosis = +0.307
- **MMSE**: correlation with diagnosis = -0.237
- **BehavioralProblems**: correlation with diagnosis = +0.224
- **SleepQuality**: correlation with diagnosis = -0.057
- **EducationLevel**: correlation with diagnosis = -0.044
- **CholesterolHDL**: correlation with diagnosis = +0.043

## Modifiable risk factor patterns

- **Smoking**: diagnosis rate in top quartile = 35.0% vs bottom quartile = 35.5%
- **AlcoholConsumption**: diagnosis rate in top quartile = 35.9% vs bottom quartile = 37.7%
- **PhysicalActivity**: diagnosis rate in top quartile = 34.4% vs bottom quartile = 33.6%
- **DietQuality**: diagnosis rate in top quartile = 36.1% vs bottom quartile = 33.5%
- **SleepQuality**: diagnosis rate in top quartile = 32.3% vs bottom quartile = 38.1%
- **BMI**: diagnosis rate in top quartile = 39.0% vs bottom quartile = 34.8%

## Data quality notes

- Cleaning removed duplicate rows, invalid ranges, and nulls per `dataset_contract.json`.
- Identifier columns (`PatientID`, `DoctorInCharge`) are excluded from modeling.
- All binary flags are strictly 0/1; ordinal and continuous fields are range-validated.

## Recommendations for the Data Scientist crew

1. **Read `dataset_contract.json` before training** — schema, allowed values, and constraints are mandatory.
2. **Use stratified cross-validation** — diagnosis prevalence is ~35%, not 50/50.
3. **Prioritize interpretability** — SHAP explanations are a product requirement; avoid black-box-only approaches.
4. **Monitor MMSE, functional, and symptom features** — they dominate correlation with diagnosis.
5. **Do not leak target** — train only on features listed in `model_features` inside the contract.
6. **Re-run `ml/eda.py` after any raw data change** — refresh HTML, insights, and contract together.

## Business implications

- **Screening focus**: Patients with low MMSE and functional scores warrant closer follow-up in the product UX.
- **Prevention messaging**: Lifestyle factors (activity, diet, sleep, smoking) show measurable prevalence gaps across risk quartiles.
- **Fairness review**: Ethnicity and education are ordinal codes — treat carefully in explanations and avoid over-claiming causality.
- **Deployment**: Inference inputs must match API snake_case fields mapped in the dataset contract.
