# Alzheimer's Disease Risk Prediction System

ML-powered risk assessment with SHAP explanations, CrewAI recommendations, and patient trend tracking.

**Repository:** https://github.com/rivkavizen/Alzheimers-risk-predictor

## Quick start (ML pipeline)

```powershell
cd c:\personal\AI_course\FinalProject
pip install -r requirements-dev.txt

# Step 1: Clean raw data
python ml/clean_data.py

# Step 2: Train model
python ml/train.py

# Step 3: Verify SHAP output
python ml/evaluate.py

# Run ML tests
pytest tests/ml/ -v

# Or run everything at once
.\scripts\verify.ps1
```

## Data pipeline

| File | Purpose |
|------|---------|
| `alzheimers_disease_data.csv` | Raw data (never overwritten) |
| `alzheimers_disease_data_cleaned.csv` | Produced by `ml/clean_data.py` |
| `ml/model_artifacts/` | Trained model + feature names |

After changing the raw dataset, re-run `clean_data.py` → `train.py` → `evaluate.py`.

## Environment variables

Copy `.env.example` to `.env` for backend development (Supabase, model paths).

Claude API keys are entered by each visitor in the browser Settings page — not stored on the server.

## Project structure

See the implementation plan in the project documentation for full architecture (Flask backend, vanilla HTML/CSS/JS frontend, CrewAI agents, Supabase).
