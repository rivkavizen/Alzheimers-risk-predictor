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

## Backend API (local)

```powershell
pip install -r backend/requirements.txt
cd backend
python app.py
# Open http://localhost:5000  (frontend + API)
# GET  http://localhost:5000/api/health
# POST http://localhost:5000/api/predict  (body: tests/fixtures/sample_patient.json)
```

For AI recommendations and chat, enter your Claude API key in **Settings** in the browser.
Install CrewAI dependencies: `pip install -r backend/requirements.txt`

## Supabase setup

1. Create a project at [supabase.com](https://supabase.com)
2. Run SQL files in order in the SQL Editor:
   - `supabase/migrations/001_create_patients.sql`
   - `supabase/migrations/002_create_assessments.sql`
   - `supabase/migrations/003_create_chat_sessions.sql`
3. Copy project URL and service role key to `.env`

## Project structure

See the implementation plan in the project documentation for full architecture (Flask backend, vanilla HTML/CSS/JS frontend, CrewAI agents, Supabase).
