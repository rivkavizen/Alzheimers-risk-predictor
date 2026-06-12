# Alzheimer's Disease Risk Prediction System

ML-powered risk assessment with SHAP explanations, CrewAI recommendations, and patient trend tracking.

**Repository:** https://github.com/rivkavizen/Alzheimers-risk-predictor  
**Live app:** https://alzheimers-risk-predictor-production.up.railway.app

## Quick start (ML pipeline)

```powershell
cd c:\personal\AI_course\FinalProject
pip install -r requirements-dev.txt

# Step 1: Pre-scientist pipeline (3 agents → then Data Scientist trains in step 2)
python ml/clean_data.py
# Engineer: dataset_contract.json | Analyst: eda_report.html | Insights Analyst: insights.md

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
| `ml/dataset_contract.json` | Data Engineer → contract for Data Scientist |
| `ml/eda_report.html` | Data Analyst → visual EDA report |
| `ml/insights.md` | Insights Analyst → business summary |
| `ml/data_pipeline.py` | Orchestrates 3 pre-scientist agents |
| `ml/model_artifacts/features.csv` | Engineered feature matrix |
| `ml/model_artifacts/model.pkl` | Selected production model |
| `ml/model_artifacts/evaluation_report.md` | CV comparison of model variations |
| `ml/model_artifacts/model_card.md` | Purpose, metrics, limitations, ethics |
| `ml/model_artifacts/` | Model + feature metadata |

After changing the raw dataset, re-run `clean_data.py` → `train.py` → `evaluate.py`.

Re-run full pre-scientist pipeline: `python ml/data_pipeline.py`

## Environment variables

Copy `.env.example` to `.env` for backend development (Supabase, model paths).

Claude API keys are entered by each visitor in the browser Settings page — not stored on the server.

## Backend API (local)

```powershell
pip install -r backend/requirements.txt
.\scripts\start.ps1
# or: cd backend && python app.py
# Open http://localhost:5000  (frontend + API)
# GET  http://localhost:5000/api/health
# POST http://localhost:5000/api/predict  (body: tests/fixtures/sample_patient.json)
```

For AI recommendations and chat, enter your Claude API key in **Settings** in the browser.
Install CrewAI dependencies: `pip install -r backend/requirements.txt`

## Deploy to Railway

1. Push this repo to GitHub (already connected).
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo** → select `Alzheimers-risk-predictor`.
3. Railway reads `railway.toml` / `Procfile` at the repo root (no root directory override needed).
4. Add **Variables** in Railway:

| Variable | Value |
|----------|--------|
| `SUPABASE_URL` | `https://puhtwpurgkqylnanortj.supabase.co` |
| `SUPABASE_KEY` | [service_role key](https://supabase.com/dashboard/project/puhtwpurgkqylnanortj/settings/api) |
| `MODEL_PATH` | `ml/model_artifacts/model.pkl` |
| `FEATURE_NAMES_PATH` | `ml/model_artifacts/feature_names.json` |

5. Deploy (`railway up` or push to GitHub if connected). Root `requirements.txt` points at `backend/requirements.txt` for Nixpacks.
6. Generate a public domain in **Networking** if needed.
7. Verify: `GET https://alzheimers-risk-predictor-production.up.railway.app/api/health` → `{"status":"ok"}`

**Note:** Claude API keys are **not** set on Railway — each visitor enters their own key in **Settings**.

## Supabase setup (required for save/chat/history)

**Your project:** [Supabase dashboard](https://supabase.com/dashboard/project/puhtwpurgkqylnanortj)  
**URL:** `https://puhtwpurgkqylnanortj.supabase.co`

1. **SQL Editor** → run in order (see `supabase/README.md`):
   - `supabase/migrations/001_create_patients.sql`
   - `supabase/migrations/002_create_assessments.sql`
   - `supabase/migrations/003_create_chat_sessions.sql`
2. **[Settings → API](https://supabase.com/dashboard/project/puhtwpurgkqylnanortj/settings/api)** → copy **service_role** key into `.env` as `SUPABASE_KEY`
3. Restart the server. URL is preconfigured in `backend/config.py` and `.env.example`

## CI (GitHub Actions)

On every push/PR to `main`, `.github/workflows/test.yml` runs:

```text
clean_data → pytest tests/
```

## End-to-end test checklist

- [ ] `python ml/clean_data.py` and `pytest tests/ -v` pass locally
- [ ] `cd backend && python app.py` → http://localhost:5000 loads
- [ ] Settings → save Claude API key
- [ ] New Assessment → Results (risk + SHAP + recommendations)
- [ ] Save assessment → row in Supabase `assessments` table
- [ ] Chat → 3+ message exchanges
- [ ] Trends → chart after 2+ saved assessments for same patient
- [ ] Railway `/api/health` returns 200

## Project structure

```text
ml/           Data cleaning, training, model artifacts
backend/      Flask API, CrewAI agents, serves frontend/
frontend/     Vanilla HTML/CSS/JS
supabase/     Database migrations
tests/        ML + API tests
```
