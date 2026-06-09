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

## Deploy to Railway

1. Push this repo to GitHub (already connected).
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo** → select `Alzheimers-risk-predictor`.
3. Railway reads `railway.toml` / `Procfile` at the repo root (no root directory override needed).
4. Add **Variables** in Railway:

| Variable | Value |
|----------|--------|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_KEY` | Supabase **service role** key |
| `MODEL_PATH` | `ml/model_artifacts/xgb_model.pkl` |
| `FEATURE_NAMES_PATH` | `ml/model_artifacts/feature_names.json` |

5. Deploy. Open the generated URL → frontend + API on one domain.
6. Verify: `GET https://your-app.up.railway.app/api/health` → `{"status":"ok"}`

**Note:** Claude API keys are **not** set on Railway — each visitor enters their own key in **Settings**.

## Supabase setup (required for save/chat/history)

1. Create a project at [supabase.com](https://supabase.com)
2. **SQL Editor** → run in order:
   - `supabase/migrations/001_create_patients.sql`
   - `supabase/migrations/002_create_assessments.sql`
   - `supabase/migrations/003_create_chat_sessions.sql`
3. **Settings → API** → copy URL + `service_role` key into Railway variables (and local `.env`)

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
