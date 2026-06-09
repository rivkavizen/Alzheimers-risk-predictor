# Start the app locally (run from project root)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Test-Path "ml\model_artifacts\xgb_model.pkl")) {
    Write-Host "Model not found. Running training pipeline..."
    python ml/clean_data.py
    python ml/train.py
}

Write-Host "Starting server at http://localhost:5000"
Set-Location backend
python app.py
