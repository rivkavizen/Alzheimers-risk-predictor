# Full ML pipeline verification
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "==> Cleaning data..."
python ml/clean_data.py

Write-Host "==> Training model..."
python ml/train.py

Write-Host "==> Evaluating SHAP..."
python ml/evaluate.py

Write-Host "==> Running tests..."
pytest tests/ml/ -v

Write-Host "==> Done."
