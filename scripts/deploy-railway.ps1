# Deploy to Railway (run from project root after: railway login)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== Pre-deploy checks ===" -ForegroundColor Cyan

if (-not (Test-Path "ml\model_artifacts\model.pkl")) {
    throw "Model missing. Run: python ml/clean_data.py && python ml/train.py"
}

railway whoami 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Railway CLI is not logged in. Run this in your terminal first:" -ForegroundColor Yellow
    Write-Host "  railway login" -ForegroundColor White
    exit 1
}

Write-Host "Model OK" -ForegroundColor Green

$status = railway status 2>&1 | Out-String
if ($status -match "No linked project") {
    Write-Host ""
    Write-Host "Linking to a Railway project (choose existing or create new)..." -ForegroundColor Cyan
    railway link
}

Write-Host ""
Write-Host "=== Setting Railway variables ===" -ForegroundColor Cyan
railway variables set "SUPABASE_URL=https://puhtwpurgkqylnanortj.supabase.co"
railway variables set "MODEL_PATH=ml/model_artifacts/model.pkl"
railway variables set "FEATURE_NAMES_PATH=ml/model_artifacts/feature_names.json"
railway variables set "ANTHROPIC_MODEL=claude-sonnet-4-6"

$envFile = Join-Path $PWD ".env"
if (Test-Path $envFile) {
    $supabaseKey = (Get-Content $envFile | Where-Object { $_ -match '^SUPABASE_KEY=' }) -replace '^SUPABASE_KEY=', ''
    if ($supabaseKey) {
        railway variables set "SUPABASE_KEY=$supabaseKey"
        Write-Host "SUPABASE_KEY set from .env" -ForegroundColor Green
    }
}

if (-not $supabaseKey) {
    Write-Host ""
    Write-Host "SUPABASE_KEY not found in .env — set it in Railway dashboard:" -ForegroundColor Yellow
    Write-Host "  https://supabase.com/dashboard/project/puhtwpurgkqylnanortj/settings/api" -ForegroundColor White
}

Write-Host ""
Write-Host "=== Deploying ===" -ForegroundColor Cyan
railway up --detach

Write-Host ""
Write-Host "=== Generate public URL (if needed) ===" -ForegroundColor Cyan
railway domain 2>$null

$url = (railway status 2>&1 | Select-String -Pattern "https://[^\s]+" | ForEach-Object { $_.Matches.Value } | Select-Object -First 1)
if ($url) {
    Write-Host ""
    Write-Host "App URL: $url" -ForegroundColor Green
    Write-Host "Health:  $url/api/health" -ForegroundColor Green
} else {
    Write-Host "Open Railway dashboard → your service → Settings → Networking → Generate Domain" -ForegroundColor Yellow
}
