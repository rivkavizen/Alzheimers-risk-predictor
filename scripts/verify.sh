#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Cleaning data..."
python ml/clean_data.py

echo "==> Training model..."
python ml/train.py

echo "==> Evaluating SHAP..."
python ml/evaluate.py

echo "==> Running tests..."
pytest tests/ -v

echo "==> Done."
