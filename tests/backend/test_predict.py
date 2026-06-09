import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def client():
    import sys

    backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
    sys.path.insert(0, str(backend_dir))
    from app import app

    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict(client):
    payload = json.loads((FIXTURES / "sample_patient.json").read_text())
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "risk_score" in data
    assert "risk_label" in data
    assert "shap_explanation" in data
    assert 0 <= data["risk_score"] <= 1
    assert data["risk_label"] in ("High", "Low")
