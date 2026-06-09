from pathlib import Path
from unittest.mock import MagicMock, patch

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


@pytest.fixture
def mock_supabase():
    mock = MagicMock()
    patients_table = MagicMock()
    assessments_table = MagicMock()

    def table(name):
        return patients_table if name == "patients" else assessments_table

    mock.table.side_effect = table
    patients_table.insert.return_value.execute.return_value.data = [
        {"id": "p1", "name": "Jane", "created_at": "2026-01-01T00:00:00Z"}
    ]
    patients_table.select.return_value.order.return_value.execute.return_value.data = [
        {"id": "p1", "name": "Jane", "created_at": "2026-01-01T00:00:00Z"}
    ]
    assessments_table.select.return_value.eq.return_value.order.return_value.execute.return_value.data = []
    assessments_table.insert.return_value.execute.return_value.data = [
        {"id": "a1", "patient_id": "p1", "risk_score": 0.2, "risk_label": "Low"}
    ]
    return mock, patients_table, assessments_table


@patch("routes.patients.get_supabase")
def test_create_patient(mock_get_supabase, client, mock_supabase):
    mock_get_supabase.return_value = mock_supabase[0]
    response = client.post("/api/patients", json={"name": "Jane"})
    assert response.status_code == 201
    assert response.get_json()["name"] == "Jane"


@patch("routes.patients.get_supabase")
def test_list_patients(mock_get_supabase, client, mock_supabase):
    mock_get_supabase.return_value = mock_supabase[0]
    response = client.get("/api/patients")
    assert response.status_code == 200
    assert len(response.get_json()) == 1


@patch("routes.patients.get_supabase")
def test_create_assessment(mock_get_supabase, client, mock_supabase):
    mock_get_supabase.return_value = mock_supabase[0]
    import json

    payload = json.loads((FIXTURES / "sample_patient.json").read_text())
    body = {
        "patient_id": "p1",
        "risk_score": 0.12,
        "risk_label": "Low",
        "shap_data": {},
        "recommendations": [],
        **payload["patient_data"],
    }
    response = client.post("/api/assessments", json=body)
    assert response.status_code == 201
