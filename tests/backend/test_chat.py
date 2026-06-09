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


@patch("routes.chat.chat_reply")
@patch("routes.chat.get_supabase")
def test_chat_requires_key(mock_get_supabase, mock_chat_reply, client):
    response = client.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 401


@patch("routes.chat.is_supabase_configured", return_value=True)
@patch("routes.chat.chat_reply")
@patch("routes.chat.get_supabase")
def test_chat_with_key(mock_get_supabase, mock_chat_reply, _mock_configured, client):
    mock_db = MagicMock()
    sessions = MagicMock()
    messages = MagicMock()
    mock_db.table.side_effect = lambda name: sessions if name == "chat_sessions" else messages
    sessions.insert.return_value.execute.return_value.data = [{"id": "s1"}]
    messages.select.return_value.eq.return_value.order.return_value.execute.return_value.data = []
    messages.insert.return_value.execute.return_value.data = [{"id": "m1"}]
    mock_get_supabase.return_value = mock_db
    mock_chat_reply.return_value = {"reply": "Hi there", "follow_up_question": None}

    import json

    payload = json.loads((FIXTURES / "sample_patient.json").read_text())
    response = client.post(
        "/api/chat",
        json={
            "message": "What should I focus on?",
            "assessment_id": "a1",
            "features": payload["patient_data"],
            "prediction": {"risk_score": 0.2, "risk_label": "Low", "shap_explanation": {}},
        },
        headers={"X-Anthropic-Api-Key": "sk-ant-api03-testkey123456789"},
    )
    assert response.status_code == 200
    assert response.get_json()["reply"] == "Hi there"
