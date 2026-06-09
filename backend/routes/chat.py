from flask import Blueprint, jsonify, request

from db import get_supabase
from services.crew.crew import chat_reply

chat_bp = Blueprint("chat", __name__)


def _get_api_key() -> str | None:
    return request.headers.get("X-Anthropic-Api-Key") or request.headers.get("X-Claude-Api-Key")


@chat_bp.post("/api/chat")
def post_chat():
    api_key = _get_api_key()
    if not api_key:
        return jsonify({"error": "Claude API key required. Add X-Anthropic-Api-Key header."}), 401

    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    features = payload.get("features", {})
    prediction = payload.get("prediction", {})
    session_id = payload.get("session_id")
    assessment_id = payload.get("assessment_id")

    db = get_supabase()
    if not session_id:
        if not assessment_id:
            return jsonify({"error": "session_id or assessment_id is required"}), 400
        session = db.table("chat_sessions").insert({"assessment_id": assessment_id}).execute()
        session_id = session.data[0]["id"]

    history_rows = (
        db.table("chat_messages")
        .select("role, content")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
        .data
    )

    db.table("chat_messages").insert(
        {"session_id": session_id, "role": "user", "content": message}
    ).execute()

    result = chat_reply(api_key, features, prediction, history_rows, message)

    db.table("chat_messages").insert(
        {"session_id": session_id, "role": "assistant", "content": result["reply"]}
    ).execute()

    return jsonify({"session_id": session_id, **result})


@chat_bp.get("/api/chat/<session_id>")
def get_chat(session_id: str):
    messages = (
        get_supabase()
        .table("chat_messages")
        .select("id, role, content, created_at")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
        .data
    )
    return jsonify({"session_id": session_id, "messages": messages})
