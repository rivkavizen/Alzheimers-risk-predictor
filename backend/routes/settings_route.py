from flask import Blueprint, jsonify, request

from services.anthropic_client import complete
from services.api_key_utils import normalize_api_key, validate_key_format

settings_bp = Blueprint("settings", __name__)


@settings_bp.post("/api/validate-api-key")
def validate_api_key():
    payload = request.get_json(silent=True) or {}
    key = normalize_api_key(
        payload.get("api_key")
        or request.headers.get("X-Anthropic-Api-Key")
        or request.headers.get("X-Claude-Api-Key")
    )

    fmt_error = validate_key_format(key)
    if fmt_error:
        return jsonify({"valid": False, "error": fmt_error}), 400

    try:
        complete(
            key,
            "You are a test assistant.",
            "Reply with exactly: OK",
            max_tokens=16,
        )
        return jsonify({"valid": True, "message": "API key works."})
    except Exception as exc:
        from services.api_key_utils import friendly_auth_error

        return jsonify({"valid": False, "error": friendly_auth_error(exc)}), 401
