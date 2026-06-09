from flask import Blueprint, jsonify, request

from services.api_key_utils import friendly_auth_error, normalize_api_key, validate_key_format
from services.crew.crew import generate_recommendations
from services.ml_service import predict
from services.validation import validate_patient_data

predict_bp = Blueprint("predict", __name__)


def _get_api_key() -> str | None:
    key = request.headers.get("X-Anthropic-Api-Key") or request.headers.get("X-Claude-Api-Key")
    return normalize_api_key(key) or None


@predict_bp.post("/api/predict")
def predict_route():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON body required"}), 400

    patient_data = payload.get("patient_data", payload)
    validation_errors = validate_patient_data(patient_data)
    if validation_errors:
        return jsonify({"error": "Invalid input", "details": validation_errors}), 400

    try:
        result = predict(patient_data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    api_key = _get_api_key()
    if api_key:
        fmt_error = validate_key_format(api_key)
        if fmt_error:
            result["recommendations"] = []
            result["ai_error"] = fmt_error
        else:
            try:
                ai = generate_recommendations(api_key, result["features"], result)
                result["recommendations"] = ai["recommendations"]
                result["risk_analysis"] = ai["risk_analysis"]
            except Exception as exc:
                result["recommendations"] = []
                result["ai_error"] = f"Recommendations failed: {friendly_auth_error(exc)}"
    else:
        result["recommendations"] = []
        result["ai_note"] = "Add your Claude API key in Settings for personalized recommendations."

    return jsonify(result)
