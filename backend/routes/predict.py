from flask import Blueprint, jsonify, request

from services.ml_service import predict

predict_bp = Blueprint("predict", __name__)


@predict_bp.post("/api/predict")
def predict_route():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "JSON body required"}), 400

    patient_data = payload.get("patient_data", payload)
    try:
        result = predict(patient_data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)
