from flask import Blueprint, jsonify, request

from db import get_supabase
from services.ml_service import FEATURE_MAP

patients_bp = Blueprint("patients", __name__)

FEATURE_COLUMNS = list(FEATURE_MAP.keys())


def _extract_features(payload: dict) -> dict:
    return {k: payload[k] for k in FEATURE_COLUMNS if k in payload}


@patients_bp.post("/api/patients")
def create_patient():
    payload = request.get_json(silent=True) or {}
    name = payload.get("name", "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400

    row = {
        "name": name,
        "date_of_birth": payload.get("date_of_birth"),
        "notes": payload.get("notes"),
    }
    result = get_supabase().table("patients").insert(row).execute()
    return jsonify(result.data[0]), 201


@patients_bp.get("/api/patients")
def list_patients():
    result = (
        get_supabase()
        .table("patients")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return jsonify(result.data)


@patients_bp.get("/api/patients/<patient_id>/history")
def patient_history(patient_id: str):
    result = (
        get_supabase()
        .table("assessments")
        .select("*")
        .eq("patient_id", patient_id)
        .order("assessed_at", desc=False)
        .execute()
    )
    return jsonify(result.data)


@patients_bp.post("/api/assessments")
def create_assessment():
    payload = request.get_json(silent=True) or {}
    patient_id = payload.get("patient_id")
    if not patient_id:
        return jsonify({"error": "patient_id is required"}), 400

    required = ["risk_score", "risk_label"]
    for field in required:
        if field not in payload:
            return jsonify({"error": f"{field} is required"}), 400

    row = _extract_features(payload)
    row.update(
        {
            "patient_id": patient_id,
            "risk_score": payload["risk_score"],
            "risk_label": payload["risk_label"],
            "shap_data": payload.get("shap_data", {}),
            "recommendations": payload.get("recommendations", []),
        }
    )
    if payload.get("assessed_at"):
        row["assessed_at"] = payload["assessed_at"]

    result = get_supabase().table("assessments").insert(row).execute()
    return jsonify(result.data[0]), 201
