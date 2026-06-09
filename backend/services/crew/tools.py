import json


def format_patient_context(features: dict, prediction: dict) -> str:
    shap = prediction.get("shap_explanation", {})
    return json.dumps(
        {
            "patient_features": features,
            "risk_score": prediction.get("risk_score"),
            "risk_label": prediction.get("risk_label"),
            "top_risk_factors": shap.get("top_risk_factors", []),
            "top_protective_factors": shap.get("top_protective_factors", []),
        },
        indent=2,
    )
