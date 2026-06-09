import { getApiKey } from "./settings.js";

const API_BASE = "";

async function request(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  const apiKey = getApiKey();
  if (apiKey) {
    headers["X-Anthropic-Api-Key"] = apiKey;
  }

  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  const text = await response.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { error: text.slice(0, 200) };
    }
  }
  if (!response.ok) {
    const detail = data.details?.length ? `: ${data.details.join("; ")}` : "";
    throw new Error((data.error || `Request failed (${response.status})`) + detail);
  }
  return data;
}

export function getHealth() {
  return request("/api/health");
}

export function predict(patientData) {
  return request("/api/predict", {
    method: "POST",
    body: JSON.stringify({ patient_data: patientData }),
  });
}

export function listPatients() {
  return request("/api/patients");
}

export function createPatient(payload) {
  return request("/api/patients", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getPatientHistory(patientId) {
  return request(`/api/patients/${patientId}/history`);
}

export function saveAssessment(payload) {
  return request("/api/assessments", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function sendChat(payload) {
  return request("/api/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getChat(sessionId) {
  return request(`/api/chat/${sessionId}`);
}

export function validateApiKey(apiKey) {
  return request("/api/validate-api-key", {
    method: "POST",
    body: JSON.stringify({ api_key: apiKey }),
    headers: { "X-Anthropic-Api-Key": apiKey },
  });
}
