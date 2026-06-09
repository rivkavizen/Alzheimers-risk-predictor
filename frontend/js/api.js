import { getApiKey } from "./settings.js";

const API_BASE = "";

async function request(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  const apiKey = getApiKey();
  if (apiKey) {
    headers["X-Anthropic-Api-Key"] = apiKey;
  }

  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `Request failed (${response.status})`);
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
