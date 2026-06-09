const API_KEY_STORAGE = "anthropic_api_key";

export function normalizeApiKey(key) {
  return (key || "").trim().replace(/^["']|["']$/g, "");
}

export function validateKeyFormat(key) {
  const k = normalizeApiKey(key);
  if (!k) return "API key is required";
  if (k.startsWith("sb_secret_") || k.startsWith("sb_publishable_")) {
    return "This is a Supabase key. Use an Anthropic key from console.anthropic.com (sk-ant-...).";
  }
  if (k.startsWith("eyJ")) {
    return "This looks like a JWT/Supabase key. Use sk-ant-... from console.anthropic.com.";
  }
  if (!k.startsWith("sk-ant-")) {
    return "Anthropic keys start with sk-ant-. Get one at console.anthropic.com";
  }
  return null;
}

export function getApiKey() {
  return normalizeApiKey(localStorage.getItem(API_KEY_STORAGE));
}

export function saveApiKey(key) {
  const normalized = normalizeApiKey(key);
  const error = validateKeyFormat(normalized);
  if (error) throw new Error(error);
  localStorage.setItem(API_KEY_STORAGE, normalized);
}

export function clearApiKey() {
  localStorage.removeItem(API_KEY_STORAGE);
}

export function hasApiKey() {
  const key = getApiKey();
  return Boolean(key) && !validateKeyFormat(key);
}
