"""Validate visitor Anthropic API keys."""


def normalize_api_key(key: str | None) -> str:
    if not key:
        return ""
    return key.strip().strip('"').strip("'")


def validate_key_format(key: str) -> str | None:
    """Return error message if format is invalid, else None."""
    if not key:
        return "API key is required"
    if key.startswith("sb_secret_") or key.startswith("sb_publishable_"):
        return (
            "This looks like a Supabase key, not an Anthropic key. "
            "Get a Claude key at console.anthropic.com (starts with sk-ant-)."
        )
    if key.startswith("eyJ"):
        return (
            "This looks like a JWT/Supabase key. "
            "Use an Anthropic API key from console.anthropic.com (starts with sk-ant-)."
        )
    if not key.startswith("sk-ant-"):
        return "Anthropic API keys start with sk-ant-. Check console.anthropic.com"
    if len(key) < 20:
        return "API key appears too short"
    return None


def friendly_auth_error(exc: Exception) -> str:
    msg = str(exc)
    if "401" in msg or "authentication_error" in msg or "invalid x-api-key" in msg:
        return (
            "Invalid Anthropic API key. Open Settings and paste your Claude key "
            "from console.anthropic.com (starts with sk-ant-). "
            "Do not use your Supabase key here."
        )
    if "404" in msg or "not_found_error" in msg or "model:" in msg:
        return (
            "Claude model not available on your API account. "
            "Contact support or set ANTHROPIC_MODEL in .env (default: claude-sonnet-4-6)."
        )
    return msg
