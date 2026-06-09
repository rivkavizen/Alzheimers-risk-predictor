"""Direct Anthropic API calls (used by CrewAI and as fallback)."""

from __future__ import annotations

import json
import re

from config import ANTHROPIC_MODEL


def _client(api_key: str):
    import anthropic

    return anthropic.Anthropic(api_key=api_key)


def complete(api_key: str, system: str, user: str, max_tokens: int = 2048) -> str:
    from services.api_key_utils import normalize_api_key

    api_key = normalize_api_key(api_key)
    response = _client(api_key).messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def generate_recommendations_direct(api_key: str, context: str) -> tuple[str, list[dict]]:
    system = (
        "You are a health advisor for Alzheimer's risk reduction. "
        "Never diagnose. Use plain language. Respond with valid JSON only."
    )
    user = (
        f"Patient context:\n{context}\n\n"
        "Return a JSON array of 4-6 objects with keys: "
        "category, action, evidence, priority (high|medium|low). "
        "Focus on modifiable risk factors from SHAP. JSON only."
    )
    raw = complete(api_key, system, user)
    return raw, _parse_recommendations_json(raw)


def chat_direct(api_key: str, context: str, history: str, user_message: str) -> str:
    from services.chat_format import CHAT_SYSTEM_PROMPT

    user = (
        f"Patient context:\n{context}\n\n"
        f"Conversation:\n{history}\n\n"
        f"User: {user_message}"
    )
    return complete(api_key, CHAT_SYSTEM_PROMPT, user, max_tokens=600)


def _parse_recommendations_json(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return [{"category": "General", "action": raw[:500], "evidence": "AI summary", "priority": "medium"}]
