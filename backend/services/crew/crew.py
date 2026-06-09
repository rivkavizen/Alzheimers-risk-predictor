from __future__ import annotations

import json
import re

from services.anthropic_client import chat_direct, generate_recommendations_direct
from services.chat_format import clean_chat_reply
from services.crew.agents import build_llm
from services.crew.tasks import build_chat_task, build_recommendation_tasks
from services.crew.tools import format_patient_context


def _parse_recommendations(raw: str) -> list[dict]:
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


def _split_follow_up(reply: str) -> tuple[str, str | None]:
    if "FOLLOW_UP:" not in reply:
        return reply.strip(), None
    parts = reply.rsplit("FOLLOW_UP:", 1)
    message = parts[0].strip()
    question = parts[1].strip() if len(parts) > 1 else None
    return message, question or None


def _crew_recommendations(api_key: str, context: str) -> dict:
    from crewai import Crew, Process

    llm = build_llm(api_key)
    agents, tasks = build_recommendation_tasks(llm, context)
    crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False)
    result = crew.kickoff(inputs={"context": context})
    raw = str(result)
    return {"risk_analysis": raw, "recommendations": _parse_recommendations(raw)}


def generate_recommendations(api_key: str, features: dict, prediction: dict) -> dict:
    context = format_patient_context(features, prediction)
    try:
        return _crew_recommendations(api_key, context)
    except Exception:
        raw, recommendations = generate_recommendations_direct(api_key, context)
        return {"risk_analysis": raw, "recommendations": recommendations}


def chat_reply(
    api_key: str,
    features: dict,
    prediction: dict,
    history: list[dict],
    user_message: str,
) -> dict:
    context = format_patient_context(features, prediction)
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)

    try:
        from crewai import Crew, Process

        llm = build_llm(api_key)
        agents, tasks = build_chat_task(llm, context, history_text, user_message)
        crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False)
        result = crew.kickoff(
            inputs={"context": context, "history": history_text, "user_message": user_message}
        )
        reply, follow_up = _split_follow_up(clean_chat_reply(str(result)))
    except Exception:
        reply = chat_direct(api_key, context, history_text, user_message)
        reply, follow_up = _split_follow_up(clean_chat_reply(reply))

    return {"reply": reply, "follow_up_question": follow_up}
