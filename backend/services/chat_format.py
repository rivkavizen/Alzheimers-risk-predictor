"""Format AI chat replies for plain-text UI."""

import re

CHAT_SYSTEM_PROMPT = """You are a supportive health coach in a simple chat window.

Reply like a warm, concise human — NOT a medical report.

STRICT RULES:
- Answer the user's question directly in the first 1-2 sentences.
- Use 2-4 short paragraphs maximum (about 120-180 words unless they ask for more).
- Plain text only: NO markdown headers (#), NO horizontal rules (---), NO bullet lists, NO tables, NO blockquotes (>).
- NO emojis or decorative symbols.
- NO section titles like "What Your Data Shows" or "Recommendations".
- Never diagnose. Use cautious language ("may", "could", "worth discussing with a doctor").
- If a medical disclaimer is needed, use ONE brief sentence woven into the reply.
- If you need one clarifying question, put it on its own last line as: FOLLOW_UP: <question>
"""


def clean_chat_reply(text: str) -> str:
    """Strip markdown/report formatting the model may still produce."""
    if "FOLLOW_UP:" in text:
        main, follow = text.rsplit("FOLLOW_UP:", 1)
        cleaned = _clean_body(main)
        question = follow.strip()
        return f"{cleaned}\n\nFOLLOW_UP: {question}" if question else cleaned
    return _clean_body(text)


def _clean_body(text: str) -> str:
    text = re.sub(r"\n-{3,}\n", "\n\n", text)
    text = re.sub(r"\n\*{3,}\n", "\n\n", text)

    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or re.fullmatch(r"[-*_]{3,}", line):
            continue
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"^[-*•]\s+", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        line = re.sub(r"^>\s*", "", line)
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        line = re.sub(r"\*([^*]+)\*", r"\1", line)
        line = re.sub(r"__([^_]+)__", r"\1", line)
        line = re.sub(
            r"^[\U0001F300-\U0001FAFF\U00002600-\U000027BF\u2705\u26A0\uFE0F\u2695\uFE0F\u2B50]+\s*",
            "",
            line,
        )
        lines.append(line)

    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result).strip()
    return result
