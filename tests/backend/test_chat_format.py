import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from services.chat_format import clean_chat_reply


def test_clean_chat_reply_strips_markdown():
    raw = """# Hereditary Risk

---

## What Your Data Shows

- **Diabetes** – you have it
- ⚠️ Cholesterol – elevated

> Important: see a doctor.

Would you like tips?
"""
    cleaned = clean_chat_reply(raw)
    assert "#" not in cleaned
    assert "---" not in cleaned
    assert ">" not in cleaned
    assert "Diabetes" in cleaned


def test_clean_chat_reply_preserves_follow_up():
    raw = "Short answer here.\n\nFOLLOW_UP: Do your children know their cholesterol numbers?"
    cleaned = clean_chat_reply(raw)
    assert cleaned.endswith("FOLLOW_UP: Do your children know their cholesterol numbers?")
