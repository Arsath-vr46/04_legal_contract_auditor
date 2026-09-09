"""
Flask + Gemini backend. Zero external database -- knowledge_base.json is
loaded into memory at startup and injected into every request as grounding
context that Gemini is instructed to prioritize over general knowledge.
"""
import os
import json
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types

import chatbot_config as cfg

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load domain knowledge base into memory once, at startup (zero external DB).
# ---------------------------------------------------------------------------
with open(BASE_DIR / "knowledge_base.json", "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = json.load(f)

KNOWLEDGE_BASE_TEXT = json.dumps(KNOWLEDGE_BASE, indent=2)

# ---------------------------------------------------------------------------
# Gemini client (lazy init so the app can still boot without a key set,
# and fail with a clear error only when a chat request actually comes in).
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY")
_client = None


def get_client():
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Set it before starting the server, e.g. "
                "export GEMINI_API_KEY=your_key_here"
            )
        _client = genai.Client(api_key=API_KEY)
    return _client


def build_system_instruction():
    """Combine the bot persona/scope with the in-memory knowledge base."""
    return (
        cfg.SYSTEM_INSTRUCTION
        + "\n\n=== DOMAIN KNOWLEDGE BASE (authoritative -- prioritize over "
        "general knowledge, cite it when relevant) ===\n"
        + KNOWLEDGE_BASE_TEXT
    )


def to_gemini_contents(history, message):
    """Convert the client's {role, parts} history into genai Content objects."""
    contents = []
    for turn in history or []:
        role = turn.get("role")
        if role not in ("user", "model"):
            continue
        parts = turn.get("parts") or []
        text_parts = [
            types.Part.from_text(text=str(p)) for p in parts if str(p).strip()
        ]
        if text_parts:
            contents.append(types.Content(role=role, parts=text_parts))
    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=message)])
    )
    return contents


@app.route("/")
def index():
    return render_template(
        "index.html", bot_name=cfg.BOT_NAME, bot_title=cfg.BOT_TITLE
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        message = (payload.get("message") or "").strip()
        history = payload.get("history") or []

        if not message:
            return jsonify({"error": "The 'message' field is required and cannot be empty."}), 400
        if not isinstance(history, list):
            return jsonify({"error": "'history' must be a list of {role, parts} turns."}), 400

        client = get_client()
        contents = to_gemini_contents(history, message)

        response = client.models.generate_content(
            model=cfg.MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=build_system_instruction(),
                temperature=cfg.TEMPERATURE,
            ),
        )

        reply = getattr(response, "text", None)
        if not reply:
            return jsonify({"error": "The model returned an empty response. Please try again."}), 502

        return jsonify({"reply": reply})

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Unexpected server error: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
