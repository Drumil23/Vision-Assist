"""
Conversational scene model — Gemini 2.5 Flash Lite
Uses correct google-genai SDK types.Content format for multi-turn history.
"""

import os
import io
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL          = "gemini-2.5-flash-lite"

SYSTEM_PROMPT = """\
You are a real-time visual guide for a blind person. You can see through their \
camera. You are their calm, caring, street-smart friend — not a robot.

How to respond:
- Talk TO them directly: "you're in...", "ahead of you...", "to your left..."
- Be warm, natural, conversational. Like texting a friend.
- When they ask a question, answer it specifically using what you see.
- Proactively warn about anything dangerous or in their path.
- Keep responses SHORT — 2-3 sentences max unless they ask for more detail.
- If they say "take me to the door" or "help me find X", guide them step by step.
- Read any visible text aloud when relevant.
- Never say "the image shows" or "I can see" — just describe directly.
- End with a gentle next-step suggestion when it makes sense.
"""

_client     = None
_history    = []   # list of types.Content objects
_last_frame = None


def _load():
    global _client
    if _client:
        return
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set.")
    _client = genai.Client(api_key=GEMINI_API_KEY)


def _img_part(pil_image: Image.Image) -> types.Part:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=80)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def set_frame(pil_image: Image.Image):
    global _last_frame
    _last_frame = pil_image


def chat(user_message: str, pil_image: Image.Image = None) -> str:
    """
    Send a user message with the current camera frame.
    Maintains full conversation history using proper SDK types.
    """
    _load()
    global _history

    # build user parts — image first, then text
    frame   = pil_image or _last_frame
    parts   = []
    if frame is not None:
        parts.append(_img_part(frame))
    parts.append(types.Part.from_text(text=user_message))

    # append user turn using proper types.Content
    _history.append(
        types.Content(role="user", parts=parts)
    )

    # keep last 10 turns to stay within token limits
    trimmed = _history[-10:]

    try:
        response = _client.models.generate_content(
            model=MODEL,
            contents=trimmed,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.25,
                max_output_tokens=200,
            ),
        )
        reply = response.text.strip()

        # append assistant turn
        _history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=reply)],
            )
        )
        return reply

    except Exception as e:
        err = str(e)
        # remove the failed user turn so history stays clean
        if _history and _history[-1].role == "user":
            _history.pop()
        if "429" in err:
            return "I'm taking a short break — hit my daily limit. Try again in a minute."
        if "404" in err:
            return f"Model error: {err}"
        return f"Error: {err}"


def background_describe(pil_image: Image.Image) -> str:
    """Silent background scene check — doesn't touch conversation history."""
    _load()
    if pil_image is None:
        return ""
    try:
        response = _client.models.generate_content(
            model=MODEL,
            contents=[
                _img_part(pil_image),
                types.Part.from_text(
                    text="In one sentence, what is the key thing in this scene?"
                ),
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=60,
            ),
        )
        return response.text.strip()
    except Exception:
        return ""


def clear_history():
    global _history
    _history = []