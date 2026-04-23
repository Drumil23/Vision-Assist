"""
Inference engine.
- Scene loop : proactively describes surroundings every 60s (or on change)
              and speaks the result automatically
- OCR loop  : reads visible text every 5s
- Finder    : locates specific objects every 3s
- send_message: handles user questions, always speaks reply
"""

import threading
import time
import cv2
import numpy as np
from PIL import Image

import torch
import torchvision      # noqa
import easyocr          # noqa
import open_clip        # noqa
import google.genai     # noqa

PROACTIVE_PROMPT = (
    "Look at what's around me and give me a brief, helpful description. "
    "Tell me where I am, what's immediately ahead, and one thing I should know or do. "
    "Be warm and conversational, max 2 sentences."
)

MIN_PROACTIVE_INTERVAL = 60   # seconds between auto-descriptions


class InferenceEngine:
    def __init__(self):
        self._frame      = None
        self._prev_frame = None
        self._lock       = threading.Lock()
        self._running    = False

        self.query         = ""
        self.ocr_result    = ""
        self.finder_result = ""
        self.annotated     = None

        self.scene_status  = "⏳ starting…"
        self.ocr_status    = "⏳ starting…"
        self.finder_status = "⏳ no query"

        self.on_speak      = None   # speak(text) callback
        self.on_ai_message = None   # on_ai_message(text) → adds to chat UI

        self._last_proactive = 0.0  # timestamp of last auto-description

    # ── frame exchange ────────────────────────────────────────────────────────
    def push_frame(self, bgr: np.ndarray):
        with self._lock:
            self._frame = bgr.copy()

    def _get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _get_pil(self):
        f = self._get_frame()
        return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) if f is not None else None

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self):
        if self._running:
            return
        self._running = True
        for fn in (self._scene_loop, self._ocr_loop, self._finder_loop):
            threading.Thread(target=fn, daemon=True).start()

    def stop(self):
        self._running = False

    # ── scene change detection ────────────────────────────────────────────────
    def _scene_changed(self, frame: np.ndarray) -> bool:
        if self._prev_frame is None:
            return True
        diff = cv2.absdiff(
            cv2.resize(frame, (80, 60)),
            cv2.resize(self._prev_frame, (80, 60)),
        )
        return float(diff.mean()) > 10.0

    # ── user message (question / command) ─────────────────────────────────────
    def send_message(self, user_text: str) -> str:
        from models.scene import chat
        pil   = self._get_pil()
        reply = chat(user_text, pil_image=pil)
        if self.on_speak:
            self.on_speak(reply)
        return reply

    # ── proactive scene loop ──────────────────────────────────────────────────
    def _scene_loop(self):
        from models.scene import chat, set_frame
        while self._running:
            frame = self._get_frame()
            if frame is not None:
                now            = time.time()
                changed        = self._scene_changed(frame)
                time_ok        = (now - self._last_proactive) >= MIN_PROACTIVE_INTERVAL

                if changed and time_ok:
                    self.scene_status = "🔄 describing scene…"
                    try:
                        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        set_frame(pil)
                        description = chat(PROACTIVE_PROMPT, pil_image=pil)
                        self._prev_frame    = frame.copy()
                        self._last_proactive = time.time()
                        self.scene_status   = "✅ ok"

                        # speak it and push to chat UI
                        if self.on_speak:
                            self.on_speak(description)
                        if self.on_ai_message:
                            self.on_ai_message(description)

                    except Exception as e:
                        self.scene_status = "⚠️ paused"
                        if "429" in str(e):
                            time.sleep(90)

            time.sleep(8)   # check for changes every 8s, but only call Gemini if changed + 60s passed

    # ── OCR loop ──────────────────────────────────────────────────────────────
    def _ocr_loop(self):
        from models.reader import read_text
        while self._running:
            frame = self._get_frame()
            if frame is not None:
                self.ocr_status = "🔄 reading…"
                try:
                    pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    text, _, _ = read_text(pil, min_confidence=0.5)
                    self.ocr_result = "" if text == "No text detected." else text
                    self.ocr_status = "✅ ok"
                except Exception as e:
                    self.ocr_status = f"❌ {e}"
            time.sleep(5)

    # ── finder loop ───────────────────────────────────────────────────────────
    def _finder_loop(self):
        from models.finder import find_object
        while self._running:
            frame = self._get_frame()
            if frame is not None and self.query.strip():
                self.finder_status = f"🔄 finding '{self.query}'…"
                try:
                    loc, depth, annotated = find_object(frame, self.query.strip())
                    self.finder_result = f"{self.query} is {loc}, {depth}"
                    self.annotated     = annotated
                    self.finder_status = "✅ ok"
                except Exception as e:
                    self.finder_status = f"❌ {e}"
            else:
                self.finder_result = ""
                self.annotated     = None
                self.finder_status = "⏳ no query set"
            time.sleep(3)


_engine = None


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
        _engine.start()
    return _engine