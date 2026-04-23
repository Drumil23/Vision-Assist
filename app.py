"""
VisionAssist — Conversational real-time visual guide
Run locally:
    export GEMINI_API_KEY="your_free_key"
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    streamlit run app.py
"""

import datetime
import threading
import platform
import io
import cv2
import numpy as np
import streamlit as st
from streamlit_mic_recorder import mic_recorder

from utils.inference import get_engine
from utils.audio     import speak, play_pending

try:
    from utils.voice import listen_once
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

IS_LOCAL = platform.system() == "Darwin"

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VisionAssist",
    page_icon="🦯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], section.main,
[data-testid="stHeader"] {
  background: #07070f !important;
  font-family: 'Inter', system-ui, sans-serif !important;
}
.block-container {
  padding: 2.4rem 2rem 2rem !important;
  max-width: 1440px !important;
}
[data-testid="stDecoration"] { display:none !important; }
[data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; }
html, body, p, span, div, label, li { color: #c9cad4 !important; }

.live-badge {
  display:inline-flex; align-items:center; gap:6px;
  background:#0a160d; border:1px solid #122918;
  border-radius:20px; padding:3px 10px 3px 7px;
  font-size:0.68rem; font-weight:700; color:#34d399 !important;
  text-transform:uppercase; letter-spacing:0.1em; margin-bottom:9px;
}
.live-dot {
  width:7px; height:7px; border-radius:50%;
  background:#34d399; box-shadow:0 0 7px #34d399;
  animation:ldot 1.5s ease-in-out infinite;
}
@keyframes ldot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.35;transform:scale(.75)} }
[data-testid="stImage"] img { border-radius:11px !important; width:100% !important; }

[data-testid="stTextInput"] input {
  background:#0d0d1a !important; border:1px solid #1c1c30 !important;
  border-radius:12px !important; color:#e2e8f0 !important;
  font-size:0.9rem !important; padding:0.55rem 1rem !important;
}
[data-testid="stTextInput"] input:focus {
  border-color:#7c3aed !important; box-shadow:0 0 0 3px #7c3aed18 !important;
}
[data-testid="stTextInput"] input::placeholder { color:#2e2e4a !important; }
[data-testid="stTextInput"] label { display:none !important; }

.stButton button {
  background:linear-gradient(135deg,#6d28d9,#4f46e5) !important;
  border:none !important; border-radius:12px !important;
  color:#fff !important; font-weight:600 !important;
  font-size:0.85rem !important; padding:0.55rem 1.1rem !important;
  transition: opacity .15s, transform .1s !important;
}
.stButton button:hover  { opacity:.85 !important; }
.stButton button:active { transform:scale(.97) !important; }
.stButton button:disabled { opacity:.35 !important; }

[data-testid="stToggle"] p,
[data-testid="stToggle"] span { color:#374151 !important; font-size:0.8rem !important; }

.chat-heading {
  font-size:0.65rem; font-weight:700; text-transform:uppercase;
  letter-spacing:0.14em; color:#1e1e38 !important;
  padding-bottom:8px; border-bottom:1px solid #0f0f20; margin-bottom:10px;
}
.bubble-wrap { display:flex; flex-direction:column; gap:8px; margin-bottom:6px; }
.bubble { max-width:92%; padding:9px 13px; border-radius:14px; font-size:0.9rem; line-height:1.6; }
.bubble.user {
  align-self:flex-end;
  background:linear-gradient(135deg,#4c1d95,#3730a3);
  color:#e9d5ff !important; border-radius:14px 14px 3px 14px;
}
.bubble.ai {
  align-self:flex-start; background:#0f0f1e; border:1px solid #1a1a30;
  color:#e2e8f0 !important; border-radius:14px 14px 14px 3px;
}
.bubble.system {
  align-self:center; background:#0a0a14; border:1px solid #111120;
  color:#374151 !important; font-size:0.75rem; text-align:center;
  border-radius:8px; padding:5px 10px;
}
.bubble-time { font-size:0.65rem; color:#374151 !important; margin-top:2px; text-align:right; }

.scard { background:#0b0b17; border:1px solid #13132a; border-radius:12px; padding:9px 12px; margin-bottom:7px; }
.scard-label { font-size:0.62rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:4px; }
.scard-label.ocr    { color:#0ea5e9 !important; }
.scard-label.finder { color:#10b981 !important; }
.scard-body { font-size:0.85rem; color:#9ca3af !important; line-height:1.55; }

.mic-on {
  display:inline-flex; align-items:center; gap:7px; background:#150a0a;
  border:1px solid #3b1212; border-radius:10px; padding:6px 12px;
  font-size:0.82rem; color:#f87171 !important; animation:micon 0.9s ease-in-out infinite;
}
@keyframes micon { 0%,100%{opacity:1} 50%{opacity:.5} }

.status-mono {
  font-family:'SF Mono','Fira Code',monospace !important; font-size:0.73rem;
  background:#080810; border:1px solid #0f0f1e; border-radius:9px;
  padding:9px 11px; line-height:1.9; color:#2d2d4a !important;
}
::-webkit-scrollbar { width:3px; }
::-webkit-scrollbar-thumb { background:#131325; border-radius:2px; }
</style>
""", unsafe_allow_html=True)

# ── session state ──────────────────────────────────────────────────────────────
for k, v in [
    ("messages", []),
    ("listening", False),
    ("voice_transcript", ""),
    ("processing", False),
    ("pending_ai", []),
    ("pending_audio", []),
    ("muted", False),
]:
    if k not in st.session_state:
        st.session_state[k] = v

def ts():
    return datetime.datetime.now().strftime("%H:%M")

def add_message(role: str, text: str):
    st.session_state.messages.append({"role": role, "text": text, "time": ts()})

def add_and_speak(role: str, text: str):
    add_message(role, text)
    if role == "ai" and not st.session_state.muted:
        speak(text)

# ── singletons ─────────────────────────────────────────────────────────────────
@st.cache_resource
def get_local_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

@st.cache_resource
def get_cached_engine():
    eng = get_engine()
    def _on_proactive(text: str):
        st.session_state.pending_ai.append({"role": "ai", "text": text, "time": ts()})
    eng.on_ai_message = _on_proactive
    return eng

engine = get_cached_engine()

# play any queued audio on main thread
if not st.session_state.muted:
    play_pending()

# wire proactive callback
def _on_proactive(text: str):
    st.session_state.pending_ai.append({"role": "ai", "text": text, "time": ts()})

engine.on_ai_message = _on_proactive

# flush proactive messages from background thread
if st.session_state.pending_ai:
    for msg in st.session_state.pending_ai:
        st.session_state.messages.append(msg)
        if not st.session_state.muted and msg["text"]:
            speak(msg["text"])
    st.session_state.pending_ai = []

# ── header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:14px; margin-bottom:0.2rem;">
  <div style="width:46px; height:46px; border-radius:13px;
    background:linear-gradient(135deg,#6d28d9,#2563eb);
    display:flex; align-items:center; justify-content:center;
    font-size:22px; flex-shrink:0; box-shadow:0 0 22px #6d28d966;">🦯</div>
  <div>
    <div style="font-size:1.85rem; font-weight:700; letter-spacing:-0.03em;
      line-height:1.1; background:linear-gradient(120deg,#a78bfa 0%,#60a5fa 50%,#34d399 100%);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      color:#a78bfa;">VisionAssist</div>
    <div style="font-size:0.78rem; color:#374151; margin-top:2px;">
      Real-time AI visual guide · Always watching · Always ready
    </div>
  </div>
</div>
<div style="height:1px; margin:0.8rem 0 1.1rem;
  background:linear-gradient(90deg,#7c3aed22,#2563eb18,transparent);"></div>
""", unsafe_allow_html=True)

# ── controls ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([4, 1.3, 1.1, 0.8])

with c1:
    typed_msg = st.text_input("msg",
        placeholder="Type a question or use mic below…",
        key="typed_input", label_visibility="collapsed")
with c2:
    mic_btn = st.button("🎙 Ask with voice", key="mic_btn",
                        disabled=(not VOICE_AVAILABLE and not True))
with c3:
    send_btn = st.button("➤ Send", key="send_btn")
with c4:
    st.session_state.muted = st.toggle("🔇", value=False, key="mute")

engine.on_speak = None if st.session_state.muted else speak

# ── handle typed send ──────────────────────────────────────────────────────────
if send_btn and typed_msg.strip():
    add_and_speak("user", typed_msg.strip())
    st.session_state.processing = True

# ── voice: local mac uses listen_once, cloud uses browser mic recorder ─────────
if IS_LOCAL and VOICE_AVAILABLE:
    if mic_btn and not st.session_state.listening:
        st.session_state.listening = True
        st.rerun()
    if st.session_state.listening:
        st.markdown('<div class="mic-on">🔴 &nbsp;Listening… speak now</div>',
                    unsafe_allow_html=True)
        result = {"text": ""}
        def _listen():
            result["text"] = listen_once(timeout=6, phrase_limit=10)
        t = threading.Thread(target=_listen)
        t.start(); t.join()
        transcript = result["text"].strip()
        st.session_state.listening = False
        if transcript:
            st.session_state.voice_transcript = transcript
            add_and_speak("user", transcript)
            st.session_state.processing = True
        else:
            add_message("system", "Didn't catch that — try again")
        st.rerun()
else:
    # cloud: browser mic recorder — no pyaudio needed
    st.markdown(
        '<p style="font-size:0.72rem;color:#374151;margin:4px 0 4px">🎙 Hold to speak</p>',
        unsafe_allow_html=True)
    audio = mic_recorder(
        start_prompt="🎙 Hold & speak",
        stop_prompt="⏹ Release to send",
        just_once=True,
        key="mic_rec",
    )
    if audio and audio.get("bytes"):
        from models.scene import transcribe
        with st.spinner("Transcribing…"):
            transcript = transcribe(audio["bytes"], mime_type="audio/wav")
        if transcript.strip():
            add_and_speak("user", transcript.strip())
            st.session_state.processing = True
            st.rerun()
        else:
            add_message("system", "Didn't catch that — try again")

# ── process AI reply ───────────────────────────────────────────────────────────
if st.session_state.processing:
    st.session_state.processing = False
    last_user = next(
        (m["text"] for m in reversed(st.session_state.messages) if m["role"] == "user"),
        None)
    if last_user:
        with st.spinner(""):
            reply = engine.send_message(last_user)
        add_and_speak("ai", reply)
    st.rerun()

# ── layout ─────────────────────────────────────────────────────────────────────
cam_col, chat_col = st.columns([3, 2], gap="large")

# ── camera ─────────────────────────────────────────────────────────────────────
with cam_col:
    st.markdown(
        '<div class="live-badge"><div class="live-dot"></div>Live</div>',
        unsafe_allow_html=True)

    if IS_LOCAL:
        cap = get_local_camera()

        @st.fragment(run_every=0.5)
        def camera_fragment():
            cap = get_local_camera()
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera not accessible.")
                return
            engine.push_frame(frame)
            display = (engine.annotated if engine.annotated is not None
                       and engine.query else frame)
            st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), width=660)

        camera_fragment()
    else:
        cam_img = st.camera_input(
            "Point camera at your surroundings",
            key="browser_cam",
            label_visibility="collapsed",
        )
        if cam_img is not None:
            arr   = np.frombuffer(cam_img.getvalue(), np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                engine.push_frame(frame)
                display = (engine.annotated if engine.annotated is not None
                           and engine.query else frame)
                st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), width=660)
        else:
            st.markdown(
                '<div style="height:280px;display:flex;align-items:center;'
                'justify-content:center;background:#0b0b17;border-radius:12px;'
                'color:#2d2d45;font-size:0.9rem;text-align:center;padding:1rem;">'
                '📷 Click Take Photo to capture a frame and get guidance</div>',
                unsafe_allow_html=True)

    st.markdown('<div style="margin-top:10px"></div>', unsafe_allow_html=True)
    engine.query = st.text_input(
        "Find object",
        placeholder='Find something specific: "my keys", "the door"…',
        key="finder_input", label_visibility="collapsed").strip()

# ── chat panel ─────────────────────────────────────────────────────────────────
with chat_col:
    st.markdown('<div class="chat-heading">Conversation</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown(
            '<div class="bubble system">👋 Hey! Ask me anything — I can see what\'s around you.</div>',
            unsafe_allow_html=True)
    else:
        html = '<div class="bubble-wrap">'
        for m in st.session_state.messages[-20:]:
            html += (f'<div class="bubble {m["role"]}">{m["text"]}'
                     f'<div class="bubble-time">{m["time"]}</div></div>')
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    if st.session_state.messages:
        if st.button("↺ Clear conversation", key="clear_btn"):
            st.session_state.messages = []
            from models.scene import clear_history
            clear_history()
            st.rerun()

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    @st.fragment(run_every=3)
    def side_info():
        if engine.ocr_result:
            st.markdown(
                f'<div class="scard"><div class="scard-label ocr">📄 Text visible</div>'
                f'<div class="scard-body">{engine.ocr_result}</div></div>',
                unsafe_allow_html=True)
        if engine.finder_result and engine.query:
            st.markdown(
                f'<div class="scard"><div class="scard-label finder">📍 {engine.query}</div>'
                f'<div class="scard-body">{engine.finder_result}</div></div>',
                unsafe_allow_html=True)
        if not engine.ocr_result and not engine.finder_result:
            st.markdown(
                f'<div class="status-mono">'
                f'Scene &nbsp;&nbsp; {engine.scene_status}<br>'
                f'Text &nbsp;&nbsp;&nbsp;&nbsp; {engine.ocr_status}<br>'
                f'Finder &nbsp;&nbsp; {engine.finder_status}</div>',
                unsafe_allow_html=True)
    side_info()