"""
TTS via gTTS. Returns MP3 bytes that can be played via st.audio().
Thread-safe — does NOT call st.audio() directly.
app.py calls play_pending() on the main Streamlit thread.
"""

import io
import streamlit as st
from gtts import gTTS


def make_audio_bytes(text: str) -> bytes | None:
    """Convert text to MP3 bytes. Returns None on failure."""
    if not text or not text.strip():
        return None
    try:
        buf = io.BytesIO()
        gTTS(text=text.strip(), lang="en", slow=False).write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


def speak(text: str) -> None:
    """
    Queue audio for playback on the main thread.
    Safe to call from background threads.
    """
    audio_bytes = make_audio_bytes(text)
    if audio_bytes:
        if "pending_audio" not in st.session_state:
            st.session_state.pending_audio = []
        st.session_state.pending_audio.append(audio_bytes)


def play_pending() -> None:
    """
    Play all queued audio. Must be called from the main Streamlit thread.
    """
    pending = st.session_state.get("pending_audio", [])
    if pending:
        for audio_bytes in pending:
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
        st.session_state.pending_audio = []
"""
Server-side TTS using gTTS.
Saves audio to a BytesIO buffer and plays via st.audio(autoplay=True).
Works on Streamlit Cloud — no pyaudio, no browser Speech API needed.
"""

import io
import streamlit as st
from gtts import gTTS


def speak(text: str) -> None:
    """Convert text to speech and autoplay in the browser."""
    if not text or not text.strip():
        return
    try:
        buf = io.BytesIO()
        gTTS(text=text.strip(), lang="en", slow=False).write_to_fp(buf)
        buf.seek(0)
        st.audio(buf, format="audio/mp3", autoplay=True)
    except Exception:
        pass  # never crash the app over audio