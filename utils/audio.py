"""
TTS via gTTS — works locally and on Streamlit Cloud.
speak()       : queues audio bytes (thread-safe, callable from background threads)
play_pending(): plays all queued audio via st.audio() — call from main thread only
"""

import io
import platform
import streamlit as st

IS_MAC = platform.system() == "Darwin"


def _make_mp3(text: str) -> bytes | None:
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text.strip(), lang="en", slow=False).write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


def _mac_say(text: str) -> None:
    """Fallback: use macOS say command locally for instant playback."""
    import subprocess
    subprocess.Popen(
        ["say", "-r", "175", "-v", "Samantha", text],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def speak(text: str) -> None:
    """
    Queue text for speech. Safe to call from any thread.
    On Mac locally: speaks immediately via say command.
    On Cloud: queues MP3 bytes for play_pending() to play on main thread.
    """
    if not text or not text.strip():
        return
    if IS_MAC:
        _mac_say(text)
    else:
        mp3 = _make_mp3(text)
        if mp3:
            if "pending_audio" not in st.session_state:
                st.session_state.pending_audio = []
            st.session_state.pending_audio.append(mp3)


def play_pending() -> None:
    """
    Play all queued MP3 audio via st.audio(autoplay=True).
    Must be called from the main Streamlit thread.
    No-op on Mac (say command handles it directly).
    """
    if IS_MAC:
        return
    pending = st.session_state.get("pending_audio", [])
    if pending:
        for mp3 in pending:
            st.audio(mp3, format="audio/mp3", autoplay=True)
        st.session_state.pending_audio = []