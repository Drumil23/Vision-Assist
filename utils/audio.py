"""
Speech output using macOS built-in 'say' command.
Far more reliable than pyttsx3 in multi-threaded Streamlit apps.
No pip install needed — it's built into every Mac.
"""

import subprocess
import threading
import queue
import platform

_IS_MAC = platform.system() == "Darwin"
_q      = queue.Queue(maxsize=2)
_proc   = None
_lock   = threading.Lock()


def _worker():
    global _proc
    while True:
        text = _q.get()
        if text is None:
            break
        with _lock:
            # kill any currently speaking process before starting new one
            if _proc and _proc.poll() is None:
                _proc.terminate()
                _proc.wait()
            _proc = subprocess.Popen(
                ["say", "-r", "175", "-v", "Samantha", text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _proc.wait()
        _q.task_done()


_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()


def speak(text: str) -> None:
    """Queue text for speech. No-op on non-Mac (e.g. Streamlit Cloud)."""
    if not _IS_MAC or not text or not text.strip():
        return
    # drain stale pending item
    while not _q.empty():
        try:
            _q.get_nowait()
            _q.task_done()
        except queue.Empty:
            break
    try:
        _q.put_nowait(text)
    except queue.Full:
        pass


def stop():
    """Immediately stop any current speech."""
    global _proc
    with _lock:
        if _proc and _proc.poll() is None:
            _proc.terminate()