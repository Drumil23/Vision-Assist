"""
Voice input — records from mic and transcribes using Google Speech Recognition.
Free, no API key needed for basic usage.
"""

import speech_recognition as sr

_recognizer = sr.Recognizer()
_recognizer.pause_threshold = 0.8   # stops listening after 0.8s silence
_recognizer.energy_threshold = 300  # mic sensitivity


def listen_once(timeout: int = 5, phrase_limit: int = 8) -> str:
    """
    Listen for a single spoken phrase and return transcribed text.
    timeout     : max seconds to wait for speech to start
    phrase_limit: max seconds of speech to record
    Returns transcribed string, or empty string on failure.
    """
    with sr.Microphone() as src:
        _recognizer.adjust_for_ambient_noise(src, duration=0.3)
        try:
            audio = _recognizer.listen(
                src, timeout=timeout, phrase_time_limit=phrase_limit
            )
            return _recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""
        except Exception:
            return ""