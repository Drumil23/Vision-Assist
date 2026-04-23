"""
Tool 3 — Text Reader
Model : EasyOCR 1.7 (English)
Device: CPU — EasyOCR does not support MPS natively; fast enough on M5 CPU
"""

import cv2
import numpy as np
import easyocr
from PIL import Image

_reader = None


def _load():
    global _reader
    if _reader is not None:
        return
    _reader = easyocr.Reader(["en"], gpu=False, verbose=False)


def read_text(pil_image: Image.Image, min_confidence: float = 0.4):
    """
    Extract text from pil_image.

    Returns
    -------
    full_text : str           all detected text joined in reading order
    annotated : np.ndarray    BGR image with bounding boxes drawn
    raw       : list[dict]    each item has 'text', 'confidence', 'bbox'
    """
    _load()
    bgr     = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    results = _reader.readtext(bgr)

    lines, raw = [], []
    for (bbox, text, conf) in results:
        if conf < min_confidence:
            continue
        lines.append(text)
        raw.append({"text": text, "confidence": round(conf, 3), "bbox": bbox})

        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(bgr, [pts], isClosed=True, color=(0, 160, 255), thickness=2)
        cv2.putText(
            bgr, f"{text} ({conf:.0%})",
            (int(pts[0][0]), max(int(pts[0][1]) - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 160, 255), 1,
        )

    full_text = " ".join(lines).strip() or "No text detected."
    return full_text, bgr, raw