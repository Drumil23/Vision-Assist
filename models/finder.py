"""
Tool 2 — Object Finder
Models: OpenCLIP ViT-B-32 (laion2b_s34b_b79k) + Depth Anything V2 Small
Device: Apple Silicon MPS (falls back to CPU)
"""

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from transformers import pipeline

DEVICE          = "mps" if torch.backends.mps.is_available() else "cpu"
CLIP_MODEL      = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
DEPTH_MODEL_ID  = "depth-anything/Depth-Anything-V2-Small-hf"

GRID_ROWS = 3
GRID_COLS = 3
QUADRANT_NAMES = [
    ["top-left",    "top-center",    "top-right"],
    ["middle-left", "center",        "middle-right"],
    ["bottom-left", "bottom-center", "bottom-right"],
]

_clip_model      = None
_clip_preprocess = None
_clip_tokenizer  = None
_depth_pipe      = None


def _load_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED
    )
    _clip_model = _clip_model.to(DEVICE).eval()
    _clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL)


def _load_depth():
    global _depth_pipe
    if _depth_pipe is not None:
        return
    _depth_pipe = pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL_ID,
        device=DEVICE,
    )


def _make_crops(bgr: np.ndarray):
    """Slice frame into a 3x3 grid; yield (pil_crop, label, bbox)."""
    h, w  = bgr.shape[:2]
    rh, rw = h // GRID_ROWS, w // GRID_COLS
    crops = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x1, y1 = c * rw, r * rh
            x2, y2 = x1 + rw, y1 + rh
            crop_rgb = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            crops.append((Image.fromarray(crop_rgb),
                          QUADRANT_NAMES[r][c],
                          (x1, y1, x2, y2)))
    # full frame as fallback
    crops.append((
        Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
        "the full scene",
        (0, 0, w, h),
    ))
    return crops


def _depth_label(pil_crop: Image.Image) -> str:
    out   = _depth_pipe(pil_crop)
    dmap  = np.array(out["depth"], dtype=np.float32)
    mx    = dmap.max() or 1.0
    ratio = dmap.mean() / mx
    if ratio > 0.66:
        return "very close — within arm's reach"
    if ratio > 0.33:
        return "a few metres away"
    return "far away"


def find_object(bgr_frame: np.ndarray, query: str):
    """
    Locate `query` in `bgr_frame`.

    Returns
    -------
    location  : str          human-readable quadrant
    depth_str : str          distance description
    annotated : np.ndarray   BGR frame with bounding box
    """
    _load_clip()
    _load_depth()

    crops  = _make_crops(bgr_frame)
    tokens = _clip_tokenizer([f"a photo of {query}"]).to(DEVICE)
    scores = []

    with torch.no_grad(), torch.autocast(device_type=DEVICE):
        txt_feat = _clip_model.encode_text(tokens)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        for (crop_pil, _, _) in crops:
            img_t    = _clip_preprocess(crop_pil).unsqueeze(0).to(DEVICE)
            img_feat = _clip_model.encode_image(img_t)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            scores.append((100.0 * img_feat @ txt_feat.T).item())

    best              = int(np.argmax(scores))
    crop_pil, location, (x1, y1, x2, y2) = crops[best]
    depth_str         = _depth_label(crop_pil)

    annotated = bgr_frame.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 100), 3)
    cv2.putText(annotated, query,
                (x1 + 6, y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 100), 2)

    return location, depth_str, annotated