"""Facecam detection and layout choice.

The facecam is a fixed rectangle with a face in it. We sample frames across the VOD, run a face detector
(YuNet, bundled ONNX model, MIT licence; OpenCV's Haar cascade as a fallback on OpenCV 4) and keep the position
where a face shows up again and again (in-game faces move around).
Manual boxes (streamer config or `clipper layout --set`) always win over detection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from clipper import config
from clipper.config import PACKAGE_DIR
from clipper.media import bgr_frame_at

Box = dict  # {"x": .., "y": .., "w": .., "h": ..} as fractions of the frame
YUNET_MODEL = PACKAGE_DIR / "assets" / "models" / "face_detection_yunet_2023mar.onnx"


class FaceFinder:
    """Returns face boxes as fractions of the frame."""

    def __init__(self, score_threshold: float = 0.7) -> None:
        import cv2

        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except AttributeError:
            pass
        self.cv2 = cv2
        self.yunet = None
        self.haar = None
        if YUNET_MODEL.exists() and hasattr(cv2, "FaceDetectorYN"):
            self.yunet = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320), score_threshold, 0.3, 50)
            self.name = "yunet"
        else:
            cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
            cascade = cv2.CascadeClassifier(str(Path(cascade_dir) / "haarcascade_frontalface_default.xml"))
            if cascade.empty():
                raise RuntimeError("no face detector: the YuNet model is missing and this OpenCV has no Haar cascade")
            self.haar = cascade
            self.name = "haar"

    def find(self, bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
        h, w = bgr.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(bgr)
            if faces is None:
                return []
            return [(float(f[0]) / w, float(f[1]) / h, float(f[2]) / w, float(f[3]) / h) for f in faces]
        gray = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2GRAY)
        found = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        return [(x / w, y / h, fw / w, fh / h) for (x, y, fw, fh) in found]


def _best_edge(scores: np.ndarray, lo: int, hi: int, threshold: float, outermost: str) -> int | None:
    """Among the strong edges in [lo, hi), the one farthest from the face ('low' or 'high' index side)."""
    lo, hi = max(0, lo), min(len(scores), hi)
    if hi - lo < 1:
        return None
    seg = scores[lo:hi]
    if float(seg.max()) < threshold:
        return None
    strong = np.flatnonzero(seg >= max(threshold, 0.6 * float(seg.max())))
    return lo + int(strong[0] if outermost == "low" else strong[-1])


def refine_box(face: tuple[float, float, float, float], edge_x: np.ndarray, edge_y: np.ndarray,
               lcfg: dict) -> dict:
    """Grow the face box to the facecam overlay: its border is an edge that stays put while gameplay moves."""
    h, w = edge_y.shape[0] + 1, edge_x.shape[1] + 1
    x0, y0 = face[0] * w, face[1] * h
    x1, y1 = x0 + face[2] * w, y0 + face[3] * h
    fw, fh = x1 - x0, y1 - y0
    rows = slice(int(y0), max(int(y0) + 1, int(y1)))
    cols = slice(int(x0), max(int(x0) + 1, int(x1)))
    col_score = edge_x[rows, :].mean(axis=0)  # strength of vertical lines between column i and i+1
    row_score = edge_y[:, cols].mean(axis=1)  # strength of horizontal lines between row j and j+1
    thr_x = max(6.0, 4.0 * float(np.median(col_score)))
    thr_y = max(6.0, 4.0 * float(np.median(row_score)))
    ex, ey = float(lcfg.get("facecam_expand_w", 2.8)), float(lcfg.get("facecam_expand_h", 2.6))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + 0.15 * fh

    left = _best_edge(col_score, int(x0 - 3 * fw), int(x0) - 1, thr_x, "low")
    right = _best_edge(col_score, int(x1) + 1, int(x1 + 3 * fw), thr_x, "high")
    top = _best_edge(row_score, int(y0 - 2 * fh), int(y0) - 1, thr_y, "low")
    bottom = _best_edge(row_score, int(y1) + 1, int(y1 + 4 * fh), thr_y, "high")

    bx0 = left + 1 if left is not None else max(0.0, cx - ex * fw / 2)
    bx1 = right + 1 if right is not None else min(float(w), cx + ex * fw / 2)
    by0 = top + 1 if top is not None else max(0.0, cy - ey * fh / 2)
    by1 = bottom + 1 if bottom is not None else min(float(h), cy + ey * fh / 2)
    box = {"x": bx0 / w, "y": by0 / h, "w": (bx1 - bx0) / w, "h": (by1 - by0) / h}
    return {k: round(float(v), 4) for k, v in box.items()}


def detect_facecam(path: str | Path, duration: float, src_size: tuple[int, int], lcfg: dict) -> dict | None:
    samples = int(lcfg.get("facecam_samples", 24))
    finder = FaceFinder()
    hits: list[tuple[float, float, float, float]] = []
    edge_x = edge_y = None
    frames = 0
    for t in np.linspace(duration * 0.05, duration * 0.95, samples):
        frame = bgr_frame_at(path, float(t), src_size, width=640)
        if frame is None:
            continue
        hits.extend(finder.find(frame))
        gray = frame.mean(axis=2).astype(np.float32)
        gx, gy = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
        edge_x = gx if edge_x is None else edge_x + gx
        edge_y = gy if edge_y is None else edge_y + gy
        frames += 1
    if not hits:
        return None
    centers = np.array([(x + w / 2, y + h / 2) for x, y, w, h in hits])
    neighbours = [(np.linalg.norm(centers - c, axis=1) < 0.06).sum() for c in centers]
    best = int(np.argmax(neighbours))
    group = [hits[i] for i in range(len(hits)) if np.linalg.norm(centers[i] - centers[best]) < 0.06]
    confidence = len(group) / max(1, samples)
    if confidence < float(lcfg.get("facecam_min_hit_fraction", 0.3)):
        return None
    face = tuple(float(np.median([g[k] for g in group])) for k in range(4))
    box = refine_box(face, edge_x / frames, edge_y / frames, lcfg)  # type: ignore[operator]
    return {"facecam": box, "confidence": round(confidence, 2)}


def layout_file(workdir: Path) -> Path:
    return workdir / "layout.json"


def save_layout(workdir: Path, layout: dict) -> None:
    with open(layout_file(workdir), "w", encoding="utf-8") as fh:
        json.dump(layout, fh, indent=2)


def load_layout(workdir: Path) -> dict | None:
    path = layout_file(workdir)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def resolve(cfg: dict, workdir: Path, kind: str | None = None) -> dict:
    """Return {"kind": stacked|blur_fit|center_crop, "facecam": box|None}."""
    kind = kind or config.get(cfg, "layout.default", "auto")
    manual = config.get(cfg, "streamer.facecam")
    saved = load_layout(workdir) or {}
    facecam = manual or saved.get("facecam")
    if kind == "auto":
        kind = saved.get("kind") or ("stacked" if facecam else "blur_fit")
    if kind == "stacked" and not facecam:
        kind = "blur_fit"
    return {"kind": kind, "facecam": facecam}


def parse_box(text: str) -> Box:
    parts = [float(p) for p in text.replace(" ", "").split(",")]
    if len(parts) != 4 or not all(0.0 <= p <= 1.0 for p in parts):
        raise ValueError("box must be x,y,w,h as fractions of the frame, e.g. 0.72,0.04,0.26,0.30")
    return {"x": parts[0], "y": parts[1], "w": parts[2], "h": parts[3]}


def preview(path: str | Path, t: float, src_size: tuple[int, int], layout: dict, cfg: dict, out_png: Path) -> Path:
    """Save a frame with the facecam box (red) and the gameplay crop of the stacked layout (green)."""
    import cv2

    frame = bgr_frame_at(path, t, src_size, width=960)
    if frame is None:
        raise RuntimeError("could not read a frame for the preview")
    h, w = frame.shape[:2]
    fc = layout.get("facecam")
    if fc:
        cv2.rectangle(frame, (int(fc["x"] * w), int(fc["y"] * h)),
                      (int((fc["x"] + fc["w"]) * w), int((fc["y"] + fc["h"]) * h)), (0, 0, 255), 3)
    rw, rh = cfg["render"]["width"], cfg["render"]["height"]
    band = cfg["render"]["facecam_band_h"] if layout.get("kind") == "stacked" else 0
    aspect = rw / max(1, rh - band)
    gw = min(w, int(h * aspect))
    cx = float(config.get(cfg, "layout.gameplay_center_x", 0.5))
    gx = int((w - gw) * cx)
    cv2.rectangle(frame, (gx, 0), (gx + gw, h - 1), (0, 200, 0), 3)
    cv2.putText(frame, layout.get("kind", "?"), (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), frame)
    return out_png
