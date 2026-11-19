"""Cinematic detection: letterbox bars usually mean a cutscene, i.e. a story spoiler that Rockstar takes down.

This is a heuristic to verify on real GTA VI footage; it only flags candidates for human review.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from clipper.media import gray_frames
from clipper.signals import timeline_len


def letterbox_curve(path: str | Path, duration: float, hop_s: float, vcfg: dict) -> np.ndarray:
    n = timeline_len(duration, hop_s)
    every = float(vcfg.get("sample_every_s", 2.0))
    frac = float(vcfg.get("letterbox_bar_fraction", 0.09))
    luma_max = float(vcfg.get("letterbox_luma_max", 18))
    w, h = 160, 90
    bar = max(1, int(round(h * frac)))
    out = np.zeros(n, dtype=np.float32)
    for t, frame in gray_frames(path, fps=1.0 / every, width=w, height=h, keyframes_only=True):
        top = float(frame[:bar].mean())
        bottom = float(frame[-bar:].mean())
        middle = float(frame[bar:-bar].mean())
        if top < luma_max and bottom < luma_max and middle > luma_max + 12:
            lo, hi = int(t / hop_s), int((t + every) / hop_s)
            out[max(0, lo):min(n, hi)] = 1.0
    return out
