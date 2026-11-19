"""Clips made by viewers on Twitch (Helix "Get Clips" JSON): crowd-sourced highlight markers.

Accepted input: {"data": [{"vod_offset": 3600, "duration": 30, "view_count": 1200, ...}]} or the bare list.
vod_offset is where the clip starts in the VOD; it can be null for very recent clips, which are skipped.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from clipper.signals import bumps, timeline_len


def load_clips(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    items = data.get("data", []) if isinstance(data, dict) else data
    clips = []
    for item in items:
        if item.get("vod_offset") is None:
            continue
        clips.append({
            "vod_offset": float(item["vod_offset"]),
            "duration": float(item.get("duration") or 30.0),
            "view_count": int(item.get("view_count") or 0),
            "title": item.get("title", ""),
        })
    return clips


def clips_curve(clips: list[dict], duration: float, hop_s: float, vcfg: dict) -> np.ndarray:
    n = timeline_len(duration, hop_s)
    if not clips:
        return np.zeros(n, dtype=np.float32)
    frac = float(vcfg.get("peak_fraction", 0.75))
    times = [c["vod_offset"] + c["duration"] * frac for c in clips]
    weights = [1.0 + math.log1p(c["view_count"]) for c in clips]
    curve = bumps(times, weights, n, hop_s, float(vcfg.get("sigma_s", 6.0)))
    return np.clip(curve, 0.0, 8.0)
