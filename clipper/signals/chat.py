"""Chat velocity: on Twitch and Kick, the strongest free predictor of a highlight.

Accepted inputs:
- TwitchDownloader JSON: {"comments": [{"content_offset_seconds": 12.3, "message": {"body": "KEKW"}}]}
- a JSON list: [{"t": 12.3, "text": "KEKW"}] (keys t/time/offset and text/message/body)
- a CSV with columns offset_seconds|time|t and text|message|body (time as seconds or HH:MM:SS)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from clipper.media import parse_ts
from clipper.signals import rolling_median, rolling_sum, shift, timeline_len

TIME_KEYS = ("content_offset_seconds", "offset_seconds", "offset", "time", "t", "timestamp")
TEXT_KEYS = ("text", "message", "body", "content")


def _pick(item: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def load_chat(path: str | Path) -> list[tuple[float, str]]:
    path = Path(path)
    messages: list[tuple[float, str]] = []
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                t, text = _pick(row, TIME_KEYS), _pick(row, TEXT_KEYS)
                if t not in (None, ""):
                    messages.append((parse_ts(t), str(text or "")))
    else:
        with open(path, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        items = data.get("comments", []) if isinstance(data, dict) else data
        for item in items:
            t = _pick(item, TIME_KEYS)
            text = _pick(item, TEXT_KEYS)
            if isinstance(text, dict):
                text = text.get("body", "")
            if t is not None:
                messages.append((parse_ts(t), str(text or "")))
    messages.sort(key=lambda m: m[0])
    return messages


def chat_curve(messages: list[tuple[float, str]], duration: float, hop_s: float, ccfg: dict) -> np.ndarray:
    n = timeline_len(duration, hop_s)
    if not messages:
        return np.zeros(n, dtype=np.float32)
    emotes = {e.lower() for e in ccfg.get("emotes", []) if e.isascii()}
    emoji = [e for e in ccfg.get("emotes", []) if not e.isascii()]
    emote_weight = float(ccfg.get("emote_weight", 1.5))
    counts = np.zeros(n, dtype=np.float32)
    for t, text in messages:
        i = int(t / hop_s)
        if 0 <= i < n:
            lowered = text.lower()
            hype = bool(emotes.intersection(lowered.split())) or any(e in text for e in emoji)
            counts[i] += emote_weight if hype else 1.0
    rate = rolling_sum(counts, int(round(float(ccfg.get("window_s", 10)) / hop_s)))
    base = rolling_median(rate, int(float(ccfg.get("baseline_window_s", 300)) / hop_s))
    z = (rate - base) / np.sqrt(base + 1.0)
    z = shift(z, float(ccfg.get("delay_s", 6)), hop_s)
    return np.clip(z, 0.0, 8.0).astype(np.float32)
