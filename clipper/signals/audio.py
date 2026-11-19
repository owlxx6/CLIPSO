"""Loudness: shouting, laughing and explosions stand out against the surrounding minute."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from clipper.media import pcm_chunks
from clipper.signals import rolling_median


def loudness_db(path: str | Path, hop_s: float = 0.5, sample_rate: int = 16000) -> np.ndarray:
    """RMS level in dBFS per hop, streamed from FFmpeg (no temp WAV, flat memory on long VODs)."""
    hop = max(1, int(round(hop_s * sample_rate)))
    step = hop * 2  # int16
    rest = b""
    parts: list[np.ndarray] = []
    for chunk in pcm_chunks(path, sample_rate):
        data = rest + chunk
        n = len(data) // step
        if n:
            frames = np.frombuffer(data[: n * step], dtype="<i2").astype(np.float32).reshape(n, hop)
            rms = np.sqrt(np.mean(frames * frames, axis=1)) / 32768.0
            parts.append(20.0 * np.log10(rms + 1e-9))
        rest = data[n * step:]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.maximum(np.concatenate(parts), -90.0).astype(np.float32)


def excitement(db: np.ndarray, hop_s: float, baseline_window_s: float = 60.0, cap: float = 8.0) -> np.ndarray:
    """Robust z-score of loudness above the local median (MAD scale, floored at 1 dB)."""
    if db.size == 0:
        return db.astype(np.float32)
    base = rolling_median(db, int(baseline_window_s / hop_s))
    resid = db - base
    mad = float(np.median(np.abs(resid - np.median(resid)))) * 1.4826
    z = resid / max(mad, 1.0)
    return np.clip(z, 0.0, cap).astype(np.float32)
