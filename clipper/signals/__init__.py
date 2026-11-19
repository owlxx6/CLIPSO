"""Signals: each module turns one source of evidence into a per-hop "excitement" curve (0 = nothing, ~8 = huge).

All curves share the same time grid: index i covers [i * hop_s, (i + 1) * hop_s).
"""

from __future__ import annotations

import math

import numpy as np


def timeline_len(duration: float, hop_s: float) -> int:
    return max(1, int(math.ceil(duration / hop_s)))


def rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    """Median of a centred window, computed in blocks to keep memory flat on 10-hour VODs."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.copy()
    window = max(1, int(window)) | 1
    if window >= x.size:
        return np.full_like(x, float(np.median(x)))
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    out = np.empty_like(x)
    step = max(1, 2_000_000 // window)
    for i in range(0, x.size, step):
        out[i:i + step] = np.median(view[i:i + step], axis=1)
    return out


def rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return np.asarray(x, dtype=np.float32)
    return np.convolve(x, np.ones(window, dtype=np.float32), mode="same").astype(np.float32)


def bumps(times: list[float], weights: list[float], n: int, hop_s: float, sigma_s: float) -> np.ndarray:
    """Sum of Gaussian bumps (peak height = weight) on the timeline."""
    out = np.zeros(n, dtype=np.float32)
    if not times:
        return out
    sigma = max(hop_s, sigma_s) / hop_s
    radius = int(math.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2).astype(np.float32)
    for t, w in zip(times, weights):
        c = int(t / hop_s)
        lo, hi = c - radius, c + radius + 1
        k_lo, k_hi = max(0, -lo), len(kernel) - max(0, hi - n)
        lo, hi = max(0, lo), min(n, hi)
        if lo < hi:
            out[lo:hi] += w * kernel[k_lo:k_hi]
    return out


def shift(x: np.ndarray, seconds: float, hop_s: float) -> np.ndarray:
    """Move a curve earlier in time (positive seconds) to compensate for reaction delay."""
    k = int(round(seconds / hop_s))
    if k <= 0 or x.size == 0:
        return x
    out = np.zeros_like(x)
    out[:-k] = x[k:]
    return out
