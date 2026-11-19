"""Highlight Hunter: combine signal curves, pick peaks, cut a clean window around each peak."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from clipper import config
from clipper.signals import transcript as tr

SIGNAL_NAMES = ("audio", "speech_rate", "keywords", "chat", "viewer_clips")


@dataclass
class Candidate:
    start: float
    end: float
    peak: float
    score: float
    signals: dict[str, float] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict:
        return asdict(self)


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size == 0:
        return x
    return np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same").astype(np.float32)


def combine(signals: dict[str, np.ndarray], weights: dict[str, float], hop_s: float, smoothing_s: float) -> np.ndarray:
    n = max((len(signals[name]) for name in SIGNAL_NAMES if name in signals), default=0)
    total = np.zeros(n, dtype=np.float32)
    for name in SIGNAL_NAMES:
        arr = signals.get(name)
        weight = float(weights.get(name, 0) or 0)
        if arr is None or weight == 0 or len(arr) == 0:
            continue
        m = min(n, len(arr))
        total[:m] += weight * np.asarray(arr[:m], dtype=np.float32)
    return moving_average(total, int(round(smoothing_s / hop_s)))


def pick_peaks(score: np.ndarray, hop_s: float, min_gap_s: float, min_score: float, limit: int) -> list[int]:
    if score.size == 0:
        return []
    gap = max(1, int(round(min_gap_s / hop_s)))
    picked: list[int] = []
    for idx in np.argsort(-score, kind="stable"):
        if score[idx] < min_score or len(picked) >= limit:
            break
        if all(abs(int(idx) - p) >= gap for p in picked):
            picked.append(int(idx))
    return picked


def _quietest(db: np.ndarray | None, lo: float, hi: float, hop_s: float, fallback: float) -> float:
    if db is None or db.size == 0:
        return fallback
    i0, i1 = max(0, int(lo / hop_s)), min(db.size, int(hi / hop_s) + 1)
    if i1 <= i0:
        return fallback
    return (i0 + int(np.argmin(db[i0:i1]))) * hop_s


def snap_start(t: float, segs: list[tuple[float, float]], db: np.ndarray | None, hop_s: float, search: float) -> float:
    """Start on a sentence start (preferably before t); otherwise on the quietest moment nearby."""
    options = [s for s, _ in segs if t - search <= s <= t + 1.0]
    if options:
        return min(options, key=lambda s: abs(s - t) * (2.0 if s > t else 1.0))
    return _quietest(db, t - search / 2, t + 1.0, hop_s, t)


def snap_end(t: float, segs: list[tuple[float, float]], db: np.ndarray | None, hop_s: float, search: float) -> float:
    """End on a sentence end (preferably after t); otherwise on the quietest moment nearby."""
    options = [e for _, e in segs if t - 1.0 <= e <= t + search]
    if options:
        return min(options, key=lambda e: abs(e - t) * (2.0 if e < t else 1.0)) + 0.25
    return _quietest(db, t - 1.0, t + search / 2, hop_s, t)


def fit_bounds(start: float, end: float, peak: float, duration: float, min_s: float, max_s: float) -> tuple[float, float]:
    start, end = max(0.0, start), min(duration, end)
    if end - start < min_s:
        need = min_s - (end - start)
        add_end = min(need * 0.4, duration - end)
        end += add_end
        need -= add_end
        add_start = min(need, start)
        start -= add_start
        need -= add_start
        if need > 0:
            end = min(duration, end + need)
    if end - start > max_s:
        excess = (end - start) - max_s
        cut_start = min(excess, max(0.0, (peak - 4.0) - start))
        start += cut_start
        end -= excess - cut_start
    return round(start, 2), round(end, 2)


def _overlap(a: Candidate, b: Candidate) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return inter / max(1e-6, min(a.duration, b.duration))


def detect(
    signals: dict[str, np.ndarray],
    cfg: dict,
    duration: float,
    transcript: dict | None = None,
    target: str = "campaign",
    weights: dict[str, float] | None = None,
) -> tuple[list[Candidate], np.ndarray]:
    hop = float(config.get(cfg, "analysis.hop_s", 0.5))
    d = cfg["detection"]
    tgt = cfg["targets"][target]
    score = combine(signals, weights or d["weights"], hop, float(d["smoothing_s"]))
    peaks = pick_peaks(score, hop, float(d["min_gap_s"]), float(d["min_score"]), int(d["top_k"]) * 3)
    segs = tr.segments(transcript)
    db = signals.get("_db")
    cine = signals.get("_cinematic")
    search = float(d["snap_search_s"])
    kept: list[Candidate] = []
    for idx in peaks:
        t = idx * hop + hop / 2
        start = snap_start(t - float(d["pre_roll_s"]), segs, db, hop, search)
        end = snap_end(t + float(d["post_roll_s"]), segs, db, hop, search)
        start, end = fit_bounds(start, end, t, duration, float(tgt["min_s"]), float(tgt["max_s"]))
        lo, hi = max(0, idx - int(2 / hop)), idx + int(2 / hop) + 1
        sig = {name: round(float(np.max(signals[name][lo:hi])), 2)
               for name in SIGNAL_NAMES if name in signals and len(signals[name][lo:hi])}
        flags: dict[str, Any] = {}
        if cine is not None and cine.size:
            frac = float(np.mean(cine[int(start / hop):max(int(start / hop) + 1, int(end / hop))]))
            flags["cinematic"] = round(frac, 2)
            flags["spoiler_review"] = frac >= float(d["cinematic_flag_fraction"])
        cand = Candidate(start=start, end=end, peak=round(t, 2), score=round(float(score[idx]), 3), signals=sig,
                         flags=flags, text=tr.text_between(transcript, start, end))
        if all(_overlap(cand, k) <= float(d["max_overlap"]) for k in kept):
            kept.append(cand)
        if len(kept) >= int(d["top_k"]):
            break
    return kept, score
