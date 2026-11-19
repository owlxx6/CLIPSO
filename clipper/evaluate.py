"""Measure detection against hand-marked highlights and search signal weights that fit your content."""

from __future__ import annotations

import csv
import datetime as dt
import random
from pathlib import Path

import numpy as np
import yaml

from clipper import detect
from clipper.media import parse_ts

GRID = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


def load_truth(path: str | Path) -> list[tuple[float, float]]:
    """CSV with columns start,end (seconds or HH:MM:SS) and an optional label column (1 = highlight)."""
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("label", "1")).strip() in ("0", "no", "false"):
                continue
            rows.append((parse_ts(row["start"]), parse_ts(row["end"])))
    return sorted(rows)


def truth_from_labels(labels: list[dict]) -> list[tuple[float, float]]:
    return sorted((float(r["start_s"]), float(r["end_s"])) for r in labels if int(r["label"]) == 1)


def match(start: float, end: float, peak: float, truth: list[tuple[float, float]], min_overlap_s: float = 3.0) -> int | None:
    choices = []
    for i, (ts, te) in enumerate(truth):
        inter = max(0.0, min(end, te) - max(start, ts))
        inside = ts <= peak <= te
        if inside or inter >= min(min_overlap_s, 0.5 * (te - ts)):
            choices.append((inside, inter, -abs(peak - (ts + te) / 2), -i))
    return -max(choices)[-1] if choices else None


def evaluate(cands: list[dict], truth: list[tuple[float, float]], k: int = 10) -> dict:
    top = cands[:k]
    remaining = list(range(len(truth)))
    hits = []
    for c in top:
        pos = match(c["start"], c["end"], c["peak"], [truth[i] for i in remaining])
        hits.append(None if pos is None else remaining.pop(pos))
    found = {h for h in hits if h is not None}
    precision = len(found) / len(top) if top else 0.0
    recall = len(found) / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"k": k, "candidates": len(top), "truth": len(truth), "precision": round(precision, 3),
            "recall": round(recall, 3), "f1": round(f1, 3), "hits": hits}


def tune(signals: dict[str, np.ndarray], cfg: dict, duration: float, transcript: dict | None,
         truth: list[tuple[float, float]], k: int = 10, trials: int = 300, seed: int = 0,
         target: str = "campaign") -> dict:
    """Random search over signal weights; returns the best weights with before/after metrics."""
    present = [n for n in detect.SIGNAL_NAMES if n in signals and float(np.max(signals[n], initial=0)) > 0]
    base_w = dict(cfg["detection"]["weights"])

    def run(weights: dict) -> dict:
        cands, _ = detect.detect(signals, cfg, duration, transcript, target, weights=weights)
        return evaluate([c.as_dict() for c in cands], truth, k)

    baseline = run(base_w)
    best_m, best_w = baseline, base_w
    rng = random.Random(seed)
    for _ in range(trials):
        trial = dict(base_w)
        trial.update({n: rng.choice(GRID) for n in present})
        if not any(trial.get(n, 0) for n in present):
            continue
        m = run(trial)
        if (m["f1"], m["precision"]) > (best_m["f1"], best_m["precision"]):
            best_m, best_w = m, trial
    return {"weights": best_w, "metrics": best_m, "baseline": baseline, "signals": present}


def write_tuned(path: Path, weights: dict, metrics: dict, source_label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Written by `clipper tune` on {dt.date.today().isoformat()} from {source_label}\n"
              f"# precision@{metrics['k']}={metrics['precision']} recall={metrics['recall']} f1={metrics['f1']}\n")
    body = yaml.safe_dump({"detection": {"weights": {k: float(v) for k, v in weights.items()}}}, sort_keys=False)
    path.write_text(header + body, encoding="utf-8")
    return path
