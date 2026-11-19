"""One edit per platform: different cut points, framing, highlight colour and cover frame.

These are genuine platform versions of the same moment (each one is posted once, on its own platform).
They are not meant to disguise copies posted on several accounts of the same platform: the duplicate guard
in dedup.py blocks that case.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Variant:
    platform: str
    start: float
    end: float
    zoom: float
    highlight_color: str
    cover_at: float


def _fits(start: float, end: float, peak: float, min_s: float, max_s: float, duration: float) -> bool:
    return (0.0 <= start < peak - 2.0 and peak + 2.0 < end <= duration and min_s - 0.01 <= end - start <= max_s + 0.01)


def plan_variants(
    cand_id: int,
    start: float,
    end: float,
    peak: float,
    duration: float,
    platforms: list[str],
    cfg: dict,
    used_cuts: dict[str, list[tuple[float, float]]] | None = None,
) -> list[Variant]:
    used_cuts = used_cuts or {}
    min_dist = float(cfg["variants"]["min_cut_distance_s"])
    reuse_tol = float(cfg["variants"]["reuse_tolerance_s"])
    out: list[Variant] = []
    for platform in platforms:
        pcfg = cfg["platforms"][platform]
        target = cfg["targets"][pcfg.get("target", "campaign")]
        min_s, max_s = float(target["min_s"]), float(pcfg.get("max_s", target["max_s"]))
        jitter = float(pcfg.get("trim_jitter_s", 1.2))
        rng = random.Random(f"{cand_id}:{platform}")
        chosen = None
        for attempt in range(200):
            spread = jitter * (1 + attempt / 50)  # widen the search if the window is tight
            s = round(start + rng.uniform(-spread, spread), 2)
            e = round(end + rng.uniform(-spread, spread), 2)
            if e - s > max_s:
                e = round(s + max_s, 2)
            if not _fits(s, e, peak, min_s, max_s, duration):
                continue
            if any(abs(s - v.start) < min_dist or abs(e - v.end) < min_dist for v in out):
                continue
            if any(abs(s - us) < reuse_tol and abs(e - ue) < reuse_tol for us, ue in used_cuts.get(platform, [])):
                continue
            chosen = (s, e)
            break
        if chosen is None:
            raise RuntimeError(f"could not plan distinct cuts for candidate {cand_id} on {platform}")
        out.append(Variant(
            platform=platform,
            start=chosen[0],
            end=chosen[1],
            zoom=float(rng.choice(pcfg.get("zoom_choices", [1.0]))),
            highlight_color=str(pcfg.get("highlight_color", "#FFE500")),
            cover_at=round(rng.uniform(0.6, 1.6), 2),
        ))
    return out
