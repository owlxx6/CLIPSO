"""One edit per platform: different cut points, framing, highlight colour and cover frame.

These are genuine platform versions of the same moment (each one is posted once, on its own platform).
They are not meant to disguise copies posted on several accounts of the same platform: the duplicate guard
in dedup.py blocks that case.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from clipper.signals import transcript as tr


@dataclass
class Variant:
    platform: str
    start: float
    end: float
    zoom: float
    highlight_color: str
    cover_at: float


def _edge_options(anchor, other, peak, duration, min_s, max_s, radius, before, segments, words, cues):
    direction = -1 if before else 1
    # Include the positions needed to meet a narrower/wider campaign duration.
    anchors = {anchor, other + direction * min_s, other + direction * max_s,
               peak + direction * min_s / 2, peak + direction * max_s / 2}
    anchors = {max(0.0, min(duration, t)) for t in anchors}
    grid = {round(t + i * 0.25, 3) for t in anchors for i in range(-int(radius * 4), int(radius * 4) + 1)}
    near = lambda t: any(abs(t - a) <= radius for a in anchors)
    sentences = {s if before else e for s, e in segments if near(s if before else e)}
    nearby_words = [(s, e) for s, e in words if any(s - radius <= a <= e + radius for a in anchors)]
    word_edges = {s if before else e for s, e in nearby_words if near(s if before else e)}
    cues = {t for t in cues if near(t)}
    options = []
    for t in sorted(anchors | grid | sentences | cues | word_edges):
        if not 0 <= t <= duration or any(s < t < e for s, e in nearby_words):
            continue
        if before and t >= peak or not before and t <= peak:
            continue
        penalty = 0 if t in sentences else 0.25 if t in cues else 0.75 if t in word_edges else 1.5
        # Removing setup/payoff costs more than including a little additional context.
        loss = max(0, (anchor - t) * direction)
        options.append((t, penalty + abs(t - anchor) * 0.35 + loss))
    return options


def plan_variants(
    cand_id: int,
    start: float,
    end: float,
    peak: float,
    duration: float,
    platforms: list[str],
    cfg: dict,
    used_cuts: dict[str, list[tuple[float, float]]] | None = None,
    *, transcript: dict | None = None, cues: list[float] | None = None,
    duration_limits: tuple[float, float] | None = None, protected: bool = False,
) -> list[Variant]:
    used_cuts = used_cuts or {}
    min_dist = float(cfg["variants"]["min_cut_distance_s"])
    reuse_tol = float(cfg["variants"]["reuse_tolerance_s"])
    out: list[Variant] = []
    segments = tr.segments(transcript)
    words = [(w["start"], w["end"]) for w in tr.words(transcript)]
    cues = cues or []
    for platform in platforms:
        pcfg = cfg["platforms"][platform]
        target = cfg["targets"][pcfg.get("target", "campaign")]
        min_s, max_s = float(target["min_s"]), float(pcfg.get("max_s", target["max_s"]))
        if protected:
            min_s = min(duration, float(cfg["stream_highlights"]["min_s"]))
            max_s = min(float(cfg["stream_highlights"]["max_s"]), float(pcfg.get("max_s", duration)))
        if duration_limits:
            min_s = max(min_s, duration_limits[0])
            max_s = min(max_s, duration_limits[1])
        rng = random.Random(f"{cand_id}:{platform}")
        radius = max(float(pcfg.get("trim_jitter_s", 1.2)), float(cfg["detection"]["snap_search_s"]))
        starts = _edge_options(start, end, peak, duration, min_s, max_s, radius, True, segments, words, cues)
        ends = _edge_options(end, start, peak, duration, min_s, max_s, radius, False, segments, words, cues)
        pre = min(peak - start, float(cfg["detection"]["pre_roll_s"]), max_s * 0.5)
        post = min(end - peak, float(cfg["detection"]["post_roll_s"]), max_s * 0.3)
        best = None
        chosen = None
        for s, start_cost in starts:
            if s > peak - pre or any(abs(s - v.start) < min_dist for v in out):
                continue
            for e, end_cost in ends:
                if protected and (s > start or e < end):
                    continue
                if e < peak + post or not min_s <= e - s <= max_s:
                    continue
                if any(abs(e - v.end) < min_dist for v in out):
                    continue
                if any(abs(s - us) < reuse_tol and abs(e - ue) < reuse_tol for us, ue in used_cuts.get(platform, [])):
                    continue
                rank = (start_cost + end_cost, abs(s - start) + abs(e - end), s, e)
                if protected:
                    rank = (start - s + e - end, *rank)
                if best is None or rank < best:
                    best, chosen = rank, (s, e)
        if chosen is None:
            if protected:
                raise ValueError(f"No {platform} cut fits the reviewed interval, duration limits and distinct-cut "
                                 "rules. Review the bounds or render fewer platforms.")
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
