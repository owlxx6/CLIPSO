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


def _edge_options(anchor, other, peak, duration, min_s, max_s, radius, before, segments, words, cues,
                  allow_peak_start=False):
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
        if before and (t > peak or t == peak and not allow_peak_start) or not before and t <= peak:
            continue
        penalty = 0 if t in sentences else 0.25 if t in cues else 0.75 if t in word_edges else 1.5
        # Removing setup/payoff costs more than including a little additional context.
        loss = max(0, (anchor - t) * direction)
        options.append((t, penalty + abs(t - anchor) * 0.35 + loss))
    return options


def _styled_variant(cand_id, platform, start, end, pcfg):
    rng = random.Random(f"{cand_id}:{platform}")
    return Variant(platform, start, end, float(rng.choice(pcfg.get("zoom_choices", [1.0]))),
                   str(pcfg.get("highlight_color", "#FFE500")), round(rng.uniform(0.6, 1.6), 2))


def _protected_plans(cand_id, start, end, peak, duration, platforms, cfg, used, segments, words, cues,
                     limits, scene_start, end_ceiling, reserved, replacing):
    """Find a joint assignment of outward-only cuts. A tight platform may need the exact review bounds."""
    separation = float(cfg["variants"]["min_cut_distance_s"])
    reuse = float(cfg["variants"]["reuse_tolerance_s"])
    floor = min(start, scene_start) if scene_start is not None else 0.
    ceiling = max(end, end_ceiling) if end_ceiling is not None else duration
    available = {}
    for platform in platforms:
        pcfg = cfg["platforms"][platform]
        minimum = min(duration, cfg["stream_highlights"]["min_s"])
        maximum = min(cfg["stream_highlights"]["max_s"], pcfg.get("max_s", duration))
        if limits:
            minimum, maximum = max(minimum, limits[0]), min(maximum, limits[1])
        if end - start > maximum + 1e-8:
            raise ValueError(f"{platform}: reviewed interval {start:g}–{end:g} is {end-start:g}s; "
                             f"effective maximum is {maximum:g}s.")
        if min(ceiling-floor, maximum) < minimum - 1e-8:
            raise ValueError(f"{platform}: requires at least {minimum:g}s; only "
                             f"{min(ceiling-floor, maximum):g}s fit inside the scene and duration limits.")
        radius = max(pcfg.get("trim_jitter_s", 1.2), cfg["detection"]["snap_search_s"])

        def edges(before):
            anchor, other = (start, end) if before else (end, start)
            choices = _edge_options(anchor, other, peak, duration, minimum, maximum, radius,
                                    before, segments, words, cues, allow_peak_start=True)
            if before:
                return [(t, cost) for t, cost in choices if floor <= t <= start
                        and (scene_start is None or t == start or not any(a < t < b for a,b in segments))]
            return [(t, cost) for t, cost in choices if end <= t <= ceiling]

        starts, ends = edges(True), edges(False)
        fits = [(start - s + e - end, sc + ec, s, e) for s, sc in starts for e, ec in ends
                if minimum - 1e-8 <= e - s <= maximum + 1e-8]
        if not fits:
            raise ValueError(f"{platform}: no word-safe outward cut can contain {start:g}–{end:g}s "
                             f"inside {floor:g}–{ceiling:g}s with duration {minimum:g}–{maximum:g}s.")
        choices = [(s, e) for _, _, s, e in sorted(fits)
                   if not any(abs(s-a) < reuse and abs(e-b) < reuse for a, b in used.get(platform, ()))]
        if not choices:
            raise ValueError(f"{platform}: every feasible cut containing {start:g}–{end:g}s "
                             "reuses bounds reserved by another export.")
        available[platform] = choices

    order = sorted(platforms, key=lambda p: len(available[p]))
    chosen = {}

    def assign(remaining):
        if not remaining:
            return True
        # A previous sibling is still live until its replacement succeeds. Try moving
        # it first if it occupies the only feasible cut for a tighter platform.
        for platform in remaining if replacing else remaining[:1]:
            later = [p for p in remaining if p != platform]
            live = [*reserved, *chosen.values(), *(cut for p in later for cut in replacing.get(p, ()))]
            for s, e in available[platform]:
                if any(abs(s-a) < separation or abs(e-b) < separation for a, b in live):
                    continue
                chosen[platform] = (s, e)
                if assign(later):
                    return True
                del chosen[platform]
        return False

    if not assign(order):
        raise ValueError(f"{', '.join(platforms)}: distinct-cut spacing of {separation:g}s at both edges "
                         f"cannot preserve {start:g}–{end:g}s within {floor:g}–{ceiling:g}s "
                         "and the retained platform edits.")
    return [_styled_variant(cand_id, p, *chosen[p], cfg["platforms"][p]) for p in (chosen if replacing else platforms)]


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
    duration_limits: tuple[float, float] | None = None, protected: bool = False, end_ceiling: float | None = None,
    scene_start: float | None = None,
    reserved: tuple[tuple[float, float], ...] = (),
    replacing_cuts: dict[str, list[tuple[float, float]]] | None = None,
) -> list[Variant]:
    used_cuts = used_cuts or {}
    min_dist = float(cfg["variants"]["min_cut_distance_s"])
    reuse_tol = float(cfg["variants"]["reuse_tolerance_s"])
    out: list[Variant] = []
    segments = tr.segments(transcript)
    words = [(w["start"], w["end"]) for w in tr.words(transcript)]
    cues = cues or []
    if protected:
        return _protected_plans(cand_id, start, end, peak, duration, platforms, cfg, used_cuts, segments, words,
                                cues, duration_limits, scene_start, end_ceiling, reserved, replacing_cuts or {})
    for platform in platforms:
        pcfg = cfg["platforms"][platform]
        target = cfg["targets"][pcfg.get("target", "campaign")]
        min_s, max_s = float(target["min_s"]), float(pcfg.get("max_s", target["max_s"]))
        if duration_limits:
            min_s = max(min_s, duration_limits[0])
            max_s = min(max_s, duration_limits[1])
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
                if e < peak + post or not min_s <= e - s <= max_s:
                    continue
                if any(abs(e - v.end) < min_dist for v in out):
                    continue
                if any(abs(s - us) < reuse_tol and abs(e - ue) < reuse_tol for us, ue in used_cuts.get(platform, [])):
                    continue
                rank = (start_cost + end_cost, abs(s - start) + abs(e - end), s, e)
                if best is None or rank < best:
                    best, chosen = rank, (s, e)
        if chosen is None:
            raise RuntimeError(f"could not plan distinct cuts for candidate {cand_id} on {platform}")
        out.append(_styled_variant(cand_id, platform, *chosen, pcfg))
    return out
