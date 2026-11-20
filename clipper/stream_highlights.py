"""Event proposals with aligned observations and reviewable source-time boundaries."""

from __future__ import annotations

import math
import re

import numpy as np

from clipper.detect import Candidate
from clipper import opening
from clipper.signals.payoff import card_appearances
from clipper.signals import transcript as tr

MODE = "stream_highlights"
CLASSIC = "classic"


def _complete(text):
    if text.rstrip().endswith(("...", "…")):
        return False
    return bool(re.search(r"[.!?…][»”'\" )]*$", text.strip()))


# Restrict automatic quotes to recognizable question or imperative forms in the supported languages.
# Raw ASR often punctuates a chain of unrelated fragments as one sentence.
_QUESTION = re.compile(
    r"^(?:qu['’](?:est-ce|as|avez)|que |qui |quoi |où |pourquoi |comment |quand |est-ce |"
    r"tu (?:sais|veux|as|penses) |vous (?:voulez|avez|savez) |what |where |who |why |when |how |"
    r"(?:did|do|can|are) you )", re.I)
_IMPERATIVE = re.compile(r"^(?:regardez?|écoutez?|attendez|watch|look|listen)\b", re.I)


def _quotable(text):
    return (3 <= len(text.split()) <= 14 and _complete(text) and text[0].isupper()
            and not re.search(r"[.!?…]", text[:-1]) and len(set(text.casefold().split())) >= 3
            and (_IMPERATIVE.match(text) or (text.endswith("?") and _QUESTION.match(text))))


def _hooks(segments, start, end, peak):
    """Quote one intact utterance near the evidence peak or leave the hook empty."""
    suggestions = []
    for seg in sorted(segments, key=lambda s: abs((s["start"] + s["end"]) / 2 - peak)):
        text = " ".join(seg["text"].split())
        if not (start <= seg["start"] < seg["end"] <= end and seg["start"]-6 <= peak <= seg["end"]+6):
            continue
        if not _quotable(text):
            continue
        quote = f"« {text} »"
        if quote not in suggestions:
            suggestions.append(quote)
        if len(suggestions) == 3:
            break
    return suggestions


def _timing(segments, first, last, peak, duration, cfg, cuts, gameplay, hop, scene_start, ceiling):
    setup = max(scene_start, first - cfg["setup_margin_s"])
    end = min(ceiling, last + cfg["settle_s"])
    earliest = max(scene_start, first - cfg["max_setup_s"])
    nearby = [s for s in segments if s["end"] >= earliest and s["start"] < min(ceiling, setup + cfg["max_s"])]
    bridged = False

    def continuing_picture(left, right):
        between = gameplay[max(0, math.floor(left / hop)) : math.ceil(right / hop)]
        return (
            0 < right - left <= cfg["dialogue_bridge_s"]
            and len(between) > 0
            and between.max() >= cfg["evidence_floor"]
            and not any(left < cut < right for cut in cuts)
        )

    for seg in reversed(nearby):
        ordinary_gap = seg["end"] >= setup - cfg["speech_pause_s"]
        possible_setup = not _complete(seg["text"]) and continuing_picture(seg["end"], setup)
        if seg["start"] < setup and (ordinary_gap or possible_setup):
            if seg["start"] >= earliest or seg["end"] > setup:
                setup = max(scene_start, seg["start"])
                bridged |= not ordinary_gap
    # Prefer the nearest pre-event shot boundary in a speech-free gap.
    if not any(s["start"] <= first and s["end"] >= setup for s in nearby):
        setup = max((t for t in cuts if earliest <= t <= first), default=setup)
    limit = min(ceiling, setup + cfg["max_s"])
    if limit <= peak:  # a single long ASR segment must not displace the event
        setup = max(scene_start, first - cfg["setup_margin_s"])
        limit = min(ceiling, setup + cfg["max_s"])
    risk = end > limit
    end = min(end, limit)
    last_segment = next(
        (s for s in reversed(nearby) if last - cfg["settle_s"] - cfg["speech_pause_s"] <= s["end"] <= last), None
    )
    for seg in nearby:
        if seg["end"] < last or seg["end"] <= setup:
            continue
        if (last_segment and _complete(last_segment["text"]) and seg["start"] >= last
                and seg["start"] - last_segment["end"] >= cfg["topic_pause_s"]):
            ceiling = min(ceiling, seg["start"])
            limit, end = min(limit, ceiling), min(end, ceiling)
            break
        if seg["start"] > end + cfg["speech_pause_s"]:
            if (
                bridged
                or last_segment is None
                or _complete(last_segment["text"])
                or not continuing_picture(last_segment["end"], seg["start"])
            ):
                break
            bridged = True
        if last_segment and _complete(last_segment["text"]) and seg["start"] > last_segment["end"] + 0.35:
            if any(last_segment["end"] <= cut <= seg["start"] for cut in cuts):
                break
        if seg["end"] > limit:
            risk = True
            if end < seg["start"]:
                break
            end = limit
            break
        end = min(limit, max(end, seg["end"] + cfg["settle_s"]))
        last_segment = seg
    if last_segment and not _complete(last_segment["text"]):
        risk = True
    minimum = min(duration, cfg["min_s"])
    end = min(limit, max(end, setup + minimum))
    setup = max(scene_start, min(setup, end - minimum))
    # Prefer the gap or word edge on the safe side of the cap.
    for seg in nearby:
        if seg["start"] < end < seg["end"]:
            for word in tr.words({"segments": [seg]}):
                if word["start"] < end < word["end"]:
                    end = word["end"] if word["end"] <= limit else word["start"]
                    risk = True
                    break
        if seg["start"] < setup < seg["end"]:
            for word in tr.words({"segments": [seg]}):
                if word["start"] < setup < word["end"]:
                    setup = word["start"] if end - word["start"] <= cfg["max_s"] else word["end"]
                    risk = True
                    break
    risk = risk or bridged or first <= 0 or end >= duration - 0.01 or last > limit
    return max(scene_start, setup), min(ceiling, end), bool(risk), ceiling


def _response_pairs(features, scene_id, hop, cfg):
    """Extend the usual forward window only for a corroborated response in an uninterrupted shot."""
    g, a, f = (features[k] for k in ("gameplay_activity", "audio_reaction", "facecam_activity"))
    reaction = np.maximum(a, f)
    corroborated = np.maximum.reduce([
        features["laughter_cue"], np.minimum(f, features["facecam_confidence"]),
        np.minimum(features["speech"], features["energy_rise"]),
        features.get("voiced_rise", np.zeros_like(g)),
    ]) >= cfg["evidence_floor"]
    # A later game event gets first claim on a delayed response.
    recent = np.maximum.accumulate(np.where(g >= cfg["evidence_floor"], np.arange(len(g)), -1))
    sync = np.zeros_like(g)
    lags = np.zeros(len(g), dtype=int)
    longest = max(cfg["sync_s"], cfg["delayed_sync_s"])
    for lag in range(min(len(g), math.floor(longest / hop) + 1)):
        count = len(g) - lag
        allowed = scene_id[:count] == scene_id[lag:]
        if lag * hop > cfg["sync_s"]:
            allowed &= corroborated[lag:] & (recent[lag:] <= np.arange(count))
        joint = np.where(allowed, np.minimum(g[:count], reaction[lag:]), 0)
        better = joint > sync[:count]
        lags[:count][better] = lag
        np.maximum(sync[:count], joint, out=sync[:count])
    return sync, lags


def _laugh_endpoints(features, scene_id, hop, cfg):
    """Link timed markers through silence, never across speech, a competing event or a cut."""
    marked = np.flatnonzero(features["laughter_cue"])
    ends = np.zeros(len(scene_id), dtype=int)
    if not marked.size:
        return ends
    interruptions = np.cumsum((features["speech"] > 0) | (features["gameplay_activity"] >= cfg["evidence_floor"]))
    left, right = marked[:-1], marked[1:]
    split = ((right-left-1) * hop > cfg["repeat_laugh_gap_s"]) | (scene_id[right] != scene_id[left])
    split |= interruptions[right-1] != interruptions[left]
    for episode in np.split(marked, np.flatnonzero(split) + 1):
        ends[episode] = episode[-1]
    return ends


def _payoff_observation(cards, held, idx, last, stop, hop, cfg):
    """Look only near an existing event. Keep reading time bounded even for a permanent display."""
    upper = min(stop, math.floor((last + cfg["payoff_search_s"]) / hop) + 1)
    for at in range(idx, upper):
        if at in cards:
            return "matched result card", at * hop, min(cards[at], stop * hop)
    if held is not None:
        appearances = np.flatnonzero(held[idx:upper] > 0) + idx
        if appearances.size:
            later = appearances[appearances * hop >= last]
            at = int(later[0] if later.size else appearances[-1])
            return "held visual change", at * hop, min(float(held[at]), stop * hop)
    return "unconfirmed", None, None


def _event_parts(group, strength, max_ticks):
    """Split sustained activity at the largest gap, then the lowest local activity near the search limit."""
    while group.size and group[-1] - group[0] + 1 > max_ticks:
        choices = np.flatnonzero((group[1:] >= group[0] + max_ticks / 2) & (group[1:] <= group[0] + max_ticks))
        if not len(choices):
            cut = max(1, int(np.searchsorted(group, group[0] + max_ticks)))
        else:
            cut = 1 + max(choices, key=lambda i: (group[i + 1] - group[i], -strength[group[i]], i))
        yield group[:cut]
        group = group[cut:]
    if group.size:
        yield group


def _interpret(features, idx, part, response_idx, sync_strength, narrative, hop, cfg, evidence):
    """Describe the supporting measurements without treating them as object or emotion labels."""
    chaos_words = features.get("chaos_words")
    if sync_strength >= cfg["evidence_floor"]:
        response = response_idx
        evidence["synchronized_pair"] = {
            "game_time": round(idx * hop, 3),
            "response_time": round(response * hop, 3),
            "delay_s": round((response - idx) * hop, 3),
            "window": "delayed" if (response - idx) * hop > cfg["sync_s"] else "ordinary",
            "response_channel": "tracked_face"
            if features["facecam_activity"][response] > features["audio_reaction"][response]
            else "mixed_audio",
            "strength": round(float(sync_strength), 4),
        }
        kind, reason = "event + reaction", "Gameplay changed near a measured audio or tracked-face response."
        if chaos_words is not None:
            radius = math.ceil(cfg["sync_s"] / hop)
            left, right = max(0, idx - radius), min(len(chaos_words), idx + radius + 1)
            mentions = np.flatnonzero(chaos_words[left:right])
            if len(mentions):
                evidence["spoken_chaos_cue"] = {"time": round(float(left + mentions[0]) * hop, 3)}
                kind = "chaos cue"
                reason = "A transcribed event term coincides with a gameplay change and a possible reaction."
    elif features["laughter_cue"][part].max() > 0:
        kind, reason = "humour cue", "The transcript contains a laughter marker."
    elif narrative:
        kind, reason = "narrative", "A transcribed question is followed by a completed utterance."
    elif features["gameplay_activity"][idx] >= cfg["evidence_floor"]:
        kind, reason = "curiosity cue", "Gameplay activity rose above its local baseline."
    elif features.get("voiced_rise") is not None and features["voiced_rise"][idx] >= cfg["evidence_floor"]:
        kind, reason = "reaction cue", "Energy and the estimated pitch of periodic mixed audio changed together."
    else:
        kind, reason = "reaction cue", "A change in mixed audio coincides with speech or tracked face activity."
    return kind, reason


def propose(features, transcript, duration, hop, cfg, top_k):
    """Prefer game changes followed by reaction evidence. Scores are heuristic, not calibrated probabilities."""
    a, f, g = (features[k] for k in ("audio_reaction", "facecam_activity", "gameplay_activity"))
    opening.validate(cfg["opening"])
    reaction = np.maximum(a, f)
    n = len(g)
    cards = card_appearances(features.get("result_visible", np.zeros(n, bool)), hop, duration, cfg["hold_s"])
    barriers = np.array(features.get("hard_cut", np.zeros(n, bool)), dtype=bool, copy=True)
    for at in cards:
        barriers[at] = False  # a verified result card may itself arrive on a cut
    if cards:
        # A known graphic often replaces the entire screen. Its onset is boundary evidence only.
        g = g.copy()
        g[list(cards)] = 0
        features = dict(features, gameplay_activity=g)
    scene_id = np.cumsum(barriers)
    scene_edges = np.r_[0, np.flatnonzero(barriers), n]
    sync, response_lag = _response_pairs(features, scene_id, hop, cfg)
    laugh_ends = _laugh_endpoints(features, scene_id, hop, cfg)
    segments = (transcript or {}).get("segments", [])
    narrative = np.zeros(n, np.float32)
    replies = {}
    for previous, reply in zip(segments, segments[1:]):
        if (
            previous["text"].strip().endswith("?")
            and _complete(reply["text"])
            and 0 <= reply["start"] - previous["end"] <= cfg["speech_pause_s"]
        ):
            reply_i = min(n - 1, int(reply["start"] / hop))
            if scene_id[min(n - 1, int(previous["start"] / hop))] == scene_id[reply_i]:
                narrative[reply_i] = 1
                if opening.specific_line(reply["text"]) and not reply["text"].rstrip().endswith("?"):
                    replies[previous["start"]] = (reply["start"], reply["end"], "completed response to question")
    lines = []
    for i, seg in enumerate(segments):
        if i and seg["start"] - segments[i-1]["end"] <= cfg["speech_pause_s"] and not _complete(segments[i-1]["text"]):
            continue
        if _quotable(seg["text"].strip()) and opening.specific_line(seg["text"]):
            lines.append((seg, replies.get(seg["start"])))
    opening_curves, idle = opening.measurements(features, hop, cfg, sync, response_lag)
    pictures = {i: [("matched result graphic", until)] for i, until in cards.items()}
    held = features.get("quiet_hold_until")
    if held is not None:
        for i in np.flatnonzero(held):
            pictures.setdefault(int(i), []).append(("held gameplay image", held[i]))
    curves = dict(
        audio_reaction=a,
        facecam_activity=f,
        gameplay_activity=g,
        event_reaction_sync=sync,
        narrative_coherence=narrative,
        novelty=features["novelty"],
    )
    weights = cfg["weights"]
    score = sum(curve * weights[key] for key, curve in curves.items())
    # A standalone lexical laughter cue or a bounded question-answer pair can seed a proposal.
    seed = np.maximum.reduce(
        [
            sync,
            a * np.maximum(features["laughter_cue"], features["speech"]),
            g * features["novelty"],
            narrative,
            f * a,
            features.get("voiced_rise", np.zeros_like(a)),
        ]
    )
    eligible = (seed >= cfg["evidence_floor"]) & (score >= cfg["min_score"])
    cuts = np.flatnonzero(features["scene_change"] >= cfg["scene_change"]) * hop
    strength = np.maximum.reduce([g, reaction, narrative])
    active = np.flatnonzero(strength >= cfg["evidence_floor"])
    breaks = (np.diff(active) * hop > cfg["group_gap_s"]) | (np.diff(scene_id[active]) != 0)
    groups = np.split(active, np.flatnonzero(breaks) + 1)
    candidates = []
    for group in groups:
        if not len(group):
            continue
        # A long run of elevated activity may contain several separate moments.
        for part in _event_parts(group, strength, max(2, round(cfg["max_event_s"] / hop))):
            peaks = part[eligible[part]]
            if not len(peaks):
                continue
            idx = int(peaks[np.argmax(score[peaks])])
            peak = (idx * hop + min(duration, (idx + 1) * hop)) / 2
            first, last = float(part[0] * hop), min(duration, float((part[-1] + 1) * hop))
            response = idx + int(response_lag[idx])
            scene = scene_id[idx]
            lower, stop = int(scene_edges[scene]), int(scene_edges[scene+1])
            responses = part[reaction[part] >= cfg["evidence_floor"]]
            last_response = int(responses[-1]) if len(responses) else None
            if sync[idx] >= cfg["evidence_floor"]:
                last_response = max(response, last_response if last_response is not None else response)
            reaction_end = None
            if last_response is not None:
                tail_stop = min(stop, max(last_response+1, int(first / hop + cfg["max_event_s"] / hop)))
                tail = min(tail_stop - 1, max(last_response, int(laugh_ends[part].max()), int(laugh_ends[response])))
                while tail + 1 < tail_stop and reaction[tail+1] >= cfg["reaction_release"]:
                    tail += 1
                reaction_end = min(duration, (tail + 1) * hop)
                last = max(last, reaction_end)
            status, payoff_at, visible_until = _payoff_observation(
                cards, features.get("quiet_hold_until"), idx, last, stop, hop, cfg)
            ceiling = min(duration, stop * hop)
            if payoff_at is not None:
                read_until = min(visible_until, payoff_at + cfg["read_s"])
                if status == "matched result card" and visible_until <= read_until + cfg["settle_s"]:
                    read_until = visible_until  # finish a short card at its observed disappearance
                last = max(last, read_until)
                # After the observed result and reaction, leave the next conversation to its own clip.
                ceiling = min(ceiling, last + cfg["settle_s"])
                if status == "matched result card":
                    ceiling = min(ceiling, max(read_until, (reaction_end or 0) + cfg["settle_s"]))
            start, end, risk, ceiling = _timing(segments, first, last, peak, duration, cfg, cuts, g, hop,
                                                 lower * hop, ceiling)
            if visible_until is not None and visible_until > end:
                risk = True
            if part[-1] < group[-1]:
                risk = True  # activity continued into another proposed event
            if not start <= peak < end or end - start < min(cfg["min_s"], duration) - 0.01:
                continue
            scores = {key: round(float(curve[idx]), 4) for key, curve in curves.items()}
            scores["context_completeness"] = 0.0 if risk else 1.0
            # Keep editorial uncertainty separate from the existing structural score.
            review_ending = risk or status != "matched result card"
            contributions = {key: value * weights[key] for key, value in scores.items()}
            total = sum(contributions.values())
            outcomes = []
            if payoff_at is not None:
                outcomes.append((payoff_at, visible_until, status))
            if opening_curves["visual activity"][idx] >= cfg["evidence_floor"]:
                outcomes.append((idx * hop, min(duration, (idx + 1) * hop), "measured gameplay change"))
            if reaction_end is not None and opening_curves["corroborated reaction"][response] >= cfg["evidence_floor"]:
                outcomes.append((response * hop, reaction_end, "corroborated reaction"))
            context = opening.build_context(
                opening_curves, idle, segments, lines, pictures, start=start, end=end, first=first, peak=peak,
                scene_start=lower * hop, hop=hop, cfg=cfg, outcomes=outcomes, incomplete=risk)
            start = opening.select_start(context, end, hop)
            evidence = {}
            for key, curve in {
                **curves,
                "energy_rise": features["energy_rise"],
                "laughter_cue": features["laughter_cue"],
                "pitch_change": features.get("pitch_change", np.zeros_like(a)),
                "facecam_confidence": features["facecam_confidence"],
            }.items():
                at = int(part[np.argmax(curve[part])])
                if curve[at] > 0:
                    evidence[key] = {"time": round(at * hop, 3), "value": round(float(curve[at]), 4)}
            kind, reason = _interpret(features, idx, part, response, sync[idx], narrative[idx], hop, cfg, evidence)
            confidence = (
                "moderate" if not risk and (sync[idx] >= cfg["evidence_floor"] or narrative[idx]) else "low"
            )
            timing = dict(
                setup_start=round(start, 3),
                event_start=round(first, 3),
                peak_time=round(peak, 3),
                reaction_end=round(reaction_end, 3) if reaction_end is not None else None,
                payoff_start=round(payoff_at, 3) if payoff_at is not None else None,
                clip_end=round(end, 3),
            )
            flags = dict(
                mode=MODE,
                moment_type=kind,
                reason=reason,
                confidence=confidence,
                cut_risk=review_ending,
                payoff_status=status,
                camera_type=cfg["camera_type"],
                end_ceiling=ceiling,
                timing=timing,
                contributions=contributions,
                highlight_quality=round(total, 4),
                opening_context=context,
                evidence=evidence,
                hooks=_hooks(segments, start, end, peak),
                boundary_cues=[round(float(t), 3) for t in cuts if start - 3 <= t <= end + 3],
            )
            if payoff_at is not None:
                evidence["payoff"] = {"cue": status, "first_seen": payoff_at, "observed_until": visible_until}
            flags = opening.scored_flags(flags, start, end)
            candidates.append(
                Candidate(
                    start, end, peak, round(total + flags["opening_bonus"], 4), scores, flags,
                    tr.text_between(transcript, start, end)
                )
            )
    kept = []
    for c in sorted(candidates, key=lambda c: (-c.score, c.peak)):
        if all(
            max(0, min(c.end, k.end) - max(c.start, k.start)) / min(c.duration, k.duration) < 0.5 for k in kept
        ):
            kept.append(c)
            if len(kept) >= top_k:
                break
    return kept, score
