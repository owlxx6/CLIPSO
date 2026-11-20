"""Event proposals with aligned observations and reviewable source-time boundaries."""

from __future__ import annotations

import math
import re

import numpy as np

from clipper.detect import Candidate
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


def _hooks(segments, start, end, peak):
    """Quote one intact utterance near the evidence peak or leave the hook empty."""
    suggestions = []
    for seg in sorted(segments, key=lambda s: abs((s["start"] + s["end"]) / 2 - peak)):
        text = " ".join(seg["text"].split())
        if not (start <= seg["start"] < seg["end"] <= end and seg["start"]-6 <= peak <= seg["end"]+6):
            continue
        if not 3 <= len(text.split()) <= 14 or not _complete(text) or not text[0].isupper():
            continue
        if re.search(r"[.!?…]", text[:-1]) or len(set(text.casefold().split())) < 3:
            continue
        if not (_IMPERATIVE.match(text) or (text.endswith("?") and _QUESTION.match(text))):
            continue
        quote = f"« {text} »"
        if quote not in suggestions:
            suggestions.append(quote)
        if len(suggestions) == 3:
            break
    return suggestions


def _timing(segments, first, last, peak, duration, cfg, cuts, gameplay, hop):
    setup = max(0.0, first - cfg["setup_margin_s"])
    end = min(duration, last + cfg["settle_s"])
    earliest = max(0, first - cfg["max_setup_s"])
    nearby = [s for s in segments if s["end"] >= earliest and s["start"] <= setup + cfg["max_s"]]
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
                setup = seg["start"]
                bridged |= not ordinary_gap
    # Prefer the nearest pre-event shot boundary in a speech-free gap.
    if not any(s["start"] <= first and s["end"] >= setup for s in nearby):
        setup = max((t for t in cuts if earliest <= t <= first), default=setup)
    limit = min(duration, setup + cfg["max_s"])
    if limit <= peak:  # a single long ASR segment must not displace the event
        setup = max(0, first - cfg["setup_margin_s"])
        limit = min(duration, setup + cfg["max_s"])
    risk = end > limit
    end = min(end, limit)
    last_segment = next(
        (s for s in reversed(nearby) if last - cfg["settle_s"] - cfg["speech_pause_s"] <= s["end"] <= last), None
    )
    for seg in nearby:
        if seg["end"] < last or seg["end"] <= setup:
            continue
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
    setup = max(0, min(setup, end - minimum))
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
    risk = risk or bridged or first <= 0 or end >= duration - 0.01 or last >= limit - cfg["settle_s"]
    return setup, end, bool(risk)


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
    reaction = np.maximum(a, f)
    n = len(g)
    sync = np.zeros(n, dtype=np.float32)
    response_lag = np.zeros(n, dtype=int)
    for lag in range(min(n, round(cfg["sync_s"] / hop) + 1)):
        count = n - lag
        joint = np.minimum(g[:count], reaction[lag : lag + count])
        better = joint > sync[:count]
        response_lag[:count][better] = lag
        sync[:count] = np.maximum(sync[:count], joint)
    segments = (transcript or {}).get("segments", [])
    narrative = np.zeros(n, np.float32)
    for previous, reply in zip(segments, segments[1:]):
        if (
            previous["text"].strip().endswith("?")
            and _complete(reply["text"])
            and 0 <= reply["start"] - previous["end"] <= cfg["speech_pause_s"]
        ):
            narrative[min(n - 1, int(reply["start"] / hop))] = 1
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
    groups = np.split(active, np.flatnonzero(np.diff(active) * hop > cfg["group_gap_s"]) + 1)
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
            if sync[idx] >= cfg["evidence_floor"]:
                last = max(last, min(duration, (idx + int(response_lag[idx]) + 1) * hop))
            start, end, risk = _timing(segments, first, last, peak, duration, cfg, cuts, g, hop)
            if part[-1] < group[-1]:
                risk = True  # activity continued into another proposed event
            if not start <= peak < end or end - start < min(cfg["min_s"], duration) - 0.01:
                continue
            scores = {key: round(float(curve[idx]), 4) for key, curve in curves.items()}
            scores["context_completeness"] = 0.0 if risk else 1.0
            contributions = {key: value * weights[key] for key, value in scores.items()}
            total = sum(contributions.values())
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
            responses = part[reaction[part] >= cfg["evidence_floor"]]
            reaction_end = min(duration, float(responses[-1] + 1) * hop) if len(responses) else None
            if sync[idx] >= cfg["evidence_floor"]:
                reaction_end = max(reaction_end or 0, min(duration, (response + 1) * hop))
            timing = dict(
                setup_start=round(start, 3),
                event_start=round(first, 3),
                peak_time=round(peak, 3),
                reaction_end=round(reaction_end, 3) if reaction_end is not None else None,
                clip_end=round(end, 3),
            )
            flags = dict(
                mode=MODE,
                moment_type=kind,
                reason=reason,
                confidence=confidence,
                cut_risk=risk,
                timing=timing,
                contributions=contributions,
                evidence=evidence,
                hooks=_hooks(segments, start, end, peak),
                boundary_cues=[round(float(t), 3) for t in cuts if start - 3 <= t <= end + 3],
            )
            candidates.append(
                Candidate(
                    start, end, peak, round(total, 4), scores, flags, tr.text_between(transcript, start, end)
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
