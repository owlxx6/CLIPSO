"""First-seconds evidence for an already eligible Stream Highlights candidate."""

from __future__ import annotations

import math
import re

import numpy as np

from clipper.signals.transcript import normalize


def validate(cfg):
    if not (1 <= cfg["window_s"] <= 3 and 0 <= cfg["search_s"] and 0 <= cfg["context_lead_s"]
            and 0 <= cfg["late_discount"] <= 1 and 0 < cfg["visible_s"] <= cfg["window_s"]
            and 0 < cfg["strong_min"] <= 1 and 0 <= cfg["bonus_weight"] <= .25 and cfg["silence_db"] < 0
            and all(math.isfinite(cfg[k]) for k in ("search_s", "context_lead_s", "silence_db"))):
        raise ValueError("Invalid stream_highlights.opening settings")


# A question or imperative must name something beyond a generic exhortation.
_FILLER = set("""the this that there these those what when where why who how look watch listen wait
you your they does did could would should can are was were have has just really happened happens happening
this that next about until till end guys here over again please believe unbelievable
que qui quoi quand pourquoi comment est ce c'est qu'est-ce c'etait qu'il qu'elle
vous votre elle elles nous ils les des dans par pour sur devant derriere avec fait faire passe
regarde regardez ecoute ecoutez attendez bien juste encore maintenant arrive arriver truc chose
hell earth damn fuck amazing crazy guys something really anyway happens happening
vraiment alors donc tout toute cette cela moi toi deja dieu mon incroyable jusque jusqu'a fin""".split())


def specific_line(text):
    normalized = normalize(text)
    if re.search(r"\b(?:click|subscribe|abonnez|abonne|watch (?:till|until)|jusqu[' ]a la fin)\b", normalized):
        return False
    tokens = normalized.split()
    return any(len(t) >= 4 and t not in _FILLER for t in tokens)


def measurements(features, hop, cfg, sync, response_lag):
    """Compute opening cues once per recording using the same decoded samples as the event detector."""
    g = features["gameplay_activity"]
    visual = np.minimum(g, features["novelty"])
    hard_cut = features.get("hard_cut")
    if hard_cut is not None:
        visual = np.where(hard_cut, 0, visual)  # a shot change alone is ambiguous
    audio = features["audio_reaction"]
    face = np.minimum(features["facecam_activity"], features["facecam_confidence"])
    reaction = np.minimum(audio, np.maximum(features["laughter_cue"], face))
    events = np.flatnonzero(sync >= cfg["evidence_floor"])
    responses = events + response_lag[events]
    verified = np.maximum(face, features.get("voiced_rise", np.zeros_like(face)))
    np.maximum.at(reaction, responses, np.minimum(sync[events], verified[responses]))

    idle = np.zeros(len(g), dtype=bool)
    if "changed_fraction" in features and "audio_db" in features:
        count = min(len(g), len(features["audio_db"]))
        still = features["changed_fraction"] <= cfg["quiet_still_fraction"]
        # A visual difference describes the interval ending at the decoded frame.
        idle[:count] = (still & np.r_[still[1:], False])[:count] & (
            features["audio_db"][:count] <= cfg["opening"]["silence_db"])
        idle &= (features["speech"] == 0) & (features["facecam_activity"] == 0) & (audio < cfg["reaction_release"])
        if "result_visible" in features:
            idle &= ~features["result_visible"]
        if "quiet_hold_until" in features:
            idle &= np.maximum.accumulate(features["quiet_hold_until"]) <= np.arange(len(g)) * hop
    return {"visual activity": visual, "corroborated reaction": reaction}, idle


def build_context(curves, idle, segments, lines, pictures, *, start, end, first, peak, scene_start, hop, cfg,
                  outcomes, incomplete, restore_floor=None):
    """Retain a small, candidate-local set of observations for review and platform-specific starts."""
    settings = cfg["opening"]
    # Restore a sentence that straddles the original cut when its complete beginning fits.
    earliest = min([start] + [s["start"] for s in segments if s["start"] < start < s["end"]])
    if earliest < (scene_start if restore_floor is None else restore_floor) or end - earliest > cfg["max_s"]:
        earliest = start
    protected_start = max(earliest, min(first - settings["context_lead_s"], peak))
    # Every discarded sample must have positive evidence of silence and a stable picture.
    for i in range(math.floor(earliest / hop), min(len(idle), math.ceil(protected_start / hop))):
        if not idle[i]:
            protected_start = max(earliest, i * hop)
            break
    for seg in segments:
        if seg["end"] > earliest and seg["start"] < protected_start:
            protected_start = max(earliest, seg["start"])
    latest = max(earliest, min(protected_start, start + settings["search_s"], end - cfg["min_s"]))
    window = settings["window_s"]
    horizon = min(end, protected_start + window)
    cues = []
    for kind, curve in curves.items():
        left, right = math.ceil(earliest / hop), math.ceil(horizon / hop)
        for offset in np.flatnonzero(curve[left:right] >= cfg["evidence_floor"]):
            i = left + int(offset)
            cues.append(dict(kind=kind, at=i * hop, strength=float(curve[i])))
    for i in range(math.ceil(max(scene_start, earliest - cfg["read_s"]) / hop), math.ceil(horizon / hop)):
        for kind, until in pictures.get(i, ()):
            if until > earliest:
                cues.append(dict(kind=kind, at=i * hop, until=float(until), strength=1.0))
    for seg, reply in lines:
        if not earliest <= seg["start"] < seg["end"] <= horizon:
            continue
        # A later answer or observed result grounds anticipation in the retained footage.
        support = next((fact for fact in (reply, *outcomes)
                        if fact is not None and seg["end"] <= fact[0] < fact[1] <= end), None)
        if support is not None:
            cues.append(dict(kind="spoken setup", at=seg["start"], until=seg["end"], support=support[1],
                             support_kind=support[2], text=seg["text"], strength=1.0))
        elif any(seg["start"] <= cue["at"] < seg["end"] for cue in cues if cue["kind"] in curves):
            cues.append(dict(kind="complete spoken line", at=seg["start"], until=seg["end"],
                             text=seg["text"], strength=1.0))
    speech = [[s["start"], s["end"]] for s in segments
              if s["end"] >= scene_start and s["start"] <= horizon and s["end"] >= earliest - window]
    incomplete = incomplete or any(s["start"] < end < s["end"] for s in segments)
    return dict(window_s=window, strong_min=settings["strong_min"], bonus_weight=settings["bonus_weight"],
                original_start=start, earliest=earliest, latest=latest, protected_start=protected_start,
                scene_start=scene_start, late_discount=settings["late_discount"], visible_s=settings["visible_s"],
                protected_end=end, incomplete=bool(incomplete), speech=speech, cues=cues)


def evaluate(context, start, end):
    """Score only cues observable inside this opening. Clip-relative offsets describe the actual edit."""
    start, end = float(start), float(end)  # detected shot times may be NumPy scalars
    window = context["window_s"]
    stop = min(end, start + window)
    observations = []
    reason = None
    if stop - start < 1:
        reason = "Less than one second is available to assess."
    elif start < context["scene_start"] or start > context["protected_start"]:
        reason = "The opening crosses a scene boundary or omits protected setup."
    elif any(a < start < b for a, b in context["speech"]):
        reason = "The opening interrupts a spoken line."
    if not reason:
        for cue in context["cues"]:
            at = cue["at"]
            kind = cue["kind"]
            if kind in ("matched result graphic", "held gameplay image"):
                at = max(start, at)
                if min(stop, cue["until"]) - at < context["visible_s"]:
                    continue
            elif kind in ("spoken setup", "complete spoken line") and (cue["until"] > stop or cue.get("support", end) > end):
                continue
            if not start <= at < stop:
                continue
            offset = at - start
            # Earlier evidence breaks ties. A single supported channel can earn the full score.
            value = round(float(cue["strength"] * (1 - context["late_discount"] * offset / window)), 4)
            observations.append(dict(cue, source_time=at, clip_offset=round(offset, 3), score=value))
    observations.sort(key=lambda e: -e["score"])
    # One strongest observation per cue type keeps the evidence useful in the review UI.
    best = {}
    for item in observations:
        best.setdefault(item["kind"], item)
    strength = observations[0]["score"] if observations else 0.0
    return dict(hook_strength_score=strength, weak_opening=strength < context["strong_min"],
                opening_window=[start, stop], opening_evidence=list(best.values()), opening_note=reason)


def select_start(context, end, hop):
    earliest, latest = context["earliest"], context["latest"]
    if context["incomplete"]:
        return earliest
    options = {earliest, latest}
    options.update(i * hop for i in range(math.ceil(earliest / hop), math.ceil(latest / hop)))
    # Equal scores retain more context. Only a supported opening can displace the conservative start.
    scored = [(evaluate(context, t, end)["hook_strength_score"], t) for t in sorted(options)]
    best, chosen = max(scored, key=lambda pair: pair[0])
    return chosen if best >= context["strong_min"] else earliest


def scored_flags(flags, start, end):
    """Recompute a proposed, reviewed or exported cut using the measurements from its analysis run."""
    context = flags["opening_context"]
    result = evaluate(context, start, end)
    if "channel_scores" in flags:
        result["channel_scores"] = dict(flags["channel_scores"], opening=result["hook_strength_score"])
    bonus = 0.0
    if context["incomplete"] or end < context["protected_end"]:
        result["opening_bonus_note"] = "The event or payoff boundary needs review."
    else:
        result["opening_bonus_note"] = None
    if not result["opening_bonus_note"] and not result["weak_opening"]:
        bonus = min(context["bonus_weight"] * result["hook_strength_score"], .10 * flags["highlight_quality"])
    bonus = round(bonus, 4)
    contributions = dict(flags["contributions"], opening_bonus=bonus)
    return dict(flags, **result, opening_bonus=bonus, contributions=contributions)
