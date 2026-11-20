"""Local campaign declarations and export-specific evidence. No network or publication actions."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re


FIELDS = {"terms_verified", "rights_evidence", "terms_evidence", "permission", "recordings", "platforms",
          "duration_s", "hashtags", "mentions", "overlays", "disclosure", "ai_restrictions",
          "deadline", "manual_review"}


def policy_for(cfg: dict, identifier: str | None) -> dict:
    policies = cfg.get("campaigns")
    policy = policies.get(identifier) if identifier and isinstance(policies, dict) else None
    return policy if isinstance(policy, dict) else {}


def fingerprint(data: dict) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def duration_limits(policy: dict) -> tuple[float, float] | None:
    span = policy.get("duration_s")
    if (policy.get("terms_verified") is True and isinstance(span, list) and len(span) == 2
            and all(type(v) in (float, int) and math.isfinite(v) for v in span) and 0 <= span[0] <= span[1]
            and span[1] > 0):
        return tuple(span)
    return None


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def check(meta: dict, policy: dict, *, technical: bool, evidence: dict | None = None,
          now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    policy_hash = fingerprint(policy)
    identity = {k: meta.get(k) for k in ("campaign", "source_sha256", "artifact_sha256", "platform", "start",
                                          "end", "duration_s", "caption", "disclosure", "rights")}
    export_key = fingerprint(identity)
    evidence = evidence or {}
    if (evidence.get("export_key") != export_key or evidence.get("policy_sha256") != policy_hash
            or not _text(evidence.get("reviewer"))):
        evidence = {}
    blocks = []
    if not meta.get("campaign"):
        blocks.append("campaign identifier missing")
    if policy.get("terms_verified") is not True or not _text(policy.get("terms_evidence")):
        blocks.append("complete campaign terms must be verified with evidence")
    if set(policy) - FIELDS:
        blocks.append("policy contains unsupported requirements")
    recordings = policy.get("recordings")
    rights = bool(_text(meta.get("rights")) and policy.get("permission") == "verified"
                  and _text(policy.get("rights_evidence")) and isinstance(recordings, list)
                  and _text(meta.get("source_sha256")) and meta["source_sha256"] in recordings)
    platforms = policy.get("platforms")
    if not isinstance(platforms, list) or meta.get("platform") not in platforms:
        blocks.append("platform not authorized")
    limits = duration_limits(policy)
    duration = meta.get("duration_s")
    if not limits or type(duration) not in (int, float) or not limits[0] <= duration <= limits[1]:
        blocks.append("duration unverified or outside range")
    caption = meta.get("caption") or ""
    for field, prefix in (("hashtags", "#"), ("mentions", "@")):
        terms = policy.get(field)
        if not isinstance(terms, list):
            blocks.append(f"{field} unknown")
            continue
        tokens = {x.casefold() for x in re.findall(r"(?<![\w#@])[#@]\w+(?:\.\w+)*", caption)}
        for term in terms:
            if not _text(term) or not term.startswith(prefix) or term.casefold() not in tokens:
                blocks.append(f"missing or invalid {field} entry: {term}")
    disclosure = policy.get("disclosure")
    if "disclosure" not in policy or (disclosure is not None and
            (not _text(disclosure) or (meta.get("disclosure") or "").casefold() != disclosure.casefold())):
        blocks.append("disclosure unverified or mismatched")
    for field in ("overlays", "ai_restrictions"):
        terms = policy.get(field)
        confirmed = evidence.get(field, [])
        if (not isinstance(terms, list) or not isinstance(confirmed, list)
                or any(not _text(term) or term not in confirmed for term in terms)):
            blocks.append(f"{field} unknown or awaiting visual/workflow verification")
    if "deadline" not in policy:
        blocks.append("deadline unknown")
    elif policy["deadline"] is not None:
        try:
            cutoff = dt.datetime.fromisoformat(str(policy["deadline"]).replace("Z", "+00:00"))
            if cutoff.utcoffset() is None or now > cutoff:
                blocks.append("deadline lacks timezone or has passed")
        except ValueError:
            blocks.append("invalid deadline")
    if type(policy.get("manual_review")) is not bool:
        blocks.append("manual review requirement unknown")
    rules = not blocks
    human = _text(evidence.get("human_approval"))
    if not technical:
        blocks.append("encoded artifact lacks a matching technical validation")
    if not rights:
        blocks.append("permission for this recording lacks evidence")
    if policy.get("manual_review") is True and not human:
        blocks.append("human approval required")
    return {"version": 1, "checked_at": now.isoformat(), "export_key": export_key, "policy_sha256": policy_hash,
            "technically_valid": technical, "rights_verified": rights, "campaign_rules_satisfied": rules,
            "human_approved": human, "submitted": _text(evidence.get("submission")),
            "approved_by_campaign": _text(evidence.get("campaign_approval")), "paid": _text(evidence.get("payment")),
            "ready_for_submission": not blocks, "blockers": blocks}
