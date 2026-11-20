"""Policy mechanics tested with invented fixture terms and synthetic media only."""
import datetime as dt
import json
from pathlib import Path

import pytest

from clipper import campaign, cli, db, media, pipeline

NOW = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)


@pytest.fixture()
def policy():
    return dict(terms_verified=True, terms_evidence="fixture brief v2", rights_evidence="fixture license",
                permission="verified", recordings=["abc"], platforms=["tiktok"], duration_s=[15, 34],
                hashtags=["#demo"], mentions=["@tester"], overlays=[], disclosure="PUBLICITÉ",
                ai_restrictions=[], deadline=None, manual_review=False)


@pytest.fixture()
def meta():
    return dict(source_sha256="abc", artifact_sha256="xyz", campaign="fixture", rights="authorized test",
                platform="tiktok", start=0, end=20, duration_s=20, caption="Demo #demo @tester", disclosure="PUBLICITÉ")


def test_unknown_terms_block_local_export(meta):
    report = campaign.check(meta, {}, technical=True)
    assert report["technically_valid"] and not report["rights_verified"] and not report["ready_for_submission"]


@pytest.mark.parametrize("key", sorted(campaign.FIELDS))
def test_every_policy_field_requires_explicit_declaration(meta, policy, key):
    del policy[key]
    assert not campaign.check(meta, policy, technical=True)["ready_for_submission"]


@pytest.mark.parametrize("field,value", [("platforms", ["youtube"]), ("duration_s", [21, 30]),
    ("hashtags", ["#dem"]), ("mentions", ["@test"]), ("deadline", "2019-01-01T10:00:00+00:00"),
    ("deadline", "2030-01-01"), ("permission", "pending"), ("recordings", ["other"]),
    ("disclosure", "other"), ("manual_review", "false"), ("ai_restrictions", ["Human cuts only"]),
    ("overlays", ["Sponsor mark"]), ("additional_condition", "unimplemented")])
def test_unmet_terms_block_readiness(meta, policy, field, value):
    policy[field] = value
    assert not campaign.check(meta, policy, technical=True, now=NOW)["ready_for_submission"]


def test_seven_independent_states_and_bound_evidence(meta, policy):
    policy.update(manual_review=True, overlays=["corner logo"], ai_restrictions=["no generated speech"])
    initial = campaign.check(meta, policy, technical=True)
    evidence = {k: initial[k] for k in ("export_key", "policy_sha256")}
    evidence.update(reviewer="operator", overlays=["corner logo"], ai_restrictions=["no generated speech"],
                    human_approval="QA sheet line 1")
    ok = campaign.check(meta, policy, technical=True, evidence=evidence)
    assert all(ok[k] for k in ("technically_valid", "rights_verified", "campaign_rules_satisfied",
                               "human_approved", "ready_for_submission"))
    assert not any(ok[k] for k in ("submitted", "approved_by_campaign", "paid"))
    evidence.update(submission="submission receipt", campaign_approval="approval email", payment="transfer receipt")
    tracked = campaign.check(meta, policy, technical=True, evidence=evidence)
    assert all(tracked[k] for k in ("submitted", "approved_by_campaign", "paid"))
    assert not campaign.check(dict(meta, artifact_sha256="replacement"), policy, technical=True,
                              evidence=evidence)["human_approved"]
    assert not campaign.check(meta, dict(policy, deadline="2030-01-01T12:00:00Z"), technical=True,
                              evidence=evidence)["human_approved"]


@pytest.mark.slow
def test_cli_rechecks_rendered_bytes_and_current_policy(source, cfg, fx, policy, monkeypatch, capsys):
    conn, sid = source
    conn.execute("UPDATE sources SET campaign='fixture' WHERE id=?", (sid,))
    conn.commit()
    policy.update(recordings=[media.sha256_file(fx["video"])], mentions=[], hashtags=[])
    cfg["campaigns"] = {"fixture": policy}
    monkeypatch.setattr(pipeline, "source_config", lambda *a: cfg)
    pipeline.analyze(conn, cfg, sid, transcribe=False)
    cid = db.list_candidates(conn, sid)[0]["id"]
    result, = pipeline.render_candidates(conn, cfg, sid, [cid], ["tiktok"])
    path = Path(result["path"])
    report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["validation"]
    assert report["ready_for_submission"]
    args = ["campaign-check", str(result["export_id"])]
    assert cli.main(args) == 0
    policy["terms_verified"] = False
    assert cli.main(args) != 0
    policy["terms_verified"] = True
    with path.open("ab") as fh:
        fh.write(b"altered")
    assert cli.main(args) != 0
    assert '"technically_valid": false' in capsys.readouterr().out
