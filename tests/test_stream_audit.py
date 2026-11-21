"""Regression probes for the independent review of 791188c."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clipper import db, pipeline, variants
from clipper.signals import stream, transcript as tr
from test_stream_highlights import HOP, curves, proposals, utter
from test_stream_payoffs import frames_and_card, scan


def impact(at=50):
    f = curves(140)
    i = int(at / HOP)
    f["gameplay_activity"][i:i+5] = .8
    f["gameplay_activity"][i] = f["novelty"][i] = f["audio_reaction"][i+1] = 1
    return f


@pytest.mark.parametrize("cut", [50, 30])
def test_histogram_jump_does_not_remove_or_inflate_setup(cfg, cut):
    f = impact()
    f["scene_change"][int(cut/HOP)] = 1
    c, = proposals(cfg, f)
    assert 45 <= c.start <= 49, (c.start, c.end)


@pytest.mark.parametrize("segments", [
    [utter(t, t+3.8, "Ongoing commentary.") for t in range(0, 136, 4)],
    [utter(3, 125, "This is one very long transcribed segment with no useful break")],
])
def test_short_event_in_continuous_speech_has_local_boundaries(cfg, segments):
    c, = proposals(cfg, impact(), segments)
    assert 36 <= c.start < 50 and 53 <= c.end <= 68, (c.start, c.end)


def test_visual_flash_at_impact_keeps_the_approach(cfg, monkeypatch):
    frames, _ = frames_and_card()
    frames[20][:] = 250
    f = dict(curves(40), **scan(cfg, frames, monkeypatch))
    f["audio_reaction"][21] = 1
    c, = proposals(cfg, f)
    assert f["scene_change"][20] >= cfg["stream_highlights"]["scene_change"]
    assert not f["hard_cut"][20] and 6 <= c.start <= 9.25
    assert c.flags["evidence"]["synchronized_pair"]


def test_narrative_with_long_answer_keeps_the_complete_sequence(cfg):
    segs = [utter(30, 40, "Pourquoi cette voiture recule ?"),
            utter(40.3, 76, "The driver has been trying to find a way out.")]
    c, = proposals(cfg, curves(140), segs)
    assert c.start <= 30 and c.end >= 76


def test_uncertain_payoff_is_not_structural_cut_risk(cfg):
    c, = proposals(cfg, impact())
    assert c.flags["payoff_status"] == "unconfirmed"
    assert not c.flags["cut_risk"]
    assert c.flags["channel_scores"]["gameplay"] == 1
    assert c.flags["channel_scores"]["streamer_reaction"] == 1
    assert c.flags["channel_scores"]["narrative"] == 0
    assert c.flags["channel_scores"]["opening"] == c.flags["hook_strength_score"]


@pytest.mark.parametrize("spelling", ["Ahahahaha!", "Hahahaaaa", "MDRRRR", "PTDRrr", "Ha-ha-ha"])
def test_delayed_whisper_laughter_is_linked_to_game_event(cfg, spelling):
    f = impact()
    f["audio_reaction"][:] = 0
    seg = utter(56, 57, spelling)
    seg["words"] = [dict(seg, word=spelling)]
    audio = stream.audio_features(np.full(280, -40.), {"segments": [seg]}, 140, HOP,
                                  cfg["stream_highlights"], cfg["signals"]["keywords"])
    f.update(audio)
    f["gameplay_activity"][101:] = 0
    c = proposals(cfg, f, [seg])[0]
    assert audio["laughter_cue"][112] > 0
    assert c.flags["evidence"]["synchronized_pair"]["response_time"] == 56
    assert c.end >= 57


def test_laughter_requires_complete_tokens(cfg):
    seg = utter(5, 12, "Mahaha Shahab madras haha123 ptdragon mdr_test")
    audio = stream.audio_features([], {"segments": [seg]}, 20, HOP, cfg["stream_highlights"], cfg["signals"]["keywords"])
    assert not audio["laughter_cue"].any()


def test_generated_asr_retains_optional_confidence(monkeypatch):
    word = SimpleNamespace(start=1, end=2, word="laughs", probability=.72)
    segment = SimpleNamespace(start=1, end=2, text="laughs", words=[word], no_speech_prob=.12, avg_logprob=-.35)
    model = SimpleNamespace(transcribe=lambda *a, **k: ([segment], SimpleNamespace(language="en", language_probability=.95)))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=lambda *a, **k: model))
    data = tr.transcribe("offline", {}, log=lambda _: None)
    assert data["language_probability"] == .95
    cleaned = stream.usable_transcript(data, 10)["segments"][0]
    assert (cleaned["no_speech_prob"], cleaned["avg_logprob"], cleaned["words"][0]["probability"]) == (.12, -.35, .72)


def test_missing_transcript_and_active_channels_are_reported(source, cfg, monkeypatch):
    conn, sid = source
    monkeypatch.setattr(pipeline, "_stream_transcript", lambda *a: None)
    messages = []
    rows = pipeline.analyze(conn, cfg, sid, mode="stream_highlights", transcribe=False, log=messages.append)
    assert any("no usable transcript" in m.lower() for m in messages)
    assert any("active channels" in m.lower() for m in messages)
    assert rows and all(c["flags"]["active_channels"]["transcript"] is False for c in rows)


def test_review_shows_channels_even_if_stream_detector_abstains(source, cfg, monkeypatch):
    ui = pytest.importorskip("streamlit.testing.v1")
    conn, sid = source
    monkeypatch.setattr(pipeline, "_stream_transcript", lambda *a: None)
    cfg["stream_highlights"]["min_score"] = 100
    assert not pipeline.analyze(conn, cfg, sid, mode="stream_highlights", transcribe=False)
    app = ui.AppTest.from_file(str(Path(pipeline.__file__).with_name("review_app.py"))).run()
    app.sidebar.selectbox[1].select("stream_highlights").run()
    assert not app.exception
    assert any("No usable transcript" in item.value for item in app.sidebar.warning)
    assert any("Active channels: gameplay, mixed audio" in item.value for item in app.sidebar.caption)


def test_protected_platforms_can_share_a_feasible_assignment(cfg):
    cfg["platforms"]["instagram"]["max_s"] = 12
    plans = variants.plan_variants(1, 10, 22, 17, 100, ["tiktok", "instagram"], cfg, protected=True)
    assert [p.platform for p in plans] == ["tiktok", "instagram"]
    assert (plans[1].start, plans[1].end) == (10, 22)
    assert plans[0].start < 10 and plans[0].end > 22


def test_platform_error_identifies_duration_and_reviewed_interval(cfg):
    cfg["platforms"]["instagram"]["max_s"] = 11
    with pytest.raises(ValueError, match=r"instagram.*12(?:\.0+)?\s*s.*11(?:\.0+)?\s*s"):
        variants.plan_variants(1, 10, 22, 17, 100, ["instagram"], cfg, protected=True)


def test_joint_replacement_moves_sibling_before_reusing_its_old_bounds(cfg):
    cfg["platforms"]["instagram"]["max_s"] = 12
    plans = variants.plan_variants(1, 10, 22, 17, 100, ["instagram", "tiktok"], cfg, protected=True,
                                   replacing_cuts={"tiktok": [(10, 22)]})
    assert [p.platform for p in plans] == ["tiktok", "instagram"]
    assert plans[0].start < 10 and plans[0].end > 22
    assert (plans[1].start, plans[1].end) == (10, 22)


def test_protected_start_may_equal_peak_at_recording_edge(cfg):
    plan, = variants.plan_variants(1, 0, 12, 0, 100, ["tiktok"], cfg, protected=True)
    assert (plan.start, plan.end) == (0, 12)


def test_word_boundary_conflict_is_explained(cfg):
    cfg["platforms"]["tiktok"]["max_s"] = 12
    transcript = {"segments": [dict(start=9, end=11, text="word", words=[dict(start=9, end=11, word="word")])]}
    with pytest.raises(ValueError, match="tiktok.*word-safe"):
        variants.plan_variants(1, 10, 22, 17, 100, ["tiktok"], cfg, protected=True, transcript=transcript)


@pytest.mark.slow
def test_review_replace_keeps_exact_bounds_and_edited_hook(source, cfg):
    conn, sid = source
    row = dict(start=10.5, end=22.5, peak=17, score=2, flags={"mode": "stream_highlights", "end_ceiling": 22.5})
    cid, = db.replace_candidates(conn, sid, [row], mode="stream_highlights")
    cfg["platforms"]["tiktok"]["max_s"] = 12
    cfg["render"].update(width=180, height=320, facecam_band_h=80)
    for start, end, hook, replace in [(10.5, 22.5, "", False), (10.5, 22.5, "", True),
                                       (10.5, 22.5, "Reviewed hook", True), (11.7, 23.7, "Edited bounds", True)]:
        db.save_stream_review(conn, cid, start, end, hook)
        out, = pipeline.render_candidates(conn, cfg, sid, [cid], ["tiktok"], replace=replace)
        assert (out["start"], out["end"], out["hook"]) == (start, end, hook or None)
        assert Path(out["path"]).is_file() and len(db.list_exports(conn, candidate_id=cid)) == 1


@pytest.mark.slow
def test_multi_platform_render_preserves_review_and_posted_sibling(source, cfg):
    conn, sid = source
    row = dict(start=10.5, end=22.5, peak=17, score=2, flags={"mode": "stream_highlights"})
    cid, = db.replace_candidates(conn, sid, [row], mode="stream_highlights")
    cfg["platforms"]["instagram"]["max_s"] = 12
    cfg["render"].update(width=180, height=320, facecam_band_h=80)
    platforms = ["tiktok", "instagram"]
    first = pipeline.render_candidates(conn, cfg, sid, [cid], platforms)
    cuts = {r["platform"]: (r["start"], r["end"]) for r in first}
    assert cuts["instagram"] == (10.5, 22.5)
    assert cuts["tiktok"][0] < 10.5 and cuts["tiktok"][1] > 22.5
    repeated = pipeline.render_candidates(conn, cfg, sid, [cid], platforms, replace=True)
    assert {r["platform"]: (r["start"], r["end"]) for r in repeated} == cuts
    posted = next(r for r in repeated if r["platform"] == "instagram")
    db.add_post(conn, posted["export_id"], "https://example.test/posted")
    db.save_stream_review(conn, cid, 11.7, 23.7, "Revised")
    edited = pipeline.render_candidates(conn, cfg, sid, [cid], platforms, replace=True)
    tiktok = next(r for r in edited if r["platform"] == "tiktok")
    assert tiktok["start"] <= 11.7 and tiktok["end"] >= 23.7 and tiktok["hook"] == "Revised"
    assert "posted" in next(r["blocked_reason"] for r in edited if r["platform"] == "instagram")
    assert Path(posted["path"]).is_file() and db.post_for(conn, posted["export_id"])
