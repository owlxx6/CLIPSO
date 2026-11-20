"""First-seconds scores use the actual edit, supported footage and a preserved event."""

import json
from pathlib import Path

import numpy as np
import pytest

from clipper import db, opening, variants
from test_stream_highlights import HOP, curves, proposals, utter


def visual_event(cfg, *, idle=True):
    settings = cfg["stream_highlights"]
    settings["setup_margin_s"] = 7
    settings["opening"]["search_s"] = 8
    f = curves()
    f["changed_fraction"] = np.zeros(200)
    f["audio_db"] = np.full(200, -80. if idle else -24.)
    f["gameplay_activity"][40] = f["novelty"][40] = 1
    f["quiet_hold_until"] = np.zeros(200)
    f["quiet_hold_until"][46] = 27  # a later quiet result that must survive every start choice
    return f


def test_visual_only_opening_trims_verified_idle_padding_without_losing_payoff(cfg):
    f = visual_event(cfg)
    c, = proposals(cfg, f)
    ctx = c.flags["opening_context"]
    assert ctx["original_start"] == 13 and 18 < c.start < 20
    assert c.end >= 27 and not c.flags["weak_opening"]
    evidence, = c.flags["opening_evidence"]
    assert evidence["kind"] == "visual activity"
    assert (evidence["source_time"], evidence["clip_offset"]) == (20, 20 - c.start)
    assert opening.evaluate(ctx, 13, c.end)["weak_opening"]
    assert not c.signals["audio_reaction"] and not c.signals["facecam_activity"]
    assert c.flags["opening_bonus"] > 0
    assert c.score == pytest.approx(c.flags["highlight_quality"] + c.flags["opening_bonus"], abs=.0001)


def test_reaction_first_uses_corroboration_without_creating_a_caption(cfg):
    f = curves()
    f["audio_reaction"][40:48] = f["laughter_cue"][40:48] = 1
    candidate, = proposals(cfg, f)
    assert not candidate.flags["weak_opening"] and not candidate.flags["hooks"]
    assert [e["kind"] for e in candidate.flags["opening_evidence"]] == ["corroborated reaction"]


@pytest.mark.parametrize("kind", ["matched result graphic", "held gameplay image"])
def test_quiet_curiosity_opening_is_measured_before_a_later_reaction(cfg, kind):
    f = curves()
    f["audio_reaction"][44:49] = f["laughter_cue"][44:49] = 1
    if kind == "matched result graphic":
        f["result_visible"] = np.zeros(200, bool)
        f["result_visible"][38:44] = True
    else:
        f["quiet_hold_until"] = np.zeros(200)
        f["quiet_hold_until"][38] = 22.
    c, = proposals(cfg, f)
    evidence = c.flags["opening_evidence"][0]
    assert evidence["kind"] == kind and evidence["clip_offset"] == 0
    assert c.flags["hook_strength_score"] == 1 and c.end >= 24


def test_boring_lead_in_ignores_source_intro_peak_and_caption(cfg):
    f = visual_event(cfg, idle=False)
    f["gameplay_activity"][2] = f["novelty"][2] = 1
    c = next(c for c in proposals(cfg, f) if c.peak > 15)
    assert c.start == 13 and c.flags["opening_window"] == [13, 16]
    assert c.flags["hook_strength_score"] == c.flags["opening_bonus"] == 0
    assert c.flags["weak_opening"] and not c.flags["opening_evidence"]
    assert opening.scored_flags(dict(c.flags, hooks=["You won't believe this!"]), c.start, c.end)["weak_opening"]


def test_review_score_serializes_grid_derived_shot_start(cfg):
    f = visual_event(cfg, idle=False)
    f["scene_change"][37] = 1  # the nearest shot boundary becomes the proposed start
    c, = proposals(cfg, f)
    assert c.start == 18.5
    assert json.loads(json.dumps(c.as_dict()))["flags"]["weak_opening"] is False


def test_misleading_start_cannot_skip_spoken_setup_to_show_later_action(cfg):
    f = visual_event(cfg)
    setup = utter(13, 18.5, "Je vous explique la règle avant de commencer.")
    c, = proposals(cfg, f, [setup])
    assert c.start == setup["start"] and c.flags["weak_opening"] and c.end >= 27
    for misleading in (14., 18.75, 20.):
        edited = opening.scored_flags(c.flags, misleading, c.end)
        assert edited["hook_strength_score"] == edited["opening_bonus"] == 0 and edited["opening_note"]


def test_short_spoken_setup_needs_complete_line_and_retained_support(cfg):
    f = visual_event(cfg)
    line = utter(13, 15, "Pourquoi cette voiture recule ?")
    c, = proposals(cfg, f, [line])
    assert c.start == 13
    cue, = c.flags["opening_evidence"]
    assert cue["kind"] == "spoken setup" and cue["text"] == line["text"]
    assert cue["until"] == 15 and 15 < cue["support"] <= c.end
    assert opening.scored_flags(c.flags, 13, 15.1)["weak_opening"]
    partial, = proposals(cfg, f, [utter(13, 16.3, line["text"])])
    assert partial.flags["weak_opening"]


@pytest.mark.parametrize("text", ["Oh mon dieu !", "What the hell just happened?", "Regarde jusqu'à la fin !"])
def test_generic_exclamations_and_clickbait_cannot_boost_loud_audio(cfg, text):
    f = curves()
    f["audio_reaction"][40] = f["speech"][40] = f["energy_rise"][40] = 1
    c, = proposals(cfg, f, [utter(17.5, 19.5, text)])
    assert c.flags["weak_opening"] and not c.flags["opening_bonus"]


def test_hook_cannot_admit_ineligible_event_or_rescue_known_incomplete_end(cfg):
    f = visual_event(cfg)
    f["quiet_hold_until"][46] = 70
    cfg["stream_highlights"]["setup_margin_s"] = 1
    c, = proposals(cfg, f)
    assert c.flags["hook_strength_score"] > 0 and c.flags["opening_context"]["incomplete"]
    assert c.flags["opening_bonus"] == 0 and c.score == c.flags["highlight_quality"]
    cfg["stream_highlights"]["min_score"] = 2
    assert not proposals(cfg, f)


def test_window_and_bonus_are_configurable_and_bounded(cfg):
    f = visual_event(cfg)
    settings = cfg["stream_highlights"]["opening"]
    c, = proposals(cfg, f)
    assert 0 < c.flags["opening_bonus"] <= .1 * c.flags["highlight_quality"] + .00005
    settings.update(window_s=1, bonus_weight=0)
    other, = proposals(cfg, f)
    assert other.flags["highlight_quality"] == c.flags["highlight_quality"]
    assert other.flags["opening_window"][1] - other.start == 1
    assert other.score == other.flags["highlight_quality"] and other.end == c.end
    settings["bonus_weight"] = 20
    with pytest.raises(ValueError, match="opening"):
        proposals(cfg, f)


def test_safe_manual_start_can_be_scored_beyond_automatic_search_radius(cfg):
    f = visual_event(cfg)
    cfg["stream_highlights"]["opening"]["search_s"] = 1
    c, = proposals(cfg, f)
    assert c.start == 13 and c.flags["weak_opening"]
    reviewed = opening.scored_flags(c.flags, 19, c.end)
    assert reviewed["hook_strength_score"] > 0 and reviewed["opening_window"] == [19, 22]


def test_complete_spoken_line_over_measured_action(cfg):
    f = curves()
    f["gameplay_activity"][40] = f["novelty"][40] = 1
    cfg["stream_highlights"]["setup_margin_s"] = 1.5
    c, = proposals(cfg, f, [utter(18.5, 21, "Pourquoi cette voiture recule ?")])
    assert any(e["kind"] == "complete spoken line" for e in c.flags["opening_evidence"])


def test_opening_does_not_trim_past_subtle_visual_setup_or_unmeasured_frames(cfg):
    for sample in (.10, np.nan):
        f = visual_event(cfg)
        f["changed_fraction"][29] = sample  # sample at 14.5 s ends the 14.0–14.5 s interval
        c, = proposals(cfg, f)
        assert c.start <= 14 and c.flags["weak_opening"]


def test_review_changes_rank_and_reanalysis_retains_manual_opening(source, cfg):
    conn, sid = source
    f = visual_event(cfg)
    f["gameplay_activity"][100] = f["novelty"][100] = 1
    f["quiet_hold_until"][106] = 57
    f["audio_db"][85:100] = -20
    candidates = proposals(cfg, f)
    assert len(candidates) == 2 and candidates[0].flags["highlight_quality"] == candidates[1].flags["highlight_quality"]
    # The weaker opening comes first chronologically so a score tie has a predictable order.
    f["audio_db"][:] = -80
    f["audio_db"][25:40] = -20
    rows = [c.as_dict() for c in proposals(cfg, f)]
    strong_id, weak_id = db.replace_candidates(conn, sid, rows, mode="stream_highlights")
    assert rows[0]["peak"] > rows[1]["peak"]
    db.save_stream_review(conn, strong_id, 42, rows[0]["end"], "Edited caption")
    for _ in range(2):
        ranked = db.list_candidates(conn, sid, "stream_highlights")
        assert [c["id"] for c in ranked] == [weak_id, strong_id]
        assert ranked[1]["flags"]["opening_window"] == [42, 45]
        assert ranked[1]["flags"]["weak_opening"] and ranked[1]["hook"] == "Edited caption"
        db.replace_candidates(conn, sid, rows, mode="stream_highlights")


def test_platform_extensions_stay_in_scene_and_at_sentence_boundaries(cfg):
    segs = {"segments": [utter(10, 15, "This sentence is essential.")]}
    plans = variants.plan_variants(7, 16, 25, 20, 80, ["tiktok", "instagram"], cfg,
                                   protected=True, scene_start=9.5, transcript=segs)
    assert all(9.5 <= p.start <= 16 and not 10 < p.start < 15 for p in plans)


def test_streamlit_shows_evidence_and_rescores_unsaved_opening(source, cfg):
    from clipper import cli

    ui = pytest.importorskip("streamlit.testing.v1")
    conn, sid = source
    c, = proposals(cfg, visual_event(cfg))
    cid, = db.replace_candidates(conn, sid, [c.as_dict()], mode="stream_highlights")
    app = ui.AppTest.from_file(str(Path(cli.__file__).with_name("review_app.py"))).run()
    app.sidebar.selectbox[1].select("stream_highlights").run()
    assert not app.exception
    assert any("Opening strength" in item.value and "Highlight quality" in item.value for item in app.markdown)
    assert any("visual activity" in item.value for item in app.caption)
    app.number_input(key=f"start_{cid}").set_value(13.).run()
    assert any("Weak opening" in item.value for item in app.warning)
    app.button(key=f"save_{cid}").click().run()
    assert not app.exception and db.get_candidate(conn, cid)["flags"]["weak_opening"]
