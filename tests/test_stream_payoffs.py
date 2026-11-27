"""Payoff regressions use sampled synthetic pictures, local templates and known event times."""
import cv2
import numpy as np
import pytest

from clipper import cli, db, layout, media, pipeline, variants
from clipper.signals import payoff, stream
from test_stream_highlights import HOP, curves, proposals, utter


def event(f):
    f["gameplay_activity"][20] = f["novelty"][20] = f["audio_reaction"][21] = 1


def frames_and_card():
    frames = [np.full((96, 160), 32, np.uint8) for _ in range(80)]
    card = frames[0][24:48, 40:120].copy()
    cv2.putText(card, "FAIL", (4, 19), cv2.FONT_HERSHEY_SIMPLEX, .7, 240, 2)
    return frames, card


def scan(cfg, frames, monkeypatch, box=None, reference=None):
    src = dict(path="synthetic", height=96, width=160)
    monkeypatch.setattr(media, "gray_frames", lambda *args: iter(zip(np.arange(len(frames))*HOP, frames)))
    return stream.visual_features(src, box, len(frames)*HOP, HOP, cfg["stream_highlights"], reference)


def test_quiet_visual_result_survives_drop_in_sound_and_motion(cfg, monkeypatch):
    frames, _ = frames_and_card()
    for frame in frames[28:]:
        frame[12:60, 25:75] = 155
    f = dict(curves(40), **scan(cfg, frames, monkeypatch))
    event(f)
    c = proposals(cfg, f)[0]
    assert c.flags["evidence"]["payoff"]["first_seen"] == 14
    assert c.flags["payoff_status"] == "held visual change" and c.flags["cut_risk"]
    assert 18 <= c.end <= 19 and c.flags["timing"]["reaction_end"] == 11


def test_static_text_card_after_event_is_held_until_disappearance(cfg, monkeypatch, tmp_path):
    frames, card = frames_and_card()
    path = media.write_image(tmp_path / "résultat.png", card)
    cfg["stream_highlights"]["result_card"].update(template=str(path), roi=[.25, .25, .5, .25])
    for frame in frames[28:36]:
        frame[24:48, 40:120] = card
    f = dict(curves(40), **scan(cfg, frames, monkeypatch, reference=payoff.result_reference(cfg["stream_highlights"])))
    assert not proposals(cfg, f)
    event(f)
    c = proposals(cfg, f, [utter(18.5, 28, "Une autre conversation commence.")])[0]
    assert c.flags["payoff_status"] == "matched result card"
    assert c.flags["evidence"]["payoff"] == dict(cue="matched result card", first_seen=14, observed_until=18)
    assert c.end == 18 and c.flags["end_ceiling"] == 18
    assert not c.flags["hooks"] and not c.flags["cut_risk"]


@pytest.mark.parametrize("where", ["permanent", "flash", "different"])
def test_hud_and_one_frame_messages_are_ignored(cfg, monkeypatch, where):
    frames, card = frames_and_card()
    cfg["stream_highlights"]["result_card"]["roi"] = [.25, .25, .5, .25]
    selection = {"permanent": frames, "flash": frames[28:29], "different": frames[28:36]}
    for frame in selection[where]:
        frame[24:48, 40:120] = np.fliplr(card) if where == "different" else card
    measured = scan(cfg, frames, monkeypatch, reference=card)
    assert not payoff.card_appearances(measured["result_visible"], HOP, 40, 1.5)
    assert not proposals(cfg, dict(curves(40), **measured))


def test_result_may_enter_on_a_cut_but_respawn_stops_clip(cfg):
    f = curves()
    event(f)
    f["result_visible"] = np.zeros(200, bool)
    f["result_visible"][26:36] = True
    f["hard_cut"] = np.zeros(200, bool)
    f["hard_cut"][[26, 36]] = True
    f["gameplay_activity"][[26, 36]] = f["novelty"][[26, 36]] = 1
    f["audio_reaction"][37] = 1
    clips = proposals(cfg, f)
    before = next(c for c in clips if c.peak < 13)
    assert before.end == 18 and before.flags["timing"]["payoff_start"] == 13
    assert all(c.start >= 18 for c in clips if c.peak >= 18)


def test_message_can_disappear_before_delayed_laugh_in_same_scene(cfg):
    f = curves()
    f["gameplay_activity"][20] = f["novelty"][20] = f["audio_reaction"][34] = f["laughter_cue"][34] = 1
    f["result_visible"] = np.zeros(200, bool)
    f["result_visible"][24:30] = True
    c = proposals(cfg, f)[0]
    assert c.flags["evidence"]["payoff"]["observed_until"] == 15
    assert c.flags["timing"]["reaction_end"] == 17.5 and c.end >= 17.5


def test_long_result_card_has_bounded_reading_time_and_review_flag(cfg):
    f = curves()
    event(f)
    f["result_visible"] = np.zeros(200, bool)
    f["result_visible"][26:] = True
    c = proposals(cfg, f)[0]
    assert c.end == 17 and c.flags["cut_risk"]
    assert c.flags["evidence"]["payoff"]["observed_until"] == 100


@pytest.mark.parametrize("condition,expected", [
    ("same_shot", True), ("uncorroborated", False), ("cut", False), ("other_event", False), ("short_window", False)
])
def test_six_second_response_requires_context(cfg, condition, expected):
    f = curves()
    f["gameplay_activity"][20] = f["novelty"][20] = f["audio_reaction"][32] = 1
    f["laughter_cue"][32] = condition != "uncorroborated"
    if condition == "cut":
        f["hard_cut"] = np.zeros(200, bool)
        f["hard_cut"][27] = True
    if condition == "other_event":
        f["gameplay_activity"][27] = 1
    if condition == "short_window":
        cfg["stream_highlights"]["delayed_sync_s"] = 4
    paired = [c for c in proposals(cfg, f) if c.flags["evidence"].get("synchronized_pair", {}).get("game_time") == 10]
    assert bool(paired) == expected
    if paired:
        pair = paired[0].flags["evidence"]["synchronized_pair"]
        assert (pair["response_time"], pair["delay_s"], pair["window"]) == (16, 6, "delayed")
        assert paired[0].flags["timing"]["reaction_end"] >= 16.5


@pytest.mark.parametrize("interruption,linked", [("silence", True), ("speech", False), ("cut", False)])
def test_laughter_plateau_then_second_laugh(cfg, interruption, linked):
    segs = [dict(utter(s, e, "Haha"), words=[dict(start=s, end=e, word="Haha")]) for s,e in [(11,14),(18,20)]]
    if interruption == "speech":
        segs.insert(1, utter(15, 16, "Changeons de sujet."))
    f = curves()
    f.update(stream.audio_features(np.full(200, -40), {"segments": segs}, 100, HOP,
                                   cfg["stream_highlights"], cfg["signals"]["keywords"]))
    assert f["laughter_cue"][22:28].all() and not f["energy_rise"].any()
    f["gameplay_activity"][20] = f["novelty"][20] = 1
    if interruption == "cut":
        f["hard_cut"] = np.zeros(200, bool)
        f["hard_cut"][32] = True
    c = next(c for c in proposals(cfg, f, segs) if c.peak < 15)
    assert (c.flags["timing"]["reaction_end"] >= 20) == linked
    if linked:
        assert c.end >= 20


def test_supported_reaction_keeps_weaker_continuous_tail(cfg):
    f = curves()
    event(f)
    f["facecam_activity"][22:34] = .4
    c = proposals(cfg, f)[0]
    assert c.flags["timing"]["reaction_end"] == 17 and c.end >= 17


@pytest.mark.parametrize("camera", ["unknown", "gameplay_only", "full_body", "screen_reaction", "avatar"])
def test_camera_profiles_do_not_run_webcam_motion(cfg, monkeypatch, camera):
    frames, _ = frames_and_card()
    for i, frame in enumerate(frames):
        frame[:48, :80] = 200 if i % 2 else 32
    monkeypatch.setattr(layout, "FaceFinder", lambda: pytest.fail("unconfirmed human face tracker"))
    cfg["stream_highlights"]["camera_type"] = camera
    measured = scan(cfg, frames, monkeypatch, dict(x=0, y=0, w=.5, h=.5))
    assert not measured["facecam_activity"].any()
    assert bool(measured["gameplay_activity"].any()) == (camera == "gameplay_only")
    if camera in {"screen_reaction", "full_body", "avatar"}:
        assert not scan(cfg, frames, monkeypatch)["gameplay_activity"].any()


def test_source_camera_override_preserves_layout_and_changes_cached_analysis(source, cfg):
    conn, sid = source
    wd = pipeline.workdir(cfg, sid)
    before = layout.load_layout(wd)
    assert cli.main(["layout", str(sid), "--camera-type", "avatar", "--set", "bad"]) != 0
    assert layout.load_layout(wd) == before
    for kind in ("webcam", "avatar"):
        assert cli.main(["layout", str(sid), "--camera-type", kind]) == 0
        assert layout.load_layout(wd) == dict(before, camera_type=kind)
        pipeline.analyze(conn, cfg, sid, transcribe=False, mode="stream_highlights")
        with np.load(wd/"stream_activity.npz") as cache:
            assert kind in str(cache["_key"])
        assert all(c["flags"]["camera_type"] == kind for c in db.list_candidates(conn, sid, "stream_highlights"))


def test_one_frame_flash_is_not_a_scene_boundary(cfg, monkeypatch):
    frames, _ = frames_and_card()
    frames[20][:] = 250
    for frame in frames[40:]:
        frame[:] = 180
    f = scan(cfg, frames, monkeypatch)
    assert np.flatnonzero(f["hard_cut"]).tolist() == [40]
    assert f["quiet_hold_until"][21] == 0


def test_completed_speech_and_scene_cut_stop_unrelated_ending(cfg):
    f = curves()
    event(f)
    segs = [utter(8, 11, "Voilà le résultat !"), utter(12.4, 25, "Une autre conversation commence.")]
    c = proposals(cfg, f, segs)[0]
    assert c.end <= 12.4 and c.flags["end_ceiling"] == 12.4
    f["hard_cut"] = np.zeros(200, bool)
    f["hard_cut"][23] = True
    c = proposals(cfg, f, segs)[0]
    assert c.end <= 11.5 and c.flags["end_ceiling"] == 11.5


def test_platforms_respect_boundary_and_manual_override(cfg):
    plan, = variants.plan_variants(7, 8, 20, 12, 100, ["tiktok"], cfg, protected=True, end_ceiling=20)
    assert plan.end == 20
    revised, = variants.plan_variants(7, 8, 21, 12, 100, ["tiktok"], cfg, protected=True, end_ceiling=20)
    assert revised.end == 21
    with pytest.raises(ValueError, match="distinct-cut"):
        variants.plan_variants(7, 8, 20, 12, 100, ["tiktok","instagram"], cfg, protected=True, end_ceiling=20)


def test_result_reference_rejects_missing_flat_or_invalid_crop(cfg, tmp_path):
    settings = cfg["stream_highlights"]
    spec = settings["result_card"]
    spec.update(template="absent.png", roi=[0, 0, 1, 1])
    with pytest.raises(ValueError, match="Cannot read"):
        payoff.result_reference(settings)
    spec["template"] = str(media.write_image(tmp_path/"flat.png", np.zeros((20,40), np.uint8)))
    with pytest.raises(ValueError, match="contrasting"):
        payoff.result_reference(settings)
    spec["roi"] = [.8, 0, .5, 1]
    with pytest.raises(ValueError, match="ROI"):
        payoff.result_reference(settings)


def test_sparse_scan_does_not_infer_a_four_second_card_from_one_sample():
    visible = np.zeros(48, bool)
    visible[16] = True
    sampled = [0, 8, 16, 24, 32, 40]
    assert not payoff.confirmed_card_samples(visible, sampled, HOP, 1.5).any()
    visible[24] = True
    assert payoff.confirmed_card_samples(visible, sampled, HOP, 1.5)[16:32].all()
