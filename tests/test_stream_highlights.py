from pathlib import Path

import numpy as np
import pytest

from clipper import cli, db, layout, media, pipeline, stream_highlights as hunter, variants
from clipper.signals import stream

HOP = 0.5


def curves(duration=100):
    return {
        name: np.zeros(round(duration / HOP), dtype=np.float32)
        for name in (
            "audio_reaction",
            "facecam_activity",
            "gameplay_activity",
            "novelty",
            "facecam_confidence",
            "energy_rise",
            "speech",
            "laughter_cue",
            "scene_change",
        )
    }


def proposals(cfg, features, segments=()):
    return hunter.propose(
        features, {"segments": list(segments)}, len(features["speech"]) * HOP, HOP, cfg["stream_highlights"], 10
    )[0]


def utter(start, end, text):
    return dict(start=start, end=end, text=text)


def test_audio_alone_abstains_and_sustained_loudness_is_not_a_reaction(cfg):
    levels = np.full(200, -4.0)
    a = stream.audio_features(levels, None, 100, HOP, cfg["stream_highlights"], cfg["signals"]["keywords"])
    assert not a["audio_reaction"].any()
    levels[70:80] = -40
    a = stream.audio_features(levels, None, 100, HOP, cfg["stream_highlights"], cfg["signals"]["keywords"])
    assert 0 < max(a["audio_reaction"]) < 0.2
    assert proposals(cfg, dict(curves(), **a)) == []


def test_synchronized_event_beats_laughter_and_narration(cfg):
    f = curves()
    f["gameplay_activity"][40] = f["novelty"][40] = 1
    f["facecam_activity"][43] = 0.95
    f["audio_reaction"][120] = f["laughter_cue"][120] = f["speech"][120] = 1
    segs = [utter(81, 84, "Tu as vu ça ?"), utter(84.4, 87, "Oui, regarde devant toi !")]
    ranked = proposals(cfg, f, segs)
    assert [r.flags["moment_type"] for r in ranked] == ["event + reaction", "humour cue", "narrative"]
    pair = ranked[0].flags["evidence"]["synchronized_pair"]
    assert (pair["game_time"], pair["response_time"], pair["response_channel"]) == (20, 21.5, "tracked_face")
    assert ranked[0].flags["hooks"] == []
    assert round(sum(ranked[0].flags["contributions"].values()), 4) == ranked[0].score
    assert [c.as_dict() for c in ranked] == [c.as_dict() for c in proposals(cfg, f, segs)]


def test_delayed_response_and_setup_are_preserved(cfg):
    f = curves()
    f["gameplay_activity"][40] = f["novelty"][40] = f["audio_reaction"][41] = 1
    f["gameplay_activity"][50] = 0.6
    segs = [
        utter(14, 19, "Regarde bien juste devant toi !"),
        utter(19, 22, "Et là il commence"),
        utter(29, 34, "Il change de direction !"),
        utter(34.1, 39, "On y est enfin !"),
    ]
    c = proposals(cfg, f, segs)[0]
    assert c.start <= 14 and c.end >= 39 and c.end < 43
    assert c.flags["cut_risk"]  # continuity across a long uncertain dialogue gap needs review
    assert c.flags["hooks"] and all(any(s["text"] in quote for s in segs) for quote in c.flags["hooks"])


@pytest.mark.parametrize("at", [0, 199])
def test_edges_have_valid_bounds_and_cut_warning(at, cfg):
    f = curves()
    f["gameplay_activity"][at] = f["audio_reaction"][at] = 1
    c = proposals(cfg, f)[0]
    assert 0 <= c.start <= c.peak < c.end <= 100
    assert c.flags["cut_risk"]


def test_duration_cap_flags_unfinished_speech(cfg):
    cfg["stream_highlights"]["max_s"] = 20
    f = curves()
    f["gameplay_activity"][40] = f["audio_reaction"][40] = 1
    c = proposals(cfg, f, [utter(19, 65, "Elle commence seulement à expliquer")])[0]
    assert c.duration <= 20 and c.flags["cut_risk"] and not c.flags["hooks"]


def test_malformed_transcript_is_not_quoted():
    data = {
        "segments": [
            None,
            {"end": 0},
            utter(3, 2, "wrong timing"),
            utter(0, 400, "outside recording"),
            dict(utter(1, 3, "Probable silence."), no_speech_prob=0.99),
            dict(utter(7, 9, "en train de"), words=[{"word": 2}]),
        ]
    }
    clean = stream.usable_transcript(data, 12)
    assert len(clean["segments"]) == 1
    assert hunter._hooks(clean["segments"], 0, 12, 8) == []
    assert stream.usable_transcript(None, 12) is None


def test_motion_in_facecam_is_excluded_from_gameplay(cfg, monkeypatch):
    frames = [np.zeros((96, 160), np.uint8) for _ in range(5)]
    rng = np.random.default_rng(42)
    for frame in frames:
        frame[:48, :80] = rng.integers(10, 240, (48, 80), dtype=np.uint8)
    monkeypatch.setattr(media, "gray_frames", lambda *args: iter(zip(np.arange(5) * HOP, frames)))
    monkeypatch.setattr(layout.FaceFinder, "find", lambda self, img: [(0.1, 0.1, 0.7, 0.7)])
    src = dict(path="unused", width=160, height=96)
    box = dict(x=0, y=0, w=0.5, h=0.5)
    cam = stream.visual_features(src, box, 2.5, HOP, cfg["stream_highlights"])
    assert cam["facecam_activity"].max() > 0.5 and not cam["gameplay_activity"].any()
    no_cam = stream.visual_features(src, None, 2.5, HOP, cfg["stream_highlights"])
    assert not no_cam["facecam_activity"].any() and not no_cam["facecam_confidence"].any()
    tiny = stream.visual_features(src, dict(x=0, y=0, w=0.03, h=0.04), 2.5, HOP, cfg["stream_highlights"])
    assert not tiny["facecam_confidence"].any()


@pytest.mark.slow
def test_mode_review_preview_and_vertical_render(source, cfg):
    conn, sid = source
    pipeline.analyze(conn, cfg, sid, transcribe=False)
    classic = db.list_candidates(conn, sid)
    assert cli.main(["analyze", str(sid), "--mode", hunter.MODE, "--no-transcribe"]) == 0
    candidate = db.list_candidates(conn, sid, hunter.MODE)[0]
    first, last = candidate["start_s"], candidate["end_s"]
    src = db.get_source(conn, sid)
    with pytest.raises(ValueError, match="inside the recording"):
        db.save_stream_review(conn, candidate["id"], first, src["duration"] + 0.25, "")
    review_start = max(0, first - 0.25)
    review_end = min(src["duration"], last + 0.25)
    db.save_stream_review(conn, candidate["id"], review_start, review_end, "")
    db.add_label(conn, sid, first, last, 1)
    pipeline.analyze(conn, cfg, sid, transcribe=False, mode=hunter.MODE)
    assert db.list_candidates(conn, sid) == classic
    edited = db.get_candidate(conn, candidate["id"])
    assert edited["hook_edited"] and edited["hook"] is None
    assert edited["flags"]["review_bounds"] == [review_start, review_end]
    assert len(db.list_labels(conn, sid)) == 1
    proxy = pipeline.preview_proxy(cfg, src, dict(edited, start_s=review_start, end_s=review_end), "stacked")
    info = media.probe(proxy)
    assert info["height"] > info["width"]
    frame = media.bgr_frame_at(proxy, 0.5, (info["width"], info["height"]), width=180)
    # The fixture camera is a fixed blue rectangle; the gameplay contains a moving texture.
    top = frame[5 : frame.shape[0] // 4]
    assert top[:, :, 0].mean() > top[:, :, 2].mean() + 40
    assert frame[frame.shape[0] // 2 :].std() > 5
    (result,) = pipeline.render_candidates(conn, cfg, sid, [edited["id"]], ["tiktok"], layout_kind="stacked")
    assert cli.main(["render", str(sid), "--ids", str(edited["id"]), "--mode", "classic"]) != 0
    assert result["layout"] == "stacked" and Path(result["path"]).is_file()
    assert result["start"] <= review_start and result["end"] >= review_end and result["hook"] is None
    assert result["highlight"]["mode"] == hunter.MODE and result["validation"]["technically_valid"]
    assert cli.main(["eval", str(sid), "--mode", hunter.MODE]) == 0
    assert cli.main(["candidates", str(sid), "--mode", hunter.MODE]) == 0


def test_periodic_pitch_jump_needs_energy_to_be_a_reaction(cfg, monkeypatch):
    rate = 16000
    t = np.arange(rate // 2) / rate
    audio = (
        np.concatenate(
            [
                600 * np.sin(2 * np.pi * 150 * t),
                10000 * np.sin(2 * np.pi * 270 * t),
                10000 * np.sin(2 * np.pi * 270 * t),
            ]
        )
        .astype("<i2")
        .tobytes()
    )
    monkeypatch.setattr(media, "pcm_chunks", lambda *args: iter([audio[:1234], audio[1234:]]))
    settings = cfg["stream_highlights"]
    pitch = stream.pitch_changes(dict(duration=1.5, has_audio=1, path="sample"), HOP, rate, settings)
    assert pitch[1] > 0.9 and pitch[0] == pitch[2] == 0
    a = stream.audio_features(
        np.array([-35.0, -12.0, -12.0]), None, 1.5, HOP, settings, cfg["signals"]["keywords"], pitch
    )
    assert a["voiced_rise"][1] > 0.9
    assert a["voiced_rise"][2] == 0


def test_spoken_event_term_needs_visual_support(cfg):
    f = curves()
    f["chaos_words"] = np.ones(200)
    assert not proposals(cfg, f)
    f["gameplay_activity"][40] = f["audio_reaction"][41] = 1
    c = proposals(cfg, f)[0]
    assert c.flags["moment_type"] == "chaos cue"
    assert c.flags["hooks"] == [] and "spoken_chaos_cue" in c.flags["evidence"]


def test_review_screen_saves_cuts_and_removes_hook(source, cfg):
    streamlit = pytest.importorskip("streamlit.testing.v1")
    conn, sid = source
    row = dict(
        start=12,
        end=26,
        peak=20,
        score=3,
        hook="Original",
        signals={"event_reaction_sync": 0.8},
        flags={
            "mode": hunter.MODE,
            "moment_type": "event + reaction",
            "reason": "Observed change",
            "confidence": "low",
            "timing": {"clip_end": 26, "reaction_end": None},
            "evidence": {},
            "hooks": ["Proposed quote"],
            "cut_risk": True,
        },
    )
    (cid,) = db.replace_candidates(conn, sid, [row], mode=hunter.MODE)
    app = streamlit.AppTest.from_file(str(Path(cli.__file__).with_name("review_app.py"))).run()
    app.sidebar.selectbox[1].select(hunter.MODE).run()
    assert not app.exception
    app.number_input(key=f"end_{cid}").set_value(29.0)
    app.text_input(key=f"hook_{cid}").set_value("")
    app.button(key=f"save_{cid}").click().run()
    assert not app.exception
    saved = db.get_candidate(conn, cid)
    assert saved["flags"]["review_bounds"] == [12, 29] and saved["hook"] is None

def test_hooks_abstain_on_punctuated_fragments_and_distant_speech():
    segments = [utter(4, 6, "Ah ! Je reprends, on était à quoi ?"),
                utter(8, 10, "Pourquoi la voiture recule ?"),
                utter(22, 24, "Regarde bien le panneau !")]
    assert hunter._hooks(segments, 0, 30, 7) == ["« Pourquoi la voiture recule ? »"]

def test_protected_cut_can_end_exactly_at_a_fractional_source_duration(cfg):
    duration = 180.00073
    plan, = variants.plan_variants(1, 150., duration, 159., duration, ["tiktok"], cfg, protected=True)
    assert plan.start == 150. and plan.end == duration
