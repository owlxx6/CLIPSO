import numpy as np

from clipper import fixture
from clipper.media import parse_ts, probe
from clipper.signals import audio, chat, viewer_clips, visual
from clipper.signals import transcript as tr

HOP = 0.5


def _window_max(curve, t0, t1):
    return float(np.max(curve[int(t0 / HOP):int(t1 / HOP)]))


def test_probe(fx):
    info = probe(fx["video"])
    assert abs(info["duration"] - 180) < 1 and info["has_audio"] and info["width"] == 640


def test_loudness_finds_bursts(fx):
    db = audio.loudness_db(fx["video"], HOP)
    assert abs(len(db) - 360) <= 2
    z = audio.excitement(db, HOP)
    for b in fixture.BURSTS:
        assert _window_max(z, b, b + 3) > 4
    assert float(np.median(z)) < 1


def test_chat_curve_peaks_after_delay_compensation(fx, cfg):
    curve = chat.chat_curve(chat.load_chat(fx["chat"]), 180, HOP, cfg["signals"]["chat"])
    for b in (fixture.BURSTS[0], fixture.BURSTS[2]):
        peak_t = (int(b / HOP) - 20 + int(np.argmax(curve[int(b / HOP) - 20:int(b / HOP) + 20]))) * HOP
        assert abs(peak_t - b) <= 8
        assert _window_max(curve, b - 8, b + 8) > 3
    assert _window_max(curve, fixture.BURSTS[1] - 5, fixture.BURSTS[1] + 5) < 2


def test_viewer_clip_marks_middle_burst(fx, cfg):
    curve = viewer_clips.clips_curve(viewer_clips.load_clips(fx["clips"]), 180, HOP, cfg["signals"]["viewer_clips"])
    assert abs(int(np.argmax(curve)) * HOP - fixture.BURSTS[1]) <= 3


def test_keywords_fire_on_hype_words(fx, cfg):
    data = tr.load(fx["transcript"])
    curve = tr.keyword_curve(data, cfg["signals"]["keywords"], 180, HOP, "fr")
    for b in fixture.BURSTS:
        assert _window_max(curve, b - 1, b + 4) >= 0.9
    assert _window_max(curve, 60, 80) == 0


def test_keyword_normalisation_handles_accents_and_apostrophes():
    words = [{"start": 1.0, "end": 1.2, "word": " C’est"}, {"start": 1.3, "end": 1.5, "word": "QUOI"},
             {"start": 1.6, "end": 1.8, "word": "ça!"}]
    assert tr.keyword_hits(words, ["c'est quoi ça"]) == [(1.0, 1.0)]


def test_letterbox_flags_fake_cutscene(fx, cfg):
    curve = visual.letterbox_curve(fx["video"], 180, HOP, cfg["analysis"]["visual"])
    lb0, lb1 = fixture.LETTERBOX
    assert curve[int((lb0 + 3) / HOP):int((lb1 - 3) / HOP)].mean() == 1.0
    assert curve[int(20 / HOP):int(100 / HOP)].sum() == 0


def test_parse_ts():
    assert parse_ts("1:02:03") == 3723
    assert parse_ts("02:03.5") == 123.5
    assert parse_ts(42) == 42.0


def test_files_saved_with_a_bom_still_load(tmp_path):
    """Excel and Windows PowerShell 5 write UTF-8 with a byte-order mark."""
    from clipper import evaluate
    from clipper.signals import chat, viewer_clips

    bom = "﻿"
    (tmp_path / "chat.csv").write_text(bom + "time,message\n00:01:05,KEKW\n", encoding="utf-8")
    (tmp_path / "clips.json").write_text(bom + '{"data": [{"vod_offset": 73, "duration": 30}]}', encoding="utf-8")
    (tmp_path / "truth.csv").write_text(bom + "start,end\n40,60\n", encoding="utf-8")
    assert chat.load_chat(tmp_path / "chat.csv") == [(65.0, "KEKW")]
    assert viewer_clips.load_clips(tmp_path / "clips.json")[0]["vod_offset"] == 73.0
    assert evaluate.load_truth(tmp_path / "truth.csv")[0][:2] == (40.0, 60.0)
