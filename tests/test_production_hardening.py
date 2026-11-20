"""Regression cases for source identity, decoder failures, assignment and cut planning."""
import os
import shutil
import sys
from pathlib import Path

import pytest

from clipper import cli, db, evaluate, media, pipeline, variants
from clipper.signals import transcript as tr


def test_label_verifies_source_before_writing(source, cfg, fx):
    conn, sid = source
    other = pipeline.ingest(conn, cfg, fx["video"], rights="synthetic second source")
    db.replace_candidates(conn, sid, [{"start": 10, "end": 30, "peak": 23, "score": 1}])
    cid = str(db.list_candidates(conn, sid)[0]["id"])
    assert cli.main(["label", str(other), "--candidate", cid, "--good"]) != 0
    assert not db.list_labels(conn, other) and not db.list_labels(conn, sid)
    assert cli.main(["label", str(sid), "--candidate", cid, "--good"]) == 0
    assert db.list_labels(conn, sid)[0]["label"] == 1


def test_sidecar_copy_failure_is_atomic(cfg, fx, monkeypatch):
    conn = db.connect(cfg)
    monkeypatch.setattr(shutil, "copy2", lambda *_: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        pipeline.ingest(conn, cfg, fx["video"], rights="test", chat_file=fx["chat"])
    assert not db.list_sources(conn)
    assert not list((pipeline.config.data_dir(cfg) / "sources").iterdir())


def test_import_matches_content_even_if_file_stats_are_identical(cfg, tmp_path, monkeypatch):
    video = tmp_path / "vod"
    video.write_bytes(b"first")
    src = {"id": 9, "path": str(video)}
    wd = pipeline.workdir(cfg, 9)
    supplied = {"segments": [{"start": 0, "end": 2, "text": "first"}]}
    path = tmp_path / "words.json"
    tr.save(path, supplied)
    pipeline.bind_transcript(video, path, wd, 5)
    monkeypatch.setattr(tr, "transcribe", lambda *_: pytest.fail("unnecessary speech inference"))
    assert pipeline.get_transcript(cfg, src, force=True) == supplied
    stat = video.stat()
    video.write_bytes(b"other")
    os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert pipeline.get_transcript(cfg, src, transcribe=False) is None
    assert tr.load(wd / "transcript.bound.json")["transcript"] == supplied
    tr.save(wd / "transcript.imported.json", supplied)
    (wd / "transcript.bound.json").unlink()
    assert pipeline.get_transcript(cfg, src, transcribe=False) is None


@pytest.mark.parametrize("kind", ["audio", "video"])
def test_streaming_reports_invalid_and_truncated_media(tmp_path, fx, kind):
    decode = media.pcm_chunks if kind == "audio" else lambda p: media.gray_frames(p, 1, 16, 16)
    bad = tmp_path / "broken.mp4"
    for payload in (b"not a media file", Path(fx["video"]).read_bytes()[:8192]):
        bad.write_bytes(payload)
        with pytest.raises(media.MediaError):
            list(decode(bad))


def test_partial_stream_then_failed_process_reports_stderr(monkeypatch):
    original = media.subprocess.Popen
    def failed(*args, **kwargs):
        code = "import sys; sys.stdout.buffer.write(b'abcd'); sys.stderr.write('damaged packet'); sys.exit(7)"
        return original([sys.executable, "-c", code], **kwargs)
    monkeypatch.setattr(media.subprocess, "Popen", failed)
    chunks = media.pcm_chunks("source", chunk_bytes=4)
    assert next(chunks) == b"abcd"
    with pytest.raises(media.MediaError, match="damaged packet"):
        next(chunks)


def test_sparse_keyframes_have_presentation_times(tmp_path):
    path = tmp_path / "irregular.mp4"
    media.run([media.ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x48:r=10:d=6",
               "-c:v", "libx264", "-g", "100", "-sc_threshold", "0", "-force_key_frames", "0,1.3,4.7", str(path)])
    stamps = [t for t, _ in media.gray_frames(path, 2, 8, 8, keyframes_only=True)]
    assert stamps == pytest.approx([0, 1.3, 4.7], abs=0.01)


def test_video_only_is_valid_but_decoding_absent_audio_fails(tmp_path, cfg):
    video = tmp_path / "quiet.mp4"
    media.run([media.ffmpeg_bin(), "-v", "error", "-f", "lavfi", "-i", "color=s=64x48:r=10:d=4",
               "-c:v", "libx264", str(video)])
    with pytest.raises(media.MediaError):
        list(media.pcm_chunks(video))
    conn = db.connect(cfg)
    source = db.get_source(conn, pipeline.ingest(conn, cfg, video, rights="test"))
    curves = pipeline.raw_signals(cfg, source)
    assert len(curves["_db"]) == 8 and all(curves["_db"] == -90)


def test_global_assignment_preserves_cardinality_and_quality():
    truth = [(10, 20), (22, 32)]
    flexible = dict(start=10, end=29, peak=16)
    first_only = dict(start=9, end=19, peak=14)
    for predictions, expected in (([flexible, first_only], [1, 0]), ([first_only, flexible], [0, 1])):
        result = evaluate.evaluate(predictions, truth)
        assert result["hits"] == expected
        assert result["precision"] == result["recall"] == result["f1"] == 1
    closer = dict(start=10, end=20, peak=15)
    result = evaluate.evaluate([flexible, closer], truth[:1])
    assert result["hits"] == [None, 0] and result["precision"] == 0.5
    shifted = evaluate.evaluate([dict(start=12, end=22, peak=16)], truth[:1])
    assert shifted["hits"] == [0]


@pytest.mark.parametrize("predictions, truth", [([], [(0, 1)]), ([dict(start=0, end=2, peak=1)], []), ([], [])])
def test_empty_evaluation(predictions, truth):
    m = evaluate.evaluate(predictions, truth)
    assert (m["precision"], m["recall"], m["f1"]) == (0, 0, 0)


def test_cut_planner_preserves_words_and_setup(cfg):
    transcript = {"segments": [{"start": 9.5, "end": 13, "text": "setup",
                                "words": [{"start": 9.5, "end": 13, "word": "setup"}]},
                               {"start": 29, "end": 31, "text": "payoff",
                                "words": [{"start": 29, "end": 31, "word": "payoff"}]}]}
    plans = variants.plan_variants(4, 10, 30, 21, 60, ["tiktok", "instagram"], cfg,
                                   transcript=transcript, duration_limits=(20, 24))
    assert (plans[0].start, plans[0].end) == (9.5, 31)
    for plan in plans:
        assert plan.start <= 9.5 and plan.end >= 31 and 20 <= plan.end - plan.start <= 24
    silent = variants.plan_variants(4, 10, 30, 21, 60, ["tiktok"], cfg, cues=[9.75, 30.25])[0]
    assert (silent.start, silent.end) == (9.75, 30.25)
