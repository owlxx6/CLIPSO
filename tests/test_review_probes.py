"""Independent review probes for gta-ai-clipper.

Each test states the behaviour the tool needs before it is used for paid clipping campaigns.
A failing probe is a reproduced defect. Drop this file into the repo's tests/ folder (it uses the repo's
conftest fixtures: fx, cfg, source) and run: pytest -q tests/test_review_probes.py

R1..R11 refer to the findings of the review of 2026-11-19. Some probes patch internals (render.render_clip,
media.run, pipeline.tr.transcribe, pipeline.audio.loudness_db): if a fix renames them, adapt the probe; an
AttributeError there is not a defect.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from clipper import db, evaluate, media, pipeline, render, subtitles

QUIET = dict(log=lambda *_: None)


def _analyze(conn, cfg, sid):
    return pipeline.analyze(conn, cfg, sid, transcribe=False, **QUIET)


def _top(conn, sid):
    return db.list_candidates(conn, sid)[0]


def _same_moment(conn, sid, cand):
    return next(c for c in db.list_candidates(conn, sid) if abs(c["peak_s"] - cand["peak_s"]) < 1.0)


def _render(conn, cfg, sid, cid, **kw):
    return pipeline.render_candidates(conn, cfg, sid, [cid], kw.pop("platforms", ["tiktok"]), **QUIET, **kw)


# --- R1 rights and fail-closed behaviour ---------------------------------------------------------------------------

def test_rights_blank_is_rejected(cfg, fx):
    conn = db.connect(cfg)
    with pytest.raises(ValueError):
        pipeline.ingest(conn, cfg, fx["video"], rights="   ")


def test_render_fails_closed_when_rights_are_missing(source, cfg):
    conn, sid = source
    _analyze(conn, cfg, sid)
    try:
        conn.execute("UPDATE sources SET rights = '' WHERE id = ?", (sid,))
        conn.commit()
    except sqlite3.IntegrityError:
        return  # the database itself refuses a source without rights: fail-closed
    cand = _top(conn, sid)
    try:
        _render(conn, cfg, sid, cand["id"])
    except Exception:
        pass
    assert db.list_exports(conn, source_id=sid) == [], "a source without rights was rendered"


# --- R2 campaign policy --------------------------------------------------------------------------------------------

def test_campaign_clip_carries_the_ad_disclosure_by_default(cfg, fx):
    conn = db.connect(cfg)
    sid = pipeline.ingest(conn, cfg, fx["video"], rights="campaign:test", campaign="Paid campaign",
                          chat_file=fx["chat"], clips_file=fx["clips"], transcript_file=fx["transcript"])
    pipeline.set_layout(cfg, db.get_source(conn, sid), "stacked", fx["facecam"])
    _analyze(conn, cfg, sid)
    res = _render(conn, cfg, sid, _top(conn, sid)["id"])[0]
    assert res.get("path"), res
    meta = json.loads(Path(res["path"]).with_suffix(".json").read_text(encoding="utf-8"))
    assert meta.get("disclosure"), "a paid-campaign clip was rendered without the PUBLICITÉ label"


# --- R5 / R6 candidates and exports across repeated analysis ------------------------------------------------------------

def test_reanalysis_keeps_exports_attached_to_their_moment(source, cfg):
    conn, sid = source
    _analyze(conn, cfg, sid)
    cand = _top(conn, sid)
    first = _render(conn, cfg, sid, cand["id"])[0]
    _analyze(conn, cfg, sid)
    again = _same_moment(conn, sid, cand)
    assert [e["id"] for e in db.list_exports(conn, candidate_id=again["id"])] == [first["export_id"]]
    orphans = conn.execute("SELECT COUNT(*) FROM exports e LEFT JOIN candidates c ON c.id = e.candidate_id "
                           "WHERE e.candidate_id IS NOT NULL AND c.id IS NULL").fetchone()[0]
    assert orphans == 0, f"{orphans} export(s) point at candidates that no longer exist"


def test_replace_still_works_after_reanalysis(source, cfg):
    conn, sid = source
    _analyze(conn, cfg, sid)
    cand = _top(conn, sid)
    _render(conn, cfg, sid, cand["id"])
    _analyze(conn, cfg, sid)
    again = _same_moment(conn, sid, cand)
    res = _render(conn, cfg, sid, again["id"], hook="nouveau hook", replace=True)[0]
    assert res["blocked_reason"] is None, res["blocked_reason"]
    rows = db.list_exports(conn, source_id=sid, platform="tiktok")
    assert len(rows) == 1 and rows[0]["hook"] == "nouveau hook"


def test_reanalysis_keeps_an_edited_hook(source, cfg):
    conn, sid = source
    _analyze(conn, cfg, sid)
    cand = _top(conn, sid)
    db.set_hook(conn, cand["id"], "mon hook à moi")
    _analyze(conn, cfg, sid)
    assert _same_moment(conn, sid, cand)["hook"] == "mon hook à moi"


# --- R3 / R4 atomic replacement and output validation (+ subtitle injection guard) -------------------------------------------------------------------

def test_failed_replacement_keeps_the_previous_export(source, cfg, monkeypatch):
    conn, sid = source
    _analyze(conn, cfg, sid)
    cand = _top(conn, sid)
    first = _render(conn, cfg, sid, cand["id"])[0]

    def crash(*args, **kwargs):
        raise media.MediaError("simulated encoder crash")

    monkeypatch.setattr(render, "render_clip", crash)
    try:
        _render(conn, cfg, sid, cand["id"], hook="autre", replace=True)
    except Exception:
        pass
    assert Path(first["path"]).exists(), "the previous clip was deleted before its replacement existed"
    assert [e["id"] for e in db.list_exports(conn, source_id=sid)] == [first["export_id"]]


def test_crash_mid_encode_leaves_no_partial_clip(source, cfg, monkeypatch):
    conn, sid = source
    _analyze(conn, cfg, sid)
    cand = _top(conn, sid)
    real_run = media.run

    def crash_on_render(cmd, *args, **kwargs):
        out = str(cmd[-1])
        if "-filter_complex" in cmd and not out.endswith((".jpg", ".png")):
            Path(out).write_bytes(b"\x00" * 4096)  # what an interrupted ffmpeg leaves behind
            raise media.MediaError("simulated crash mid-encode")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(media, "run", crash_on_render)
    try:
        _render(conn, cfg, sid, cand["id"])
    except Exception:
        pass
    out_dir = pipeline.export_dir(cfg, sid, cand["id"])
    left = sorted(p.name for p in out_dir.rglob("*") if p.is_file()) if out_dir.exists() else []
    assert left == [], f"partial files left behind: {left}"
    assert db.list_exports(conn, source_id=sid) == []


def test_broken_output_is_not_recorded_as_an_export(source, cfg, monkeypatch):
    conn, sid = source
    _analyze(conn, cfg, sid)
    cand = _top(conn, sid)

    def wrong_render(src, start, end, layout, ass_text, out_path, cfg, *args, **kwargs):
        # A playable MP4, but 1 s long, 320x240 and silent: what a bad filter or a truncated VOD produces.
        subprocess.run([media.ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x240:d=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)], check=True)
        return Path(out_path)

    monkeypatch.setattr(render, "render_clip", wrong_render)
    try:
        _render(conn, cfg, sid, cand["id"])
    except Exception:
        pass
    assert db.list_exports(conn, source_id=sid) == [], "a 1-second 320x240 file was recorded as a finished clip"


def test_hook_text_cannot_inject_subtitle_events(cfg):
    evil = "ok\nDialogue: 0,0:00:00.00,0:00:30.00,Sub,,0,0,0,,INJECTED {\\fs300}\r\nend"
    ass = subtitles.build_ass([], cfg, "stacked", hook=evil, clip_duration=10.0)
    assert not any(line.startswith("Dialogue:") and ",INJECTED" in line.split(",,0,0,0,,", 1)[-1][:9]
                   for line in ass.splitlines())
    assert "{\\fs300}" not in ass


# --- R7 evaluation -------------------------------------------------------------------------------------------------

def test_eval_counts_each_highlight_once():
    truth = [(40.0, 60.0)]
    cands = [dict(start=38.0, end=58.0, peak=45.0), dict(start=44.0, end=64.0, peak=50.0)]
    m = evaluate.evaluate(cands, truth, k=2)
    assert m["precision"] == 0.5, f"two clips of one highlight both counted as hits: {m}"


def test_eval_matches_on_the_peak_before_overlap():
    truth = [(10.0, 30.0), (32.0, 50.0)]
    cands = [dict(start=20.0, end=40.0, peak=35.0),   # peak inside the 2nd highlight, overlaps the 1st
             dict(start=12.0, end=28.0, peak=20.0)]   # inside the 1st
    m = evaluate.evaluate(cands, truth, k=2)
    assert (m["precision"], m["recall"]) == (1.0, 1.0), m


# --- R8 / R11 caches -----------------------------------------------------------------------------------------------------

def test_force_reanalysis_redoes_the_transcript(cfg, fx, monkeypatch):
    conn = db.connect(cfg)
    sid = pipeline.ingest(conn, cfg, fx["video"], rights="t", chat_file=fx["chat"], clips_file=fx["clips"])
    pipeline.set_layout(cfg, db.get_source(conn, sid), "stacked", fx["facecam"])
    calls = []

    def fake_transcribe(path, tcfg, log=print, *args, **kwargs):
        calls.append(tcfg.get("model"))
        return json.loads(Path(fx["transcript"]).read_text(encoding="utf-8"))

    monkeypatch.setattr(pipeline.tr, "transcribe", fake_transcribe)
    pipeline.analyze(conn, cfg, sid, transcribe=True, **QUIET)
    pipeline.analyze(conn, cfg, sid, transcribe=True, **QUIET)
    assert len(calls) == 1, "the transcript should be cached"
    pipeline.analyze(conn, cfg, sid, transcribe=True, force=True, **QUIET)
    assert len(calls) == 2, "analyze --force kept the old transcript"


def test_changed_whisper_settings_invalidate_the_transcript(cfg, fx, monkeypatch):
    conn = db.connect(cfg)
    sid = pipeline.ingest(conn, cfg, fx["video"], rights="t")
    calls = []

    def fake_transcribe(path, tcfg, log=print, *args, **kwargs):
        calls.append(tcfg.get("model"))
        return json.loads(Path(fx["transcript"]).read_text(encoding="utf-8"))

    monkeypatch.setattr(pipeline.tr, "transcribe", fake_transcribe)
    pipeline.analyze(conn, cfg, sid, transcribe=True, **QUIET)
    cfg2 = copy.deepcopy(cfg)
    cfg2["analysis"]["transcript"]["model"] = "large-v3-turbo"
    pipeline.analyze(conn, cfg2, sid, transcribe=True, **QUIET)
    assert len(calls) == 2, "switching the Whisper model reused the transcript made with the old one"


def test_changed_audio_settings_invalidate_the_loudness_cache(source, cfg, monkeypatch):
    conn, sid = source
    _analyze(conn, cfg, sid)
    calls = []
    real = pipeline.audio.loudness_db
    monkeypatch.setattr(pipeline.audio, "loudness_db", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    cfg2 = copy.deepcopy(cfg)
    cfg2["analysis"]["audio"]["sample_rate"] = 8000
    pipeline.analyze(conn, cfg2, sid, transcribe=False, **QUIET)
    assert calls, "changing analysis.audio.sample_rate reused loudness measured with the old setting"


def test_replaced_video_file_invalidates_caches(cfg, fx, tmp_path, monkeypatch):
    video = tmp_path / "vod.mp4"
    shutil.copy2(fx["video"], video)
    conn = db.connect(cfg)
    sid = pipeline.ingest(conn, cfg, video, rights="t", transcript_file=fx["transcript"])
    _analyze(conn, cfg, sid)
    with open(video, "ab") as fh:  # the VOD at the same path is re-downloaded / re-encoded
        fh.write(b"\x00" * 1024)
    calls = []
    real = pipeline.audio.loudness_db
    monkeypatch.setattr(pipeline.audio, "loudness_db", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    _analyze(conn, cfg, sid)
    assert calls, "a different file at the same path reused the old loudness cache"


# --- R10 Windows-like conditions (reproduced on Linux) --------------------------------------------------------------

def test_paths_with_spaces_accents_quotes_and_percent(tmp_path, monkeypatch, fx):
    home = tmp_path / "Dossier d'été [1] 100% #test"
    monkeypatch.setenv("CLIPPER_HOME", str(home))
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(home / "data"))
    monkeypatch.delenv("CLIPPER_CONFIG", raising=False)
    from clipper import cli, config

    home.mkdir(parents=True)
    video = home / "VOD l'été & co.mp4"
    shutil.copy2(fx["video"], video)
    assert cli.main(["ingest", str(video), "--rights", "test", "--transcript", str(fx["transcript"]),
                     "--chat", str(fx["chat"]), "--clips", str(fx["clips"])]) == 0
    assert cli.main(["layout", "1", "--set", "0.725,0.0333,0.25,0.3333"]) == 0
    assert cli.main(["analyze", "1", "--no-transcribe"]) == 0
    assert cli.main(["render", "1", "--top", "1", "--platforms", "tiktok"]) == 0
    conn = db.connect(config.load_config())
    exp = db.list_exports(conn, platform="tiktok")
    assert len(exp) == 1 and Path(exp[0]["path"]).exists() and Path(exp[0]["path"]).with_suffix(".jpg").exists()
    info = media.probe(exp[0]["path"])
    assert (info["width"], info["height"]) == (1080, 1920)


def test_listing_survives_a_cp1252_console(source, cfg):
    """Windows uses the ANSI code page (cp1252) when output is redirected or piped."""
    conn, sid = source
    _analyze(conn, cfg, sid)
    conn.execute("UPDATE candidates SET hook = '😂 il a cru → ok' WHERE source_id = ?", (sid,))  # hooks with emoji
    conn.commit()
    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
    proc = subprocess.run([sys.executable, "-m", "clipper", "candidates", str(sid)], env=env,
                          capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")[-400:]
