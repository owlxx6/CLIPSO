import json
import shutil
from pathlib import Path

import pytest

from clipper import db, dedup, media, pipeline, render


@pytest.mark.slow
def test_render_vertical_clip_with_captions_and_duplicate_guard(cfg, source):
    conn, sid = source
    pipeline.analyze(conn, cfg, sid, transcribe=False)
    cand = db.list_candidates(conn, sid)[0]
    first = pipeline.render_candidates(conn, cfg, sid, [cand["id"]], ["tiktok"], log=lambda *_: None)[0]
    out = Path(first["path"])
    info = media.probe(out)
    assert (info["width"], info["height"]) == (1080, 1920)
    assert info["has_audio"]
    assert abs(info["duration"] - (first["end"] - first["start"])) < 0.5
    assert first["blocked_reason"] is None
    assert out.with_suffix(".jpg").exists() and out.with_suffix(".txt").exists()
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["rights"] == "test:synthetic" and meta["layout"] == "stacked"

    # Same moment, same platform: skipped before rendering.
    again = pipeline.render_candidates(conn, cfg, sid, [cand["id"]], ["tiktok"], log=lambda *_: None)[0]
    assert again["path"] is None and "already exported" in again["blocked_reason"]

    # replace=True re-renders an export that was not posted yet (e.g. a new hook) and removes the old file.
    redo = pipeline.render_candidates(conn, cfg, sid, [cand["id"]], ["tiktok"], hook="nouveau hook", replace=True,
                                      log=lambda *_: None)[0]
    assert redo["blocked_reason"] is None and Path(redo["path"]).exists()
    rows = db.list_exports(conn, platform="tiktok", candidate_id=cand["id"])
    assert len(rows) == 1 and rows[0]["hook"] == "nouveau hook"

    # Same moment on another platform is allowed (one post per platform).
    yt = pipeline.render_candidates(conn, cfg, sid, [cand["id"]], ["youtube"], log=lambda *_: None)[0]
    assert yt["blocked_reason"] is None

    # A different moment on the same platform is not mistaken for a duplicate.
    other = db.list_candidates(conn, sid)[1]
    res = pipeline.render_candidates(conn, cfg, sid, [other["id"]], ["tiktok"], log=lambda *_: None)[0]
    assert res["blocked_reason"] is None


@pytest.mark.slow
def test_same_footage_from_another_copy_is_blocked_before_rendering(cfg, source, fx, tmp_path):
    conn, sid = source
    pipeline.analyze(conn, cfg, sid, transcribe=False)
    cand = db.list_candidates(conn, sid)[0]
    first = pipeline.render_candidates(conn, cfg, sid, [cand["id"]], ["tiktok"], log=lambda *_: None)[0]
    assert first["blocked_reason"] is None

    copy = tmp_path / "same_vod_downloaded_again.mp4"
    shutil.copy2(fx["video"], copy)
    sid2 = pipeline.ingest(conn, cfg, copy, rights="test:synthetic", title="copy", chat_file=fx["chat"],
                           clips_file=fx["clips"], transcript_file=fx["transcript"])
    pipeline.set_layout(cfg, db.get_source(conn, sid2), "stacked", fx["facecam"])
    pipeline.analyze(conn, cfg, sid2, transcribe=False)
    twin = next(c for c in db.list_candidates(conn, sid2) if abs(c["peak_s"] - cand["peak_s"]) < 2)
    res = pipeline.render_candidates(conn, cfg, sid2, [twin["id"]], ["tiktok", "youtube"], log=lambda *_: None)
    by_platform = {r["platform"]: r for r in res}
    assert by_platform["tiktok"]["path"] is None
    assert "looks like export" in by_platform["tiktok"]["blocked_reason"]
    assert by_platform["youtube"]["blocked_reason"] is None  # never posted there yet
    assert not list(pipeline.export_dir(cfg, sid2, twin["id"]).glob("tiktok_*.mp4"))


@pytest.mark.slow
def test_other_layouts_render(cfg, source, tmp_path):
    conn, sid = source
    src = db.get_source(conn, sid)
    for kind in ("blur_fit", "center_crop"):
        out = render.render_clip(src["path"], 30, 36, {"kind": kind, "facecam": None}, "[Script Info]\n",
                                 tmp_path / f"{kind}.mp4", cfg, zoom=1.04)
        info = media.probe(out)
        assert (info["width"], info["height"]) == (1080, 1920)


@pytest.mark.slow
def test_signatures_tell_moments_apart(fx, tmp_path):
    def cut(start, name, width=320):
        out = tmp_path / name
        media.run([media.ffmpeg_bin(), "-nostdin", "-y", "-v", "error", "-ss", str(start), "-t", "12", "-i",
                   str(fx["video"]), "-vf", f"scale={width}:-2", "-c:v", "libx264", "-preset", "ultrafast", "-an",
                   str(out)])
        return dedup.signature(out)

    a, a_shifted, a_small, other = cut(30, "a.mp4"), cut(31.5, "a2.mp4"), cut(30, "a3.mp4", 200), cut(140, "b.mp4")
    assert dedup.similarity(a, a) == 1.0
    assert dedup.similarity(a, a_shifted) >= 0.8
    assert dedup.similarity(a, a_small) >= 0.8
    assert dedup.similarity(a, other) < 0.5
    # Reading a window straight from the source gives the same signature as the cut file.
    assert dedup.similarity(a, dedup.signature(fx["video"], start=30, duration=12)) >= 0.8


def test_gameplay_region_leaves_out_the_facecam():
    assert dedup.gameplay_region(None) == (0.0, 0.0, 1.0, 1.0)
    x, y, w, h = dedup.gameplay_region({"x": 0.725, "y": 0.0333, "w": 0.25, "h": 0.3333})
    assert (x, y, h) == (0.0, 0.0, 1.0) and w == pytest.approx(0.725)
    x, y, w, h = dedup.gameplay_region({"x": 0.02, "y": 0.70, "w": 0.2, "h": 0.28})  # bottom-left cam
    assert (y, h) == (0.0, 1.0) and x == pytest.approx(0.22) and w == pytest.approx(0.78)
    x, y, w, h = dedup.gameplay_region({"x": 0.3, "y": 0.75, "w": 0.4, "h": 0.25})  # wide cam at the bottom
    assert (x, y, w) == (0.0, 0.0, 1.0) and h == pytest.approx(0.75)
    assert dedup.crop_filter((0.0, 0.0, 0.725, 1.0)).startswith("crop=w=iw*0.7250")
    assert dedup.window(["a", "b", "c", "d", "e"], 10.0, 11.0, 13.0) == ["b", "c"]


def test_similarity_ignores_flat_frames():
    flat = [dedup.FLAT] * 10
    assert dedup.similarity(flat, flat) == 0.0


@pytest.mark.slow
def test_posted_export_is_kept_when_replacing(cfg, source, capsys):
    from clipper import cli

    conn, sid = source
    pipeline.analyze(conn, cfg, sid, transcribe=False)
    cand = db.list_candidates(conn, sid)[0]
    first = pipeline.render_candidates(conn, cfg, sid, [cand["id"]], ["tiktok"], log=lambda *_: None)[0]
    url = "https://www.tiktok.com/@example/video/1"
    assert cli.main(["posted", str(first["export_id"]), "--url", url]) == 0

    again = pipeline.render_candidates(conn, cfg, sid, [cand["id"]], ["tiktok"], hook="autre hook", replace=True,
                                       log=lambda *_: None)[0]
    assert again["path"] is None and "already exported" in again["blocked_reason"]
    assert Path(first["path"]).exists()
    assert db.post_for(conn, first["export_id"])["url"] == url

    assert cli.main(["exports", "--platform", "tiktok"]) == 0
    assert f"posted {url}" in capsys.readouterr().out
    with pytest.raises(ValueError):
        db.add_post(conn, 999, url)
