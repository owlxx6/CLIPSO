"""Command line entry point: `clipper <command>` (or `python -m clipper <command>`)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from clipper import __version__, campaign, config, db, evaluate, fixture, media, pipeline
from clipper import layout as layout_mod
from clipper.config import FONTS_DIR
from clipper.signals import transcript as tr

PLATFORMS_DEFAULT = "tiktok,instagram,youtube"


def _open(source_id: int) -> tuple:
    base = config.load_config()
    conn = db.connect(base)
    source = db.get_source(conn, source_id)
    return conn, pipeline.source_config(source), source


def _cand_dicts(rows: list[dict]) -> list[dict]:
    return [{"start": r["start_s"], "end": r["end_s"], "peak": r["peak_s"], **r} for r in rows]


def _print_candidates(rows: list[dict], labels: list[dict], top: int) -> None:
    print(f"{'id':>5} {'rank':>4}  {'start':>10}  {'end':>10} {'dur':>5} {'score':>6}  {'label':>5}  top signals / hook")
    for r in rows[:top]:
        sig = r.get("signals") or {}
        best = sorted(sig.items(), key=lambda kv: -kv[1])[:3]
        sig_txt = " ".join(f"{k}={v:.1f}" for k, v in best if v > 0)
        lab = db.label_for(labels, r["start_s"], r["end_s"])
        lab_txt = {1: "good", 0: "bad"}.get(lab, "")
        flag = " [check spoiler]" if (r.get("flags") or {}).get("spoiler_review") else ""
        hook = (r.get("hook") or "")[:40]
        print(f"{r['id']:>5} {r['rank']:>4}  {media.fmt_ts(r['start_s']):>10}  {media.fmt_ts(r['end_s']):>10} "
              f"{r['end_s'] - r['start_s']:>5.1f} {r['score']:>6.2f}  {lab_txt:>5}  {sig_txt}{flag}  {hook}")


# --- commands ----------------------------------------------------------------------------------------------------

def cmd_doctor(args) -> int:
    essential_ok = True

    def line(ok: bool, name: str, detail: str = "", essential: bool = True) -> None:
        nonlocal essential_ok
        if essential and not ok:
            essential_ok = False
        mark = "OK " if ok else ("!! " if essential else "-- ")
        print(f"[{mark}] {name}{': ' + detail if detail else ''}")

    line(sys.version_info >= (3, 10), "python", sys.version.split()[0])
    try:
        exe = media.ffmpeg_bin()
        version = media.run([exe, "-hide_banner", "-version"]).stdout.splitlines()[0]
        line(True, "ffmpeg", version)
        encoders, filters = media.encoders_and_filters()
        line("libx264" in encoders, "encoder libx264")
        line("h264_qsv" in encoders, "encoder h264_qsv (Intel Arc / Quick Sync)", "set render.encoder to use it",
             essential=False)
        line("h264_videotoolbox" in encoders, "encoder h264_videotoolbox (macOS)", essential=False)
        line("subtitles" in filters, "filter subtitles (libass)",
             "" if "subtitles" in filters else "needed for captions; macOS: `brew install ffmpeg-full`")
        line("loudnorm" in filters, "filter loudnorm")
        media.ffprobe_bin()
        line(True, "ffprobe")
    except media.MediaError as exc:
        line(False, "ffmpeg", str(exc))
    try:
        import cv2

        finder = layout_mod.FaceFinder()
        line(True, "opencv + face detector", f"{cv2.__version__}, {finder.name}")
    except Exception as exc:  # noqa: BLE001
        line(False, "opencv + face detector", str(exc))
    whisper_ok, streamlit_ok = tr.available(), importlib.util.find_spec("streamlit") is not None
    line(whisper_ok, "faster-whisper (speech signals)", "" if whisper_ok else 'pip install -e ".[transcribe]"',
         essential=False)
    line(streamlit_ok, "streamlit (review app)", "" if streamlit_ok else 'pip install -e ".[review]"',
         essential=False)
    line(any(FONTS_DIR.glob("*.ttf")), "caption font", ", ".join(p.name for p in FONTS_DIR.glob("*.ttf")))
    cfg = config.load_config()
    line(os.access(config.data_dir(cfg), os.W_OK), "data folder", str(config.data_dir(cfg)))
    print("\nready" if essential_ok else "\nfix the lines marked !! first")
    return 0 if essential_ok else 1


def cmd_demo(args) -> int:
    cfg = config.load_config()
    out = config.data_dir(cfg) / "demo"
    print(f"generating a 3-minute synthetic stream in {out} ...")
    fx = fixture.make_fixture(out)
    conn = db.connect(cfg)
    sid = pipeline.ingest(conn, cfg, fx["video"], rights="demo:synthetic-test-video", title="demo stream",
                          chat_file=fx["chat"], clips_file=fx["clips"], transcript_file=fx["transcript"])
    source = db.get_source(conn, sid)
    pipeline.set_layout(cfg, source, "stacked", fx["facecam"])
    rows = pipeline.analyze(conn, cfg, sid, transcribe=False)
    metrics = evaluate.evaluate(rows, evaluate.load_truth(fx["truth"]), k=3)
    print(f"\nsource #{sid}: expected highlights at {', '.join(media.fmt_ts(b) for b in fx['bursts'])}")
    _print_candidates(db.list_candidates(conn, sid), [], 5)
    print(f"\ndetection on the demo: precision@3={metrics['precision']} recall={metrics['recall']}")
    top = db.list_candidates(conn, sid)[: args.top]
    results = pipeline.render_candidates(conn, cfg, sid, [c["id"] for c in top], args.platforms.split(","))
    print("\nrendered:")
    for r in results:
        print(f"  {r['path']}")
    print("\nnext: `clipper review` to browse them, or ingest a real VOD with `clipper ingest`.")
    return 0


def cmd_ingest(args) -> int:
    cfg = config.load_config(streamer=args.streamer)
    conn = db.connect(cfg)
    sid = pipeline.ingest(conn, cfg, args.video, rights=args.rights, streamer=args.streamer, title=args.title,
                          campaign=args.campaign, language=args.language, chat_file=args.chat,
                          clips_file=args.clips, transcript_file=args.transcript)
    src = db.get_source(conn, sid)
    print(f"source #{sid}: {src['title']} ({media.fmt_ts(src['duration'])}, {src['width']}x{src['height']})")
    print(f"next: clipper analyze {sid}")
    return 0


def cmd_import_transcript(args) -> int:
    _, cfg, source = _open(args.source)
    pipeline.bind_transcript(Path(source["path"]), Path(args.file), pipeline.workdir(cfg, args.source), source["duration"])
    print(f"Transcript bound to recording #{args.source}")
    return 0


def cmd_sources(args) -> int:
    conn = db.connect(config.load_config())
    for s in db.list_sources(conn):
        print(f"#{s['id']:<4} {media.fmt_ts(s['duration'] or 0)}  {s['streamer'] or '-':<16} {s['title']}  "
              f"[rights: {s['rights']}]")
    return 0


def cmd_analyze(args) -> int:
    conn, cfg, source = _open(args.source)
    pipeline.analyze(conn, cfg, args.source, transcribe=not args.no_transcribe, force=args.force,
                     target=args.target)
    _print_candidates(db.list_candidates(conn, args.source), db.list_labels(conn, args.source), args.top)
    print(f"\nnext: clipper review   or   clipper render {args.source} --top 3")
    return 0


def cmd_candidates(args) -> int:
    conn, _, _ = _open(args.source)
    _print_candidates(db.list_candidates(conn, args.source), db.list_labels(conn, args.source), args.top)
    return 0


def cmd_layout(args) -> int:
    conn, cfg, source = _open(args.source)
    if args.set or args.kind:
        lay = pipeline.set_layout(cfg, source, args.kind, layout_mod.parse_box(args.set) if args.set else None)
    else:
        lay = pipeline.ensure_layout(cfg, source, force=args.detect)
    print(f"layout: {lay['kind']}  facecam: {lay.get('facecam')}")
    if args.preview:
        t = args.at if args.at is not None else float(source["duration"]) / 2
        png = layout_mod.preview(source["path"], t, (int(source["width"]), int(source["height"])), lay, cfg,
                                 Path(args.preview))
        print(f"preview: {png}  (red = facecam, green = gameplay crop)")
    return 0


def cmd_render(args) -> int:
    conn, cfg, source = _open(args.source)
    rows = db.list_candidates(conn, args.source)
    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
    else:
        labels = db.list_labels(conn, args.source)
        wanted = (1,) if args.good_only else (1, None)  # candidates labelled bad are never rendered
        pool = [r for r in rows if db.label_for(labels, r["start_s"], r["end_s"]) in wanted]
        ids = [r["id"] for r in pool[: args.top]]
    if not ids:
        print("nothing to render: run `clipper analyze` first, or label candidates as good")
        return 1
    results = pipeline.render_candidates(conn, cfg, args.source, ids, args.platforms.split(","), hook=args.hook,
                                         disclosure=args.disclosure, layout_kind=args.layout, replace=args.replace)
    for r in results:
        status = f"SKIPPED: {r['blocked_reason']}" if r["blocked_reason"] else f"ok (export #{r['export_id']})"
        print(f"#{r['candidate_id']:<4} {r['platform']:<10} {r['start_ts']} -> {r['end_ts']}  {status}")
        if r["path"]:
            print(f"      {r['path']}")
    return 0


def cmd_exports(args) -> int:
    conn = db.connect(config.load_config())
    rows = db.list_exports(conn, platform=args.platform, source_id=args.source)
    if not rows:
        print("no exports yet")
        return 0
    for e in rows:
        post = db.post_for(conn, e["id"])
        state = f"posted {post['url']}" if post else "not posted"
        print(f"#{e['id']:<5} {e['platform']:<10} source {e['source_id']} / candidate {e['candidate_id']}  "
              f"{media.fmt_ts(e['start_s'])} -> {media.fmt_ts(e['end_s'])}  {state}")
        print(f"       {e['path']}")
    return 0


def cmd_campaign_check(args) -> int:
    conn = db.connect(config.load_config())
    row = conn.execute("SELECT * FROM exports WHERE id=?", (args.export,)).fetchone()
    if row is None:
        raise ValueError(f"unknown export #{args.export}")
    source = db.get_source(conn, row["source_id"])
    path = Path(row["path"])
    meta = tr.load(path.with_suffix(".json")) or {}
    evidence = tr.load(args.evidence) if args.evidence else {}
    if not isinstance(meta, dict) or not isinstance(evidence, dict):
        raise ValueError("metadata and evidence must be JSON objects")
    receipt = meta.get("validation") or {}
    technical = (receipt.get("technically_valid") is True and path.is_file()
                 and meta.get("artifact_sha256") == media.sha256_file(path)
                 and meta.get("source_id") == source["id"] and meta.get("candidate_id") == row["candidate_id"]
                 and meta.get("platform") == row["platform"] and meta.get("start") == row["start_s"]
                 and meta.get("end") == row["end_s"])
    policy = campaign.policy_for(pipeline.source_config(source), source.get("campaign"))
    original = campaign.check(meta, policy, technical=technical)
    technical = technical and receipt.get("export_key") == original["export_key"]
    if source.get("campaign") != meta.get("campaign") or source["rights"] != meta.get("rights"):
        meta["rights"] = None
    caption = path.with_suffix(".txt")
    meta["caption"] = caption.read_text(encoding="utf-8-sig").strip() if caption.is_file() else ""
    report = campaign.check(meta, policy, technical=technical, evidence=evidence)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_submission"] else 1


def cmd_posted(args) -> int:
    conn = db.connect(config.load_config())
    db.add_post(conn, args.export, args.url, account=args.account)
    print(f"export #{args.export} marked as posted: {args.url}")
    return 0


def cmd_label(args) -> int:
    conn, _, _ = _open(args.source)
    if args.truth:
        windows = evaluate.load_truth(args.truth)
        for s, e in windows:
            db.add_label(conn, args.source, s, e, 1, note="truth file")
        print(f"{len(windows)} highlight windows stored as good")
        return 0
    if args.candidate is None or args.good == args.bad:
        print("use --truth FILE, or --candidate ID with --good or --bad")
        return 1
    cand = db.get_candidate(conn, args.candidate)
    if cand["source_id"] != args.source:
        raise ValueError(f"candidate #{args.candidate} belongs to source #{cand['source_id']}")
    db.add_label(conn, args.source, cand["start_s"], cand["end_s"], 1 if args.good else 0)
    print(f"candidate #{args.candidate} labelled {'good' if args.good else 'bad'}")
    return 0


def _truth(conn, source_id: int, path: str | None) -> list[tuple[float, float]]:
    return evaluate.load_truth(path) if path else evaluate.truth_from_labels(db.list_labels(conn, source_id))


def cmd_eval(args) -> int:
    conn, _, _ = _open(args.source)
    truth = _truth(conn, args.source, args.truth)
    if not truth:
        print("no ground truth: pass --truth FILE or label candidates as good in `clipper review`")
        return 1
    rows = _cand_dicts(db.list_candidates(conn, args.source))
    m = evaluate.evaluate(rows, truth, args.k)
    print(f"precision@{m['k']} = {m['precision']}   recall = {m['recall']}   f1 = {m['f1']}   "
          f"({m['truth']} marked highlights)")
    found = {h for h in m["hits"] if h is not None}
    for i, (s, e) in enumerate(truth):
        if i not in found:
            print(f"  missed: {media.fmt_ts(s)} -> {media.fmt_ts(e)}")
    for r, h in zip(rows[: args.k], m["hits"]):
        if h is None:
            print(f"  not a highlight: #{r['id']} {media.fmt_ts(r['start_s'])} -> {media.fmt_ts(r['end_s'])}")
    return 0


def cmd_tune(args) -> int:
    conn, cfg, source = _open(args.source)
    truth = _truth(conn, args.source, args.truth)
    if not truth:
        print("no ground truth: pass --truth FILE or label candidates as good in `clipper review`")
        return 1
    sig, tdata = pipeline.compute_signals(cfg, source, transcribe=False)
    res = evaluate.tune(sig, cfg, float(source["duration"]), tdata, truth, args.k, args.trials, args.seed)
    b, m = res["baseline"], res["metrics"]
    print(f"signals available: {', '.join(res['signals'])}")
    print(f"before: precision@{b['k']}={b['precision']} recall={b['recall']} f1={b['f1']}")
    print(f"after:  precision@{m['k']}={m['precision']} recall={m['recall']} f1={m['f1']}")
    print("weights: " + ", ".join(f"{k}={v}" for k, v in res["weights"].items()))
    if args.write:
        name = source.get("streamer") or "default"
        path = evaluate.write_tuned(config.tuned_file(name), res["weights"], m, f"source #{args.source}")
        print(f"saved to {path}; run `clipper analyze {args.source}` to use them")
    else:
        print("add --write to save these weights")
    return 0


def cmd_review(args) -> int:
    app = Path(__file__).with_name("review_app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port)]
    if importlib.util.find_spec("streamlit") is None:
        print('streamlit is not installed: pip install -e ".[review]"')
        return 1
    return subprocess.call(cmd)


# --- parser ------------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clipper", description="GTA AI Clipper: find highlights, render vertical clips.")
    p.add_argument("--version", action="version", version=f"clipper {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check FFmpeg, encoders, OpenCV, Whisper, fonts").set_defaults(func=cmd_doctor)

    s = sub.add_parser("demo", help="end-to-end run on a generated 3-minute stream")
    s.add_argument("--top", type=int, default=2)
    s.add_argument("--platforms", default=PLATFORMS_DEFAULT)
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("ingest", help="register a video you have the rights to clip")
    s.add_argument("video")
    s.add_argument("--rights", required=True,
                   help='where permission comes from, e.g. "content-rewards:<campaign>" or "streamer:<name>:email"')
    s.add_argument("--streamer", help="name of config/streamers/<name>.yaml to apply")
    s.add_argument("--title")
    s.add_argument("--campaign")
    s.add_argument("--language", help="transcript language, e.g. fr or en")
    s.add_argument("--chat", help="chat log (TwitchDownloader JSON, JSON list or CSV)")
    s.add_argument("--clips", help="Twitch 'Get Clips' JSON with vod_offset")
    s.add_argument("--transcript", help="existing transcript JSON (skips Whisper)")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("import-transcript", help="explicitly bind a verified transcript to the current recording")
    s.add_argument("source", type=int)
    s.add_argument("file")
    s.set_defaults(func=cmd_import_transcript)

    sub.add_parser("sources", help="list ingested videos").set_defaults(func=cmd_sources)

    s = sub.add_parser("analyze", help="compute signals and highlight candidates")
    s.add_argument("source", type=int)
    s.add_argument("--no-transcribe", action="store_true", help="skip Whisper (uses a cached transcript if any)")
    s.add_argument("--force", action="store_true", help="recompute generated transcripts, loudness and letterbox scans")
    s.add_argument("--target", default="campaign", help="clip length target from config (campaign, story)")
    s.add_argument("--top", type=int, default=15)
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser("candidates", help="list candidates")
    s.add_argument("source", type=int)
    s.add_argument("--top", type=int, default=30)
    s.set_defaults(func=cmd_candidates)

    s = sub.add_parser("layout", help="detect, set or preview the facecam layout")
    s.add_argument("source", type=int)
    s.add_argument("--detect", action="store_true", help="run facecam detection again")
    s.add_argument("--set", help="facecam box x,y,w,h as fractions of the frame")
    s.add_argument("--kind", choices=["stacked", "blur_fit", "center_crop"])
    s.add_argument("--preview", help="write a PNG with the boxes drawn")
    s.add_argument("--at", type=float, help="time (s) of the preview frame")
    s.set_defaults(func=cmd_layout)

    s = sub.add_parser("render", help="render vertical clips (one version per platform)")
    s.add_argument("source", type=int)
    s.add_argument("--ids", help="candidate ids, comma separated")
    s.add_argument("--top", type=int, default=3)
    s.add_argument("--good-only", action="store_true", help="only candidates labelled good")
    s.add_argument("--replace", action="store_true",
                   help="re-render exports of these candidates that were not posted yet (e.g. after a new hook)")
    s.add_argument("--platforms", default=PLATFORMS_DEFAULT)
    s.add_argument("--hook", help="hook text shown at the start (default: auto from speech)")
    s.add_argument("--disclosure", action=argparse.BooleanOptionalAction, default=None,
                   help="show the paid-content tag (default from config)")
    s.add_argument("--layout", choices=["stacked", "blur_fit", "center_crop"])
    s.set_defaults(func=cmd_render)

    s = sub.add_parser("exports", help="list rendered clips and whether they were posted")
    s.add_argument("--source", type=int)
    s.add_argument("--platform")
    s.set_defaults(func=cmd_exports)

    s = sub.add_parser("campaign-check", help="evaluate an export against current campaign terms and evidence")
    s.add_argument("export", type=int)
    s.add_argument("--evidence", help="local JSON file containing export-specific review and outcome evidence")
    s.set_defaults(func=cmd_campaign_check)

    s = sub.add_parser("posted", help="record that an export was posted (it will never be replaced)")
    s.add_argument("export", type=int, help="export id, see `clipper exports`")
    s.add_argument("--url", required=True, help="link to the post")
    s.add_argument("--account", help="which account posted it")
    s.set_defaults(func=cmd_posted)

    s = sub.add_parser("label", help="store ground truth: a CSV of highlights, or one candidate good/bad")
    s.add_argument("source", type=int)
    s.add_argument("--truth", help="CSV with start,end (seconds or HH:MM:SS)")
    s.add_argument("--candidate", type=int)
    s.add_argument("--good", action="store_true")
    s.add_argument("--bad", action="store_true")
    s.set_defaults(func=cmd_label)

    s = sub.add_parser("eval", help="precision/recall of the current candidates against ground truth")
    s.add_argument("source", type=int)
    s.add_argument("--truth")
    s.add_argument("--k", type=int, default=10)
    s.set_defaults(func=cmd_eval)

    s = sub.add_parser("tune", help="search signal weights that best match your ground truth")
    s.add_argument("source", type=int)
    s.add_argument("--truth")
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--trials", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--write", action="store_true", help="save to config/tuned/<streamer>.yaml")
    s.set_defaults(func=cmd_tune)

    s = sub.add_parser("review", help="open the review app in the browser")
    s.add_argument("--port", type=int, default=8501)
    s.set_defaults(func=cmd_review)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if getattr(stream, "reconfigure", None) and stream.errors == "strict":
            stream.reconfigure(errors="backslashreplace")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (media.MediaError, KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
