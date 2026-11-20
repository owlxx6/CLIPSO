"""Orchestration shared by the CLI and the review app: ingest, analyze, layout, render."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from clipper import campaign, config, db, dedup, detect, media, render, subtitles, variants
from clipper import layout as layout_mod
from clipper.signals import audio, chat, viewer_clips, visual
from clipper.signals import transcript as tr

RAW_SIGNALS_VERSION = 2


def source_config(source: dict | None, extra: str | Path | None = None) -> dict:
    return config.load_config(streamer=(source or {}).get("streamer"), extra=extra)


def workdir(cfg: dict, source_id: int) -> Path:
    path = config.data_dir(cfg) / "sources" / str(source_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_dir(cfg: dict, source_id: int, candidate_id: int) -> Path:
    path = config.data_dir(cfg) / "exports" / f"source_{source_id}" / f"candidate_{candidate_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- ingest ------------------------------------------------------------------------------------------------------

def ingest(conn, cfg: dict, video: str | Path, rights: str, streamer: str | None = None, title: str | None = None,
           campaign: str | None = None, language: str | None = None, chat_file: str | Path | None = None,
           clips_file: str | Path | None = None, transcript_file: str | Path | None = None) -> int:
    video = Path(video).resolve()
    if not video.exists():
        raise FileNotFoundError(video)
    info = media.probe(video)
    if not info["has_video"]:
        raise ValueError(f"{video} has no video stream")
    for extra in (chat_file, clips_file, transcript_file):  # before the source exists, not half way through
        if extra and not Path(extra).is_file():
            raise FileNotFoundError(f"no such file: {extra}")
    root = config.data_dir(cfg) / "sources"
    root.mkdir(exist_ok=True)
    published = None
    with tempfile.TemporaryDirectory(dir=root) as folder:
        staged = Path(folder) / "source"
        staged.mkdir()
        if chat_file:
            shutil.copy2(chat_file, staged / f"chat{Path(chat_file).suffix.lower()}")
        if clips_file:
            shutil.copy2(clips_file, staged / "viewer_clips.json")
        if transcript_file:
            bind_transcript(video, transcript_file, staged, info["duration"])
        try:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                source_id = db.add_source(
                    conn, commit=False, path=str(video), title=title or video.stem, streamer=streamer,
                    language=language, rights=rights, campaign=campaign, duration=info["duration"],
                    width=info["width"], height=info["height"], fps=info["fps"], has_audio=int(info["has_audio"]),
                )
                target = root / str(source_id)
                if target.exists():
                    raise ValueError(f"source directory already exists: {target}")
                staged.rename(target)
                published = target
        except BaseException:
            if published is not None:
                shutil.rmtree(published)
            raise
    return source_id


def _chat_file(wd: Path) -> Path | None:
    for ext in (".json", ".csv"):
        if (wd / f"chat{ext}").exists():
            return wd / f"chat{ext}"
    return None


# --- signals -----------------------------------------------------------------------------------------------------

def _source_key(source: dict) -> dict:
    path = Path(source["path"])
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def raw_signals(cfg: dict, source: dict, force: bool = False, log=print) -> dict[str, np.ndarray]:
    """Expensive passes over the video (loudness, letterbox), cached per source and hop size."""
    wd = workdir(cfg, source["id"])
    hop = float(config.get(cfg, "analysis.hop_s", 0.5))
    cache = wd / "raw_signals.npz"
    key = json.dumps({"version": RAW_SIGNALS_VERSION, "source": _source_key(source), "hop": hop,
                      "sample_rate": cfg["analysis"]["audio"]["sample_rate"], "visual": cfg["analysis"]["visual"]},
                     sort_keys=True)
    if not force and cache.exists():
        with np.load(cache) as data:
            if "_key" in data and str(data["_key"]) == key:
                return {k: data[k] for k in ("_db", "_cinematic")}
    log("measuring loudness ...")
    db_curve = (audio.loudness_db(source["path"], hop, int(config.get(cfg, "analysis.audio.sample_rate", 16000)))
                if source["has_audio"] else np.full(int(np.ceil(source["duration"] / hop)), -90, dtype=np.float32))
    log("scanning for cinematics (letterbox) ...")
    cine = visual.letterbox_curve(source["path"], source["duration"], hop, cfg["analysis"]["visual"])
    with tempfile.TemporaryDirectory(dir=wd) as folder:
        staged = Path(folder) / cache.name
        np.savez_compressed(staged, _key=key, _db=db_curve, _cinematic=cine)
        staged.replace(cache)
    return {"_db": db_curve, "_cinematic": cine}


def bind_transcript(video: Path, transcript_file: Path, wd: Path, duration: float) -> None:
    """Explicit operator assignment. Publish transcript and recording identity together."""
    data = tr.load(transcript_file)
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError("transcript must contain a segments array")
    for segment in data["segments"]:
        if not isinstance(segment, dict) or not isinstance(segment.get("text", ""), str):
            raise ValueError("invalid transcript segment")
        words = segment.get("words", [])
        if not isinstance(words, list):
            raise ValueError("invalid transcript words")
        for item in [segment, *words]:
            if (not isinstance(item, dict) or not all(type(item.get(k)) in (int, float) for k in ("start", "end"))
                    or not 0 <= item["start"] <= item["end"] <= duration + 0.1):
                raise ValueError("transcript timing outside recording")
        if any(not isinstance(word.get("word"), str) for word in words):
            raise ValueError("invalid transcript word")
    receipt = {"version": 1, "source_sha256": media.sha256_file(video),
               "imported_at": dt.datetime.now(dt.timezone.utc).isoformat(), "transcript": data}
    with tempfile.TemporaryDirectory(dir=wd) as folder:
        temp = Path(folder) / "transcript.bound.json"
        tr.save(temp, receipt)
        temp.replace(wd / temp.name)


def get_transcript(cfg: dict, source: dict, transcribe: bool = True, log=print, force: bool = False,
                   *, source_digest: str | None = None) -> dict | None:
    """The supplied transcript if any, else the generated one.

    With transcribe=False (render, `analyze --no-transcribe`, tune) a generated transcript of the same source file
    is used even if the Whisper settings changed since: it is the one the candidates were analysed with.
    """
    wd = workdir(cfg, source["id"])
    bound = tr.load(wd / "transcript.bound.json")
    if bound is not None:
        if bound.get("version") == 1 and bound.get("source_sha256") == (source_digest or media.sha256_file(source["path"])):
            return bound["transcript"]
        log("recording fingerprint changed; ignoring its previous imported transcript")
    elif any((wd / name).exists() for name in ("transcript.imported.json", "transcript.json")):
        log(f"legacy transcript has no recording identity; verify it and run clipper import-transcript {source['id']} FILE")
    path = wd / "transcript.generated.json"
    tcfg = dict(cfg["analysis"]["transcript"])
    if source.get("language"):
        tcfg["language"] = source["language"]
    key = {"source": _source_key(source), "settings": tcfg}
    cached = tr.load(path)
    can_transcribe = transcribe and tcfg.get("enabled", True)
    if cached and cached["key"] == key and not force:
        return cached["transcript"]
    if can_transcribe:
        data = tr.transcribe(source["path"], tcfg, log)
        if data is not None:
            with tempfile.TemporaryDirectory(dir=wd) as folder:
                staged = Path(folder) / path.name
                tr.save(staged, {"key": key, "transcript": data})
                staged.replace(path)
            return data
    if not can_transcribe and cached and cached["key"]["source"] == key["source"]:
        if cached["key"] != key:
            log("using the transcript made with earlier Whisper settings; "
                f"run `clipper analyze {source['id']}` to transcribe again with the current ones")
        return cached["transcript"]
    return None


def compute_signals(cfg: dict, source: dict, transcribe: bool = True, force: bool = False,
                    log=print) -> tuple[dict[str, np.ndarray], dict | None]:
    """All curves for one source. Cheap curves are rebuilt from cached inputs so config changes apply at once."""
    hop = float(config.get(cfg, "analysis.hop_s", 0.5))
    duration = float(source["duration"])
    wd = workdir(cfg, source["id"])
    sig = dict(raw_signals(cfg, source, force, log))
    sig["audio"] = audio.excitement(sig["_db"], hop, float(config.get(cfg, "analysis.audio.baseline_window_s", 60)))
    tdata = get_transcript(cfg, source, transcribe, log, force)
    if tdata:
        sig["keywords"] = tr.keyword_curve(tdata, cfg["signals"]["keywords"], duration, hop, source.get("language"))
        sig["speech_rate"] = tr.speech_rate_curve(tdata, duration, hop,
                                                  float(config.get(cfg, "signals.speech_rate.window_s", 3.0)))
    chat_path = _chat_file(wd)
    if chat_path:
        sig["chat"] = chat.chat_curve(chat.load_chat(chat_path), duration, hop, cfg["signals"]["chat"])
    if (wd / "viewer_clips.json").exists():
        clips = viewer_clips.load_clips(wd / "viewer_clips.json")
        sig["viewer_clips"] = viewer_clips.clips_curve(clips, duration, hop, cfg["signals"]["viewer_clips"])
    return sig, tdata


def analyze(conn, cfg: dict, source_id: int, transcribe: bool = True, force: bool = False,
            target: str = "campaign", log=print) -> list[dict]:
    source = db.get_source(conn, source_id)
    sig, tdata = compute_signals(cfg, source, transcribe, force, log)
    cands, score = detect.detect(sig, cfg, float(source["duration"]), tdata, target)
    rows = []
    for cand in cands:
        row = cand.as_dict()
        if config.get(cfg, "render.hook.auto", True):
            row["hook"] = tr.peak_phrase(tdata, cand.peak, int(config.get(cfg, "render.hook.max_words", 7))) or None
        rows.append(row)
    db.replace_candidates(conn, source_id, rows)
    np.save(workdir(cfg, source_id) / "score.npy", score)
    log(f"{len(rows)} candidates from signals: {', '.join(k for k in detect.SIGNAL_NAMES if k in sig)}")
    return rows


# --- layout ------------------------------------------------------------------------------------------------------

def ensure_layout(cfg: dict, source: dict, force: bool = False, log=print) -> dict:
    wd = workdir(cfg, source["id"])
    if not force and (layout_mod.load_layout(wd) is not None or config.get(cfg, "streamer.facecam")):
        return layout_mod.resolve(cfg, wd)
    log("looking for the facecam ...")
    found = layout_mod.detect_facecam(source["path"], float(source["duration"]),
                                      (int(source["width"]), int(source["height"])), cfg["layout"])
    if found:
        saved = {"kind": "stacked", "facecam": found["facecam"], "confidence": found["confidence"], "auto": True}
        log(f"facecam found at {found['facecam']} (seen in {int(found['confidence'] * 100)}% of samples)")
    else:
        saved = {"kind": "blur_fit", "facecam": None, "auto": True}
        log("no stable facecam found: using the blurred-background layout")
    layout_mod.save_layout(wd, saved)
    return layout_mod.resolve(cfg, wd)


def set_layout(cfg: dict, source: dict, kind: str | None = None, box: dict | None = None) -> dict:
    wd = workdir(cfg, source["id"])
    current = layout_mod.load_layout(wd) or {}
    facecam = box if box is not None else current.get("facecam")
    saved = {"kind": kind or ("stacked" if facecam else "blur_fit"), "facecam": facecam, "auto": False}
    layout_mod.save_layout(wd, saved)
    return layout_mod.resolve(cfg, wd)


# --- render ------------------------------------------------------------------------------------------------------

def make_caption(cfg: dict, source: dict, hook: str | None, text: str) -> str:
    template = str(config.get(cfg, "captions.template", "{hook}"))
    hook = (hook or " ".join(text.split()[:10])).strip()
    name = config.get(cfg, "streamer.display_name") or source.get("streamer") or ""
    caption = template.format(hook=hook, streamer=name, title=source.get("title") or "")
    return " ".join(caption.replace("|  ", "").split()).strip(" |")


def _remove_export_files(path: str, log=print) -> None:
    """Best effort: the new export is already live, so a file that cannot go (open in a player on Windows) stays."""
    p = Path(path)
    for f in (p, p.with_suffix(".jpg"), p.with_suffix(".json"), p.with_suffix(".txt")):
        try:
            f.unlink(missing_ok=True)
        except OSError as exc:
            log(f"could not delete the replaced file {f} ({exc.strerror or exc}); delete it once it is closed")


def _export_snapshot(conn, platform: str) -> list[dict]:
    rows = db.list_exports(conn, platform=platform)
    posted = {row[0] for row in conn.execute("SELECT export_id FROM posts WHERE platform=?", (platform,))}
    for row in rows:
        row["posted"] = row["id"] in posted
    return rows


def _publish(conn, staged: Path, directory: Path, previous: list[dict], replaced: list[dict],
             fields: dict, source: dict, log=print) -> tuple[int, Path]:
    moved = []
    try:
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            current = db.get_source(conn, source["id"])
            if any(current[key] != source[key] for key in ("rights", "campaign")):
                raise RuntimeError("Source permissions changed during rendering. Review the source and retry.")
            if _export_snapshot(conn, fields["platform"]) != previous:
                raise RuntimeError("Exports changed during rendering. Retry with current export records.")
            n = 1
            while any(directory.glob(f"{fields['platform']}_v{n}.*")):
                n += 1
            out = directory / f"{fields['platform']}_v{n}.mp4"
            for ext in (".jpg", ".txt", ".json", ".mp4"):
                dest = out.with_suffix(ext)
                staged.with_suffix(ext).rename(dest)
                moved.append(dest)
            conn.executemany("DELETE FROM exports WHERE id=?", [(r["id"],) for r in replaced])
            eid = db.add_export(conn, commit=False, path=str(out), **fields)
    except BaseException:
        for path in moved:
            path.unlink(missing_ok=True)
        raise
    for row in replaced:
        if Path(row["path"]).resolve().parent == directory.resolve():
            _remove_export_files(row["path"], log)
    return eid, out


def render_candidates(conn, cfg: dict, source_id: int, candidate_ids: list[int], platforms: list[str],
                      hook: str | None = None, disclosure: bool | None = None, layout_kind: str | None = None,
                      replace: bool = False, log=print) -> list[dict]:
    """Render one version per platform for each candidate.

    A moment already exported on a platform is skipped before rendering: same source with an overlapping time
    range, or the same footage found in another copy of the video (perceptual signature of the gameplay).
    Pass replace=True to re-render an export that was not posted yet, e.g. after changing the hook.
    """
    source = db.get_source(conn, source_id)
    if not (source.get("rights") or "").strip():
        raise ValueError("A rights note is required before rendering")
    unknown = [p for p in platforms if p not in cfg["platforms"]]
    if unknown:
        raise ValueError(f"unknown platform(s): {unknown}; configured: {list(cfg['platforms'])}")
    before = _source_key(source)
    source_digest = media.sha256_file(source["path"])
    policy = campaign.policy_for(cfg, source.get("campaign"))
    transcript = get_transcript(cfg, source, transcribe=False, log=log, source_digest=source_digest)
    all_words = tr.words(transcript)
    cands = [db.get_candidate(conn, cid) for cid in candidate_ids]
    for cand in cands:  # checked before anything is rendered
        if cand["source_id"] != source_id:
            raise ValueError(f"candidate #{cand['id']} belongs to source #{cand['source_id']}")
        if (config.get(cfg, "render.subtitles.enabled", True) and (cand.get("text") or "").strip()
                and not tr.text_between(transcript, cand["start_s"], cand["end_s"])):
            raise RuntimeError(f"candidate #{cand['id']} has speech but its transcript is missing or was made from "
                               f"another file, so the clip would have no captions: run clipper analyze {source_id}")
    wd = workdir(cfg, source_id)
    lay = ensure_layout(cfg, source, log=log)
    if layout_kind:
        lay = layout_mod.resolve(cfg, wd, layout_kind)
    use_disclosure = disclosure if disclosure is not None else bool(config.get(cfg, "render.disclosure.enabled"))
    configured_label = config.get(cfg, "render.disclosure.text")
    if policy.get("terms_verified") is True and isinstance(policy.get("disclosure"), str):
        configured_label = policy["disclosure"]
    label = subtitles.clean(configured_label) if isinstance(configured_label, str) else ""
    label = label or "PUBLICITÉ"
    disclosure_text = label if use_disclosure else None
    paid_rights = source["rights"].strip().casefold().startswith(("campaign:", "content-rewards:"))
    if paid_rights or (source.get("campaign") or "").strip():
        disclosure_text = label  # paid campaign clips are always labelled
    dcfg = cfg["dedup"]
    results = []
    for cand in cands:
        cid = cand["id"]
        if (cand.get("flags") or {}).get("spoiler_review"):
            log(f"candidate #{cid}: looks like a cutscene, check it for story spoilers before posting")
        used = {p: [(e["start_s"], e["end_s"]) for e in db.list_exports(conn, platform=p, source_id=source_id)]
                for p in platforms}
        plans = variants.plan_variants(cid, cand["start_s"], cand["end_s"], cand["peak_s"], float(source["duration"]),
                                       platforms, cfg, used, transcript=transcript,
                                       cues=(cand.get("flags") or {}).get("boundary_cues", []),
                                       duration_limits=campaign.duration_limits(policy))
        hook_text = hook or cand.get("hook") or None
        out_dir = export_dir(cfg, source_id, cid)
        step, size = float(dcfg.get("frame_step_s", 1.0)), int(dcfg.get("hash_size", 16))
        t0 = min((v.start for v in plans), default=0.0)
        full_sig: list[str] | None = None  # one decode of the source per candidate, shared by its platforms
        for v in plans:
            snapshot = _export_snapshot(conn, v.platform)
            replaced = [e for e in snapshot if replace and e["candidate_id"] == cid and not e["posted"]]
            previous = [e for e in snapshot if e not in replaced]
            reason = None
            already = dedup.overlapping_export(source_id, v.start, v.end, previous,
                                               float(dcfg.get("min_overlap", 0.5)))
            posted_here = next((e for e in previous if e["candidate_id"] == cid and e["posted"]), None)
            if posted_here:
                reason = f"already exported on {v.platform} as posted export #{posted_here['id']}"
            elif already:
                reason = f"already exported on {v.platform} as export #{already[0]} (same moment)"
            else:
                if full_sig is None:
                    full_sig = dedup.signature(source["path"], step, size, start=t0,
                                               duration=max(x.end for x in plans) - t0, facecam=lay.get("facecam"))
                sig = dedup.window(full_sig, t0, v.start, v.end, step)
                dup = dedup.find_duplicate(sig, previous, dcfg)
                if dup:
                    reason = (f"looks like export #{dup[0]} on {v.platform} (similarity {dup[1]}): "
                              "the same footage was already exported from another copy of the video")
            if reason:
                log(f"skipping #{cid} on {v.platform}: {reason}")
                results.append({"candidate_id": cid, "platform": v.platform, "start": v.start, "end": v.end,
                                "start_ts": media.fmt_ts(v.start), "end_ts": media.fmt_ts(v.end),
                                "blocked_reason": reason, "path": None, "export_id": None})
                continue
            ass = subtitles.build_ass(subtitles.clip_words(all_words, v.start, v.end), cfg, lay["kind"],
                                      v.highlight_color, hook_text, disclosure_text, v.end - v.start)
            log(f"rendering #{cid} for {v.platform}: {media.fmt_ts(v.start)} -> {media.fmt_ts(v.end)} "
                f"({v.end - v.start:.1f}s, {lay['kind']})")
            with tempfile.TemporaryDirectory(dir=out_dir, prefix=".pending-") as folder:
                out = Path(folder) / "clip.mp4"
                render.render_clip(source["path"], v.start, v.end, lay, ass, out, cfg, zoom=v.zoom,
                                   has_audio=bool(source["has_audio"]))
                measured = render.validate_clip(out, v.end - v.start, cfg["render"], bool(source["has_audio"]))
                render.make_cover(out, v.cover_at, out.with_suffix(".jpg"))
                if not out.with_suffix(".jpg").stat().st_size:
                    raise media.MediaError("Empty cover image")
                caption = make_caption(cfg, source, hook_text, cand.get("text") or "")
                meta = {
                    "source_id": source_id, "source_path": source["path"], "rights": source["rights"],
                    "campaign": source.get("campaign"), "candidate_id": cid, "platform": v.platform,
                    "start": v.start, "end": v.end, "start_ts": media.fmt_ts(v.start), "end_ts": media.fmt_ts(v.end),
                    "layout": lay["kind"], "zoom": v.zoom, "hook": hook_text, "caption": caption,
                    "disclosure": disclosure_text, "spoiler_review": bool((cand.get("flags") or {}).get("spoiler_review")),
                    "blocked_reason": None,
                }
                if _source_key(source) != before:
                    raise RuntimeError("Recording changed during render; reanalyze the source")
                meta.update(source_sha256=source_digest, artifact_sha256=media.sha256_file(out),
                            duration_s=measured["duration"])
                meta["validation"] = campaign.check(meta, policy, technical=True)
                out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                out.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
                fields = dict(candidate_id=cid, source_id=source_id, platform=v.platform, start_s=v.start,
                              end_s=v.end, hook=hook_text, caption=caption, signature=sig)
                eid, out = _publish(conn, out, out_dir, snapshot, replaced, fields, source, log)
            results.append(dict(meta, export_id=eid, path=str(out)))
    return results


def preview_proxy(cfg: dict, source: dict, cand: dict) -> Path:
    height = int(config.get(cfg, "render.proxy_height", 480))
    # the name is the cache key: same window of the same file at the same size
    key = hashlib.sha1(json.dumps([_source_key(source), height], sort_keys=True).encode("utf-8")).hexdigest()[:12]
    out = config.data_dir(cfg) / "proxies" / f"source_{source['id']}" / \
        f"{cand['start_s']:.2f}_{cand['end_s']:.2f}_{key}.mp4"
    return render.make_proxy(source["path"], cand["start_s"], cand["end_s"], out, height)
