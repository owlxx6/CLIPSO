"""Video Editor: FFmpeg filtergraphs for the 9:16 layouts, burned captions, loudness and encoding."""

from __future__ import annotations

import shutil
from pathlib import Path

from clipper import config, media
from clipper.config import FONTS_DIR


def _n(x: float) -> str:
    return f"{float(x):.5f}".rstrip("0").rstrip(".")


def pixel_format(rcfg: dict) -> str:
    return "nv12" if rcfg.get("encoder") == "h264_qsv" else "yuv420p"


def encoder_args(rcfg: dict) -> list[str]:
    enc = rcfg.get("encoder", "libx264")
    crf = int(rcfg.get("crf", 21))
    if enc == "h264_qsv":  # Intel Arc / Quick Sync
        return ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", str(crf + 2)]
    if enc == "h264_videotoolbox":  # macOS
        return ["-c:v", "h264_videotoolbox", "-b:v", "8M", "-maxrate", "12M", "-bufsize", "16M"]
    return ["-c:v", "libx264", "-preset", str(rcfg.get("preset", "veryfast")), "-crf", str(crf), "-profile:v", "high"]


def game_x(facecam: dict | None, cx: str, max_shift: float, margin: float = 0.005) -> str:
    """x of a gameplay crop: centred on gameplay_center_x, moved aside (max_shift of the width at most) so it
    stops before the facecam overlay instead of showing a slice of it."""
    base = f"(iw-ow)*{cx}"
    if not facecam or max_shift <= 0:
        return base
    x0 = float(facecam["x"]) - margin
    x1 = float(facecam["x"]) + float(facecam["w"]) + margin
    ms = _n(max_shift)
    if float(facecam["x"]) + float(facecam["w"]) / 2 >= 0.5:  # facecam on the right: end the crop before it
        return f"max({base}-iw*{ms},max(0,min({base},iw*{_n(x0)}-ow)))"
    return f"min({base}+iw*{ms},min(iw-ow,max({base},iw*{_n(x1)})))"  # on the left: start after it


def video_filter(kind: str, facecam: dict | None, cfg: dict, zoom: float = 1.0, out: str = "vbase") -> str:
    r = cfg["render"]
    w, h, fps = int(r["width"]), int(r["height"]), int(r["fps"])
    pix = pixel_format(r)
    cx = _n(config.get(cfg, "layout.gameplay_center_x", 0.5))
    cy = _n(config.get(cfg, "layout.gameplay_center_y", 0.5))
    z = _n(max(1.0, zoom))
    gx = game_x(facecam, cx, float(config.get(cfg, "layout.avoid_facecam_shift", 0.15)))
    if kind == "stacked":
        if not facecam:
            raise ValueError("stacked layout needs a facecam box")
        band = int(r["facecam_band_h"])
        gh = h - band
        fx, fy, fw, fh = (_n(facecam[k]) for k in ("x", "y", "w", "h"))
        return (
            f"[0:v]split=2[cam][game];"
            f"[cam]crop=iw*{fw}:ih*{fh}:iw*{fx}:ih*{fy},scale={w}:{band}:force_original_aspect_ratio=increase,"
            f"crop={w}:{band},setsar=1[top];"
            f"[game]crop=w='min(iw,ih/{z}*{w}/{gh})':h='ih/{z}':x='{gx}':y='(ih-oh)*{cy}',"
            f"scale={w}:{gh},setsar=1[bot];"
            f"[top][bot]vstack=inputs=2,fps={fps},format={pix}[{out}]"
        )
    if kind == "blur_fit":
        fg_w = int(round(w * max(1.0, zoom) / 2) * 2)
        return (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=10:1,"
            f"scale={w}:{h},eq=brightness=-0.12[bgb];"
            f"[fg]scale={fg_w}:-2,crop=w='min(iw,{w})':h=ih:x='(iw-ow)/2':y=0,setsar=1[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,fps={fps},format={pix}[{out}]"
        )
    if kind == "center_crop":
        return (
            f"[0:v]crop=w='min(iw,ih/{z}*{w}/{h})':h='ih/{z}':x='{gx}':y='(ih-oh)/2',"
            f"scale={w}:{h},setsar=1,fps={fps},format={pix}[{out}]"
        )
    raise ValueError(f"unknown layout '{kind}'")


def render_clip(
    src: str | Path,
    start: float,
    end: float,
    layout: dict,
    ass_text: str,
    out_path: str | Path,
    cfg: dict,
    zoom: float = 1.0,
    has_audio: bool = True,
) -> Path:
    """Cut [start, end] from src and render a finished vertical MP4."""
    r = cfg["render"]
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.parent / f".work_{out_path.stem}"
    shutil.rmtree(work, ignore_errors=True)
    (work / "fonts").mkdir(parents=True)
    for font in FONTS_DIR.iterdir():
        if font.suffix.lower() in (".ttf", ".otf"):
            shutil.copy2(font, work / "fonts" / font.name)
    (work / "clip.ass").write_text(ass_text, encoding="utf-8")
    # Relative names inside the filtergraph avoid Windows drive-letter escaping (C\:/...) entirely.
    graph = video_filter(layout["kind"], layout.get("facecam"), cfg, zoom) + \
        ";[vbase]subtitles=clip.ass:fontsdir=fonts[vout]"
    if has_audio:
        graph += f";[0:a]loudnorm=I={_n(r.get('loudness_lufs', -14))}:TP=-1.5:LRA=11,aresample=48000,aformat=channel_layouts=stereo[aout]"
    cmd = [media.ffmpeg_bin(), "-nostdin", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
           "-i", str(Path(src).resolve()), "-filter_complex", graph, "-map", "[vout]"]
    if has_audio:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", str(r.get("audio_bitrate", "160k"))]
    cmd += encoder_args(r) + ["-r", str(int(r["fps"])), "-movflags", "+faststart", str(out_path)]
    try:
        media.run(cmd, cwd=work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out_path


def validate_clip(path: Path, duration: float, settings: dict, has_audio: bool) -> None:
    info = media.probe(path)
    tolerance = max(0.2, 2 / settings["fps"])
    lengths = [info["duration"], info["video_duration"]]
    if has_audio:
        lengths.append(info["audio_duration"])
    if ((info["width"], info["height"]) != (settings["width"], settings["height"])
            or abs(info["fps"] - settings["fps"]) > 0.1 or info["has_audio"] != has_audio
            or info["video_codec"] != "h264" or (has_audio and info["audio_codec"] != "aac")
            or any(value is None or abs(float(value) - duration) > tolerance for value in lengths)):
        raise media.MediaError(f"Render failed stream or duration checks: {info}")
    media.run([media.ffmpeg_bin(), "-nostdin", "-v", "error", "-xerror", "-i", str(path),
               "-map", "0:v:0", "-map", "0:a?", "-f", "null", "-"])


def make_cover(video: str | Path, t: float, out_jpg: str | Path) -> Path:
    out_jpg = Path(out_jpg)
    media.run([media.ffmpeg_bin(), "-nostdin", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(Path(video).resolve()),
               "-frames:v", "1", "-q:v", "3", "-update", "1", out_jpg.name], cwd=out_jpg.parent)
    return out_jpg


def make_proxy(src: str | Path, start: float, end: float, out_path: str | Path, height: int = 480) -> Path:
    """Small, fast preview of a candidate window for the review app."""
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    media.run([media.ffmpeg_bin(), "-nostdin", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
               "-i", str(src), "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
               "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out_path)])
    return out_path
