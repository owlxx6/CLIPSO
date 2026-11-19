"""Thin FFmpeg/FFprobe helpers. Every call passes an argument list (no shell), so paths work on Windows and macOS."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np


class MediaError(RuntimeError):
    pass


# Homebrew's plain `ffmpeg` formula is built without libass (no `subtitles` filter, so no captions);
# `ffmpeg-full` has it but is keg-only (not on PATH), so look for it first.
BREW_FULL_DIRS = ("/opt/homebrew/opt/ffmpeg-full/bin", "/usr/local/opt/ffmpeg-full/bin")


def _bin(name: str) -> str:
    folders = [os.environ["CLIPPER_FFMPEG_DIR"]] if os.environ.get("CLIPPER_FFMPEG_DIR") else []
    folders += list(BREW_FULL_DIRS)
    for folder in folders:
        exe = shutil.which(name, path=folder)
        if exe:
            return exe
    exe = shutil.which(name)
    if not exe:
        raise MediaError(
            f"{name} not found. Windows: `winget install Gyan.FFmpeg`; macOS: `brew install ffmpeg-full`; "
            "then open a new terminal (or set CLIPPER_FFMPEG_DIR to the folder that holds ffmpeg and ffprobe)."
        )
    return exe


def ffmpeg_bin() -> str:
    return _bin("ffmpeg")


def ffprobe_bin() -> str:
    return _bin("ffprobe")


def run(cmd: list[str], cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        raise MediaError(f"command failed ({proc.returncode}): {' '.join(str(c) for c in cmd[:8])} ...\n{tail}")
    return proc


def probe(path: str | Path) -> dict:
    out = run([ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration") or (video or {}).get("duration") or 0.0)
    fps = 0.0
    rate = (video or {}).get("avg_frame_rate") or "0/0"
    try:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return {
        "duration": duration,
        "width": int(video["width"]) if video else 0,
        "height": int(video["height"]) if video else 0,
        "fps": round(fps, 3),
        "video_duration": (video or {}).get("duration"),
        "audio_duration": (audio or {}).get("duration"),
        "video_codec": (video or {}).get("codec_name"),
        "audio_codec": (audio or {}).get("codec_name"),
        "has_video": video is not None,
        "has_audio": audio is not None,
    }


def pcm_chunks(path: str | Path, sample_rate: int = 16000, chunk_bytes: int = 1 << 20) -> Iterator[bytes]:
    """Stream mono 16-bit PCM from any media file without writing a WAV to disk."""
    cmd = [ffmpeg_bin(), "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
           "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    try:
        while True:
            chunk = proc.stdout.read(chunk_bytes)
            if not chunk:
                break
            yield chunk
    finally:
        proc.stdout.close()
        proc.wait()


def gray_frames(
    path: str | Path, fps: float, width: int, height: int, keyframes_only: bool = False,
    start: float | None = None, duration: float | None = None, crop: str | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (time, HxW uint8) frames at a fixed rate. keyframes_only makes long VODs fast to scan.

    start/duration read only a window (times are then relative to start); crop is an ffmpeg crop filter.
    """
    cmd = [ffmpeg_bin(), "-nostdin", "-v", "error"]
    if keyframes_only:
        cmd += ["-skip_frame", "nokey"]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, start):.3f}"]
    if duration is not None:
        cmd += ["-t", f"{max(0.0, duration):.3f}"]
    chain = [f"fps={fps}"] + ([crop] if crop else []) + [f"scale={width}:{height}", "format=gray"]
    cmd += ["-i", str(path), "-an", "-vf", ",".join(chain), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    size = width * height
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    index = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield index / fps, np.frombuffer(buf, dtype=np.uint8).reshape(height, width)
            index += 1
    finally:
        proc.stdout.close()
        proc.wait()


def bgr_frame_at(path: str | Path, t: float, src_size: tuple[int, int], width: int = 640) -> np.ndarray | None:
    src_w, src_h = src_size
    height = max(2, int(round(width * src_h / max(1, src_w) / 2)) * 2)
    cmd = [ffmpeg_bin(), "-nostdin", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(path), "-frames:v", "1",
           "-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    need = width * height * 3
    if len(out) < need:
        return None
    return np.frombuffer(out[:need], dtype=np.uint8).reshape(height, width, 3).copy()


def fmt_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def parse_ts(value: str | float | int) -> float:
    """Accept 3723.5, '3723.5', '1:02:03', '62:03' or '01:02:03.5'."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if ":" not in text:
        return float(text)
    total = 0.0
    for part in text.split(":"):
        total = total * 60 + float(part)
    return total


def encoders_and_filters() -> tuple[set[str], set[str]]:
    exe = ffmpeg_bin()
    enc = run([exe, "-hide_banner", "-encoders"]).stdout
    flt = run([exe, "-hide_banner", "-filters"]).stdout

    def names(text: str) -> set[str]:
        found = set()
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and not line.startswith(" =") and len(parts[0]) <= 6:
                found.add(parts[1])
        return found

    return names(enc), names(flt)
