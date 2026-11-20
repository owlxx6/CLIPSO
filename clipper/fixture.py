"""Synthetic "stream" with known highlights, used by the tests and by `clipper demo`.

What it contains (default 180 s):
- gameplay: a camera panning over a large blurry texture (so distant moments look different and nearby
  moments look alike); facecam: a blue box in the top-right corner (box FACECAM below)
- fully deterministic: the same files on every run (fixed seeds), so test results never change between runs
- quiet pink noise, with loud bursts at BURSTS (the moments a detector must find)
- black letterbox bars between LETTERBOX[0] and LETTERBOX[1] (a fake cutscene)
- chat.json reacting after bursts 1 and 3, viewer_clips.json covering burst 2, transcript.json with hype words,
  truth.csv marking the three highlights
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np

from clipper import media

BURSTS = (40.0, 95.0, 150.0)
LETTERBOX = (120.0, 130.0)
FACECAM = {"x": 0.725, "y": 0.0333, "w": 0.25, "h": 0.3333}
FILLER = ["bon", "alors", "on", "va", "voir", "ce", "qu'il", "se", "passe", "ici", "tranquille", "la", "route"]
HYPE = {0: ["oh", "mon", "dieu", "non"], 1: ["c'est", "quoi", "ça", "mdr"], 2: ["je", "suis", "mort", "ptdr"]}


def make_texture(path: Path, size: tuple[int, int], seed: int = 7) -> Path:
    """A 3x3-frame-sized image of soft colour blobs (about a sixth of the frame wide), never dark."""
    import cv2

    w, h = size
    rng = np.random.default_rng(seed)
    grid = rng.integers(60, 200, size=(max(6, round(18 * h / w)), 18, 3), dtype=np.uint8)
    return media.write_image(path, cv2.resize(grid, (3 * w, 3 * h), interpolation=cv2.INTER_CUBIC))


def make_video(path: Path, duration: float = 180.0, size: tuple[int, int] = (1280, 720)) -> Path:
    w, h = size
    texture = make_texture(path.with_name(path.stem + "_texture.png"), size)
    cam_w, cam_h = w // 4, h // 3
    beeps = "+".join(f"between(t,{b},{b + 3})" for b in BURSTS)
    lb0, lb1 = LETTERBOX
    bars = (f"drawbox=x=0:y=0:w=iw:h=ih*0.11:color=black:t=fill:enable='between(t,{lb0},{lb1})',"
            f"drawbox=x=0:y=ih*0.89:w=iw:h=ih*0.11:color=black:t=fill:enable='between(t,{lb0},{lb1})'")
    # pan over the texture on a Lissajous path: at most ~5% of the frame width per second
    pan = (f"[0:v]loop=loop=-1:size=1:start=0,setpts=N/30/TB,"
           f"crop=w={w}:h={h}:x='{w}*(1+sin(t*0.047))':y='{h}*(1+sin(t*0.031+1))'[g];")
    graph = (pan + f"[g][1:v]overlay=x=W-w-{w // 40}:y={h // 30}:shortest=1[v0];[v0]{bars}[v];"
             f"[2:a]volume='4*({beeps})':eval=frame[beep];[3:a][beep]amix=inputs=2:duration=first:normalize=0[a]")
    media.run([
        media.ffmpeg_bin(), "-nostdin", "-y", "-v", "error",
        "-i", str(texture),
        "-f", "lavfi", "-i", f"color=c=0x2a4d8f:size={cam_w}x{cam_h}:rate=30:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={duration}",
        "-f", "lavfi", "-i", f"anoisesrc=color=pink:amplitude=0.03:sample_rate=48000:duration={duration}:seed=7",
        "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
        "-t", f"{duration}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-g", "60", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", str(path),
    ])
    return path


def make_chat(path: Path, duration: float, seed: int = 7) -> Path:
    rng = random.Random(seed)
    comments = [{"content_offset_seconds": round(t, 2), "message": {"body": rng.choice(["gg", "lol", "nice", "ok"])}}
                for t in [i * 2.0 + rng.random() for i in range(int(duration / 2))]]
    for burst in (BURSTS[0], BURSTS[2]):
        for _ in range(30):
            t = burst + 1.0 + rng.random() * 5.0
            comments.append({"content_offset_seconds": round(t, 2),
                             "message": {"body": rng.choice(["KEKW", "OMEGALUL", "MDR", "W"])}})
    comments.sort(key=lambda c: c["content_offset_seconds"])
    path.write_text(json.dumps({"comments": comments}), encoding="utf-8")
    return path


def make_viewer_clips(path: Path) -> Path:
    data = {"data": [{"vod_offset": BURSTS[1] - 22.0, "duration": 30.0, "view_count": 850, "title": "fixture clip"}]}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def make_transcript(path: Path, duration: float) -> Path:
    segments, t, i = [], 1.0, 0
    hype_at = {int(b): k for k, b in enumerate(BURSTS)}
    while t < duration - 6:
        words, start = [], t
        k = next((hype_at[b] for b in hype_at if start <= b < start + 8.7), None)  # segment + gap
        tokens = HYPE[k] if k is not None else [FILLER[(i + j) % len(FILLER)] for j in range(6)]
        if k is not None:
            t = BURSTS[k] + 0.3
            start = t
        for tok in tokens:
            words.append({"start": round(t, 2), "end": round(t + 0.45, 2), "word": tok})
            t += 0.6 if k is not None else 1.2
        segments.append({"start": start, "end": round(t, 2), "text": " ".join(w["word"] for w in words),
                         "words": words})
        t += 1.5
        i += 6
    path.write_text(json.dumps({"language": "fr", "segments": segments}, ensure_ascii=False), encoding="utf-8")
    return path


def make_truth(path: Path) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["start", "end", "label"])
        for b in BURSTS:
            writer.writerow([b - 8, b + 10, 1])
    return path


def make_fixture(out_dir: str | Path, duration: float = 180.0, size: tuple[int, int] = (1280, 720)) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "video": make_video(out / "sample_stream.mp4", duration, size),
        "chat": make_chat(out / "chat.json", duration),
        "clips": make_viewer_clips(out / "viewer_clips.json"),
        "transcript": make_transcript(out / "transcript.json", duration),
        "truth": make_truth(out / "truth.csv"),
        "facecam": dict(FACECAM),
        "bursts": BURSTS,
    }
