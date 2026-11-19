"""Duplicate guard: never post the same moment twice on the same platform.

Two checks, both per platform and both before anything is rendered:
1. Same source, overlapping time range (cheap, exact).
2. Perceptual signature of the source footage (catches the same moment from another copy of the VOD): one 256-bit
   difference hash per second of the gameplay, compared frame by frame with a few seconds of allowed shift.
   It is taken from the source, not from the finished clip, and leaves out the facecam: the facecam, hook,
   captions, colours and zoom are the same from one clip to the next and say nothing about which moment it is.
"""

from __future__ import annotations

from pathlib import Path

from clipper.media import gray_frames

FLAT = "-"  # frames with almost no detail (black, fades) carry no identity and are skipped
Region = tuple[float, float, float, float]  # x, y, w, h as fractions of the frame


def gameplay_region(facecam: dict | None) -> Region:
    """The largest part of the frame that the facecam does not cover."""
    if not facecam:
        return (0.0, 0.0, 1.0, 1.0)
    x0, y0 = float(facecam["x"]), float(facecam["y"])
    x1, y1 = x0 + float(facecam["w"]), y0 + float(facecam["h"])
    options = [(0.0, 0.0, x0, 1.0), (x1, 0.0, 1.0 - x1, 1.0), (0.0, 0.0, 1.0, y0), (0.0, y1, 1.0, 1.0 - y1)]
    return max(options, key=lambda r: r[2] * r[3])


def crop_filter(region: Region) -> str | None:
    x, y, w, h = region
    if (x, y, w, h) == (0.0, 0.0, 1.0, 1.0) or w <= 0 or h <= 0:
        return None
    return f"crop=w=iw*{w:.4f}:h=ih*{h:.4f}:x=iw*{x:.4f}:y=ih*{y:.4f}"


def signature(video: str | Path, step_s: float = 1.0, size: int = 16, start: float | None = None,
              duration: float | None = None, facecam: dict | None = None) -> list[str]:
    """One hash per step_s seconds of video (or of the window start..start+duration), facecam left out."""
    hashes = []
    crop = crop_filter(gameplay_region(facecam))
    for _, frame in gray_frames(video, fps=1.0 / step_s, width=size + 1, height=size, start=start,
                                duration=duration, crop=crop):
        if float(frame.std()) < 3.0:
            hashes.append(FLAT)
            continue
        bits = (frame[:, 1:] > frame[:, :-1]).flatten()
        value = int("".join("1" if b else "0" for b in bits), 2)
        hashes.append(f"{value:0{size * size // 4}x}")
    return hashes


def window(sig: list[str], t0: float, start: float, end: float, step_s: float = 1.0) -> list[str]:
    """The part of a signature taken from t0 onwards that covers start..end."""
    a = max(0, int(round((start - t0) / step_s)))
    b = max(a, int(round((end - t0) / step_s)))
    return sig[a:b]


def similarity(sig_a: list[str], sig_b: list[str], max_shift: int = 3, max_fraction: float = 0.16) -> float:
    """Share of comparable frames that match, at the best alignment (0 = different, 1 = same clip)."""
    a = [None if h == FLAT else int(h, 16) for h in sig_a]
    b = [None if h == FLAT else int(h, 16) for h in sig_b]
    sample = next((h for h in list(sig_a) + list(sig_b) if h != FLAT), None)
    if sample is None:
        return 0.0
    threshold = int(len(sample) * 4 * max_fraction)
    best = 0.0
    for shift in range(-max_shift, max_shift + 1):
        pairs = [(a[i], b[i - shift]) for i in range(len(a)) if 0 <= i - shift < len(b)]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if len(pairs) < 3:
            continue
        matches = sum((x ^ y).bit_count() <= threshold for x, y in pairs)
        best = max(best, matches / len(pairs))
    return round(best, 3)


def overlapping_export(source_id: int, start: float, end: float, previous: list[dict],
                       min_overlap: float = 0.5) -> tuple[int, float] | None:
    """An earlier export of the same source whose time range covers most of [start, end]."""
    for row in previous:
        if row.get("blocked_reason") or int(row["source_id"]) != int(source_id):
            continue
        inter = max(0.0, min(end, row["end_s"]) - max(start, row["start_s"]))
        share = inter / max(1e-6, min(end - start, row["end_s"] - row["start_s"]))
        if share >= min_overlap:
            return int(row["id"]), round(share, 2)
    return None


def find_duplicate(sig: list[str], previous: list[dict], dcfg: dict) -> tuple[int, float] | None:
    """previous: export rows of the same platform (with 'id' and 'signature')."""
    threshold = float(dcfg.get("block_similarity", 0.8))
    worst: tuple[int, float] | None = None
    for row in previous:
        other = row.get("signature") or []
        if not other or row.get("blocked_reason"):
            continue
        sim = similarity(sig, other, int(dcfg.get("max_shift", 3)), float(dcfg.get("hamming_max_fraction", 0.16)))
        if sim >= threshold and (worst is None or sim > worst[1]):
            worst = (int(row["id"]), sim)
    return worst
