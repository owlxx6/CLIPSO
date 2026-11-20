"""Local activity measurements for Stream Highlights. Times use the source analysis grid."""

from __future__ import annotations

import math

import cv2
import numpy as np

from clipper import layout, media
from clipper.signals import rolling_median, timeline_len
from clipper.signals import payoff
from clipper.signals import transcript as tr

CAMERA_TYPES = ("unknown", "gameplay_only", "webcam", "full_body", "screen_reaction", "avatar")


def usable_transcript(data: dict | None, duration: float) -> dict | None:
    """Drop invalid timing and dubious ASR segments before generating evidence or quoting them."""
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        return None

    def timed(item, lo, hi):
        return (
            isinstance(item, dict)
            and all(type(item.get(k)) in (int, float) for k in ("start", "end"))
            and lo <= item["start"] < item["end"] <= hi
        )

    clean = []
    for seg in data["segments"]:
        if not timed(seg, 0, duration + 0.1) or not isinstance(seg.get("text"), str) or not seg["text"].strip():
            continue
        if (
            seg.get("no_speech_prob", 0)
            and isinstance(seg["no_speech_prob"], (int, float))
            and seg["no_speech_prob"] > 0.7
        ):
            continue
        words = seg.get("words")
        words = (
            [
                w
                for w in words
                if timed(w, seg["start"] - 0.1, seg["end"] + 0.1)
                and isinstance(w.get("word"), str)
                and w["word"].strip()
            ]
            if isinstance(words, list)
            else []
        )
        clean.append(dict(seg, words=words))
    return dict(data, segments=sorted(clean, key=lambda s: s["start"])) if clean else None


def _pitch(block, rate, cfg):
    """Estimate periodicity in the loudest of three short windows of a mixed audio hop."""
    length = min(len(block), max(1, round(0.06 * rate)))
    if length <= rate / cfg["pitch_min_hz"] * 2:
        return 0.0
    windows = [block[i : i + length].astype(np.float32) for i in np.linspace(0, len(block) - length, 3, dtype=int)]
    frame = max(windows, key=lambda x: float(x.std()))
    frame -= frame.mean()
    if frame.std() < 80:
        return 0.0
    frame *= np.hanning(length)
    power = np.abs(np.fft.rfft(frame, n=2 * length)) ** 2
    corr = np.fft.irfft(power)[:length]
    corr /= max(corr[0], 1)
    lo, hi = max(2, int(rate / cfg["pitch_max_hz"])), min(length - 2, math.ceil(rate / cfg["pitch_min_hz"]))
    lags = np.arange(lo, hi + 1)
    peaks = lags[(corr[lags] >= corr[lags - 1]) & (corr[lags] > corr[lags + 1])]
    if not len(peaks):
        return 0.0
    peaks = peaks[corr[peaks] >= max(cfg["voicing_threshold"], 0.9 * max(corr[peaks]))]
    return rate / peaks[0] if len(peaks) else 0.0


def pitch_changes(source: dict, hop: float, rate: int, cfg: dict) -> np.ndarray:
    n = timeline_len(source["duration"], hop)
    contour = np.zeros(n, dtype=np.float32)
    if not source["has_audio"]:
        return contour
    step = max(1, round(rate * hop)) * 2
    rest, i = b"", 0
    for chunk in media.pcm_chunks(source["path"], rate):
        payload = rest + chunk
        full = len(payload) // step
        for offset in range(0, min(full, n - i) * step, step):
            contour[i] = _pitch(np.frombuffer(payload[offset : offset + step], dtype="<i2"), rate, cfg)
            i += 1
        rest = payload[full * step :]
    delta = np.zeros_like(contour)
    voiced = (contour[:-1] > 0) & (contour[1:] > 0)
    delta[1:][voiced] = np.abs(12 * np.log2(contour[1:][voiced] / contour[:-1][voiced]))
    return np.clip(delta / cfg["pitch_change_semitones"], 0, 1)


def audio_features(db, transcript, duration, hop, cfg, keywords, pitch=None):
    n = timeline_len(duration, hop)
    levels = np.full(n, -90.0, dtype=np.float32)
    count = min(n, len(db))
    levels[:count] = db[:count]
    baseline = rolling_median(levels, max(3, round(cfg["baseline_s"] / hop)))
    change = np.maximum(0, np.diff(levels, prepend=levels[0]))
    energy = np.clip(np.maximum(change, levels - baseline) / cfg["energy_rise_db"], 0, 1)
    speech = np.zeros(n, dtype=np.float32)
    for start, end in tr.segments(transcript):
        speech[int(start / hop) : math.ceil(end / hop)] = 1
    rate = np.clip(tr.speech_rate_curve(transcript, duration, hop)[:n] / 4, 0, 1) * speech
    words = tr.words(transcript)
    language = (transcript or {}).get("language")
    exclamation = np.clip(tr.keyword_curve(transcript, keywords, duration, hop, language), 0, 1)
    chaos_words = np.zeros(n, dtype=np.float32)
    for t, _ in tr.keyword_hits(words, cfg["chaos_terms"]):
        chaos_words[min(n - 1, max(0, int(t / hop)))] = 1
    laughter = np.zeros(n, dtype=np.float32)
    timed_words = {w["start"]: w["end"] for seg in (transcript or {}).get("segments", [])
                   for w in seg.get("words", [])}
    for t, _ in tr.keyword_hits(words, keywords["laughter"]):
        i = min(n - 1, max(0, int(t / hop)))
        # A word with observed timestamps can remain present across a plateau.
        laughter[i:max(i + 1, math.ceil(timed_words.get(t, t) / hop))] = 1
    # Mixed audio alone gives weak evidence. ASR cannot identify which speaker spoke.
    reaction = np.maximum(energy * (0.18 + 0.60 * speech), exclamation * (0.60 + 0.40 * energy))
    reaction = np.maximum(reaction, np.maximum(laughter * 0.90, rate * energy))
    pitch = np.zeros_like(energy) if pitch is None else pitch
    voiced_rise = np.minimum(pitch, energy)
    reaction = np.maximum(reaction, voiced_rise * 0.72)
    return dict(
        voiced_rise=voiced_rise,
        pitch_change=pitch,
        audio_reaction=reaction,
        energy_rise=energy,
        speech=speech,
        laughter_cue=laughter,
        speech_rate=rate,
        chaos_words=chaos_words,
    )


def _track_face(roi, finder, previous):
    if min(roi.shape) < 24 or roi.std() < 5:
        return None, 0.0
    boxes = finder.find(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR))
    if not boxes:
        return None, 0.0

    def distance(box):
        return math.hypot(
            box[0] + box[2] / 2 - previous[0] - previous[2] / 2,
            box[1] + box[3] / 2 - previous[1] - previous[3] / 2,
        )

    box = min(boxes, key=distance) if previous else max(boxes, key=lambda b: b[2] * b[3])
    x, y, w, h = box
    rh, rw = roi.shape
    face = roi[max(0, int(y * rh)) : max(0, int((y + h) * rh)), max(0, int(x * rw)) : max(0, int((x + w) * rw))]
    if not face.size:
        return None, 0.0
    quality = min(1.0, min(face.shape) / 40) * min(1.0, cv2.Laplacian(face, cv2.CV_32F).var() / 65)
    return box, float(quality)


def visual_features(source: dict, facecam: dict | None, duration: float, hop: float, cfg: dict,
                    reference=None) -> dict:
    """Stream downsized grayscale samples. Exclude the whole camera overlay from gameplay measurements."""
    n = timeline_len(duration, hop)
    width = min(int(cfg["scan_width"]), int(source["width"]))
    height = max(2, round(width * source["height"] / source["width"]))
    fps = min(float(cfg["sample_fps"]), 1 / hop)
    mask = np.ones((height, width), dtype=bool)
    finder = None
    camera = cfg["camera_type"]
    if facecam and camera != "gameplay_only":
        x, y, w, h = (facecam[k] for k in ("x", "y", "w", "h"))
        roi = np.s_[
            max(0, int(y * height)) : min(height, math.ceil((y + h) * height)),
            max(0, int(x * width)) : min(width, math.ceil((x + w) * width)),
        ]
        mask[roi] = False
        if camera == "webcam":
            finder = layout.FaceFinder()
    measure_game = camera != "screen_reaction" and (camera not in {"avatar", "full_body"} or facecam is not None)
    if reference is not None and measure_game:
        x, y, w, h = cfg["result_card"]["roi"]
        card_roi = np.s_[int(y*height):math.ceil((y+h)*height), int(x*width):math.ceil((x+w)*width)]
        if not mask[card_roi].all():
            raise ValueError("result_card ROI overlaps the camera overlay")
        rh, rw = mask[card_roi].shape
        reference = cv2.resize(reference, (rw, rh), interpolation=cv2.INTER_AREA)
        if min(rh, rw) < 3 or reference.std() < 5:
            raise ValueError("result_card ROI has too little detail at scan_width")
        mask[card_roi] = False  # a HUD or result update alone cannot seed a proposal
    raw, scene, face, confidence = (np.zeros(n, np.float32) for _ in range(4))
    changed_fraction = np.full(n, np.nan, np.float32)  # unsampled time cannot certify idle lead-in
    visible = np.zeros(n, dtype=bool)
    hold = payoff.PictureHold(cfg, n)
    histograms, indices = [], []
    prev_frame = prev_pixels = prev_hist = prev_box = None
    prev_quality = 0.0
    for t, frame in media.gray_frames(source["path"], fps, width, height):
        i = min(n - 1, int(round(t / hop)))
        pixels = frame[::4, ::4][mask[::4, ::4]]
        hist = np.histogram(pixels, bins=16, range=(0, 256))[0] / max(1, pixels.size)
        histograms.append(hist)
        indices.append(i)
        if reference is not None and measure_game:
            similarity = cv2.matchTemplate(frame[card_roi], reference, cv2.TM_CCOEFF_NORMED)[0, 0]
            visible[i] = similarity >= cfg["result_card"]["similarity"]
        if pixels.size and measure_game:
            hold.observe(i, pixels, prev_pixels, hop, t)
        if prev_frame is not None and pixels.size:
            frame_delta = cv2.absdiff(frame, prev_frame)[::4, ::4]
            delta = frame_delta[mask[::4, ::4]]
            raw[i] = np.mean(delta > 20)
            changed_fraction[i] = np.mean(frame_delta > 20)  # protect changes inside overlays as well
            scene[i] = np.abs(hist - prev_hist).sum() / 2
        if finder:
            box, quality = _track_face(frame[roi], finder, prev_box)
            confidence[i] = quality
            if prev_frame is not None and box and prev_box:
                x, y, w, h = box
                rh, rw = frame[roi].shape
                region = np.s_[
                    max(0, int(y * rh)) : max(0, int((y + h) * rh)),
                    max(0, int(x * rw)) : max(0, int((x + w) * rw)),
                ]
                change = cv2.absdiff(frame[roi][region], prev_frame[roi][region])
                motion = math.hypot(x - prev_box[0], y - prev_box[1])
                if motion < 0.35 and change is not None:
                    face[i] = min(1, change.mean() / 32 + motion * 2) * min(quality, prev_quality)
            prev_box, prev_quality = box, quality
        prev_frame, prev_pixels, prev_hist = frame, pixels, hist
    stride = max(1, round(1 / (fps * hop)))
    typical = rolling_median(raw[::stride], max(3, round(cfg["baseline_s"] * fps)))
    background = np.repeat(typical, stride)[:n]
    novelty = np.clip((raw - background) / cfg["motion_rise"], 0, 1)
    game = np.maximum(novelty * np.clip(raw / 0.18, 0, 1), np.clip(scene / cfg["scene_change"], 0, 1))
    return dict(
        gameplay_activity=game if measure_game else np.zeros_like(game),
        scene_change=scene,
        changed_fraction=changed_fraction,
        hard_cut=payoff.hard_cuts(histograms, indices, cfg["hard_cut"], n),
        quiet_hold_until=hold.until,
        result_visible=payoff.confirmed_card_samples(visible, indices, hop, cfg["hold_s"]),
        novelty=novelty if measure_game else np.zeros_like(novelty),
        facecam_activity=face,
        facecam_confidence=confidence,
    )
