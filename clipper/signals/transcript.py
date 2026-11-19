"""Speech: faster-whisper transcript with word timestamps, hype keywords, laughter and speech rate."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from clipper.signals import bumps, rolling_sum, timeline_len

# --- transcription -----------------------------------------------------------------------------------------------


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def transcribe(path: str | Path, tcfg: dict, log=print) -> dict | None:
    """Run faster-whisper on the media file. Returns None when the package is missing."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        log("faster-whisper is not installed: skipping speech signals (pip install -e \".[transcribe]\")")
        return None
    log(f"transcribing with faster-whisper '{tcfg.get('model', 'small')}' on {tcfg.get('device', 'cpu')} "
        "(the first run downloads the model)")
    model = WhisperModel(tcfg.get("model", "small"), device=tcfg.get("device", "cpu"),
                         compute_type=tcfg.get("compute_type", "int8"))
    segments, info = model.transcribe(
        str(path),
        language=tcfg.get("language") or None,
        word_timestamps=True,
        vad_filter=bool(tcfg.get("vad_filter", True)),
        beam_size=int(tcfg.get("beam_size", 1)),
    )
    out = {"language": info.language, "segments": []}
    for seg in segments:
        out["segments"].append({
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": seg.text.strip(),
            "words": [{"start": round(float(w.start), 3), "end": round(float(w.end), 3), "word": w.word.strip()}
                      for w in (seg.words or [])],
        })
    return out


def load(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def save(path: str | Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)


def words(transcript: dict | None) -> list[dict]:
    if not transcript:
        return []
    out = []
    for seg in transcript.get("segments", []):
        seg_words = seg.get("words") or []
        if seg_words:
            out.extend(w for w in seg_words if w.get("word"))
        elif seg.get("text"):
            # segment without word timings: spread the words evenly
            toks = seg["text"].split()
            span = max(0.01, (seg["end"] - seg["start"]) / max(1, len(toks)))
            out.extend({"start": seg["start"] + i * span, "end": seg["start"] + (i + 1) * span, "word": tok}
                       for i, tok in enumerate(toks))
    return out


def segments(transcript: dict | None) -> list[tuple[float, float]]:
    if not transcript:
        return []
    return [(float(s["start"]), float(s["end"])) for s in transcript.get("segments", [])]


def text_between(transcript: dict | None, start: float, end: float) -> str:
    return " ".join(w["word"] for w in words(transcript) if w["end"] > start and w["start"] < end).strip()


# --- keywords ----------------------------------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^\w'+]+", " ", text).strip()


def keyword_hits(word_list: list[dict], phrases: list[str], weight: float = 1.0) -> list[tuple[float, float]]:
    tokens: list[tuple[str, float]] = []
    for w in word_list:
        for tok in normalize(w["word"]).split():
            tokens.append((tok, float(w["start"])))
    by_first: dict[str, list[list[str]]] = {}
    for phrase in phrases:
        toks = normalize(phrase).split()
        if toks:
            by_first.setdefault(toks[0], []).append(toks)
    hits = []
    for i, (tok, t) in enumerate(tokens):
        for toks in by_first.get(tok, []):
            if [x for x, _ in tokens[i:i + len(toks)]] == toks:
                hits.append((t, weight))
    return hits


def keyword_curve(transcript: dict | None, kcfg: dict, duration: float, hop_s: float, language: str | None) -> np.ndarray:
    n = timeline_len(duration, hop_s)
    wl = words(transcript)
    if not wl:
        return np.zeros(n, dtype=np.float32)
    phrases = list(kcfg.get("fr", [])) + list(kcfg.get("en", [])) + list(kcfg.get("extra", []))
    hits = keyword_hits(wl, phrases, 1.0)
    hits += keyword_hits(wl, list(kcfg.get("laughter", [])), float(kcfg.get("laughter_weight", 1.5)))
    curve = bumps([t for t, _ in hits], [w for _, w in hits], n, hop_s, float(kcfg.get("window_s", 2.0)) / 2)
    return np.clip(curve, 0.0, 6.0)


def speech_rate_curve(transcript: dict | None, duration: float, hop_s: float, window_s: float = 3.0) -> np.ndarray:
    """Words per second against the speaker's usual pace: rapid-fire talking marks hype moments."""
    n = timeline_len(duration, hop_s)
    wl = words(transcript)
    if not wl:
        return np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)
    for w in wl:
        i = int(float(w["start"]) / hop_s)
        if 0 <= i < n:
            counts[i] += 1
    k = max(1, int(round(window_s / hop_s)))
    wps = rolling_sum(counts, k) / (k * hop_s)
    talking = wps[wps > 0]
    if talking.size < 10:
        return np.zeros(n, dtype=np.float32)
    base = float(np.median(talking))
    mad = float(np.median(np.abs(talking - base))) * 1.4826
    z = (wps - base) / max(mad, 0.3)
    return np.clip(z, 0.0, 6.0).astype(np.float32)


def peak_phrase(transcript: dict | None, peak: float, max_words: int = 7) -> str:
    """Words spoken around the peak, used as the auto hook."""
    wl = [w for w in words(transcript) if peak - 3.0 <= float(w["start"]) <= peak + 1.5]
    if not wl:
        return ""
    text = " ".join(w["word"] for w in wl[:max_words])
    return re.sub(r"\s+", " ", text).strip(" ,.;:")
