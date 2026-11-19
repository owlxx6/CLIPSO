"""Burned-in captions as an ASS file: word-by-word highlight, hook text and paid-content disclosure."""

from __future__ import annotations

from clipper import config

PLAY_W, PLAY_H = 1080, 1920


def hex_to_ass(hex_color: str, alpha: int = 0) -> str:
    """'#RRGGBB' -> '&HAABBGGRR' (style colours)."""
    h = hex_color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def hex_to_inline(hex_color: str) -> str:
    """'#RRGGBB' -> '&HBBGGRR&' (inline \\c override)."""
    h = hex_color.lstrip("#")
    return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&".upper()


def ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def clean(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ").strip()


def chunk_words(words: list[dict], max_words: int = 3, max_chars: int = 18, max_gap: float = 0.6) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        text = str(w.get("word", "")).strip()
        if not text:
            continue
        if cur:
            gap = float(w["start"]) - float(cur[-1]["end"])
            chars = sum(len(str(x["word"]).strip()) + 1 for x in cur) + len(text)
            if len(cur) >= max_words or chars > max_chars or gap > max_gap:
                chunks.append(cur)
                cur = []
        cur.append(w)
        if text[-1] in ".!?…":
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks


def positions(kind: str, cfg: dict) -> dict:
    band = int(config.get(cfg, "render.facecam_band_h", 640))
    if kind == "stacked":
        return {"hook": band, "sub": band + 290}
    if kind == "blur_fit":
        return {"hook": 470, "sub": 1430}
    return {"hook": 330, "sub": 1300}


def build_ass(
    words_rel: list[dict],
    cfg: dict,
    kind: str,
    highlight_color: str = "#FFE500",
    hook: str | None = None,
    disclosure: str | None = None,
    clip_duration: float | None = None,
) -> str:
    r = cfg["render"]
    font = r.get("font_name", "Anton")
    upper = bool(r.get("uppercase", True))
    sub_cfg, hook_cfg, tag_cfg = r.get("subtitles", {}), r.get("hook", {}), r.get("disclosure", {})
    pos = positions(kind, cfg)
    white = hex_to_inline("#FFFFFF")
    hl = hex_to_inline(highlight_color)

    def fmt(text: str) -> str:
        text = clean(text)
        return text.upper() if upper else text

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {PLAY_W}",
        f"PlayResY: {PLAY_H}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Sub,{font},{int(sub_cfg.get('font_size', 84))},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        f"0,0,0,0,100,100,1,0,1,{int(sub_cfg.get('outline', 6))},2,5,60,60,0,1",
        f"Style: Hook,{font},{int(hook_cfg.get('font_size', 62))},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
        "0,0,0,0,100,100,1,0,3,16,0,5,80,80,0,1",
        f"Style: Tag,{font},{int(tag_cfg.get('font_size', 54))},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
        f"0,0,0,0,100,100,2,0,3,10,0,7,{int(tag_cfg.get('margin_x', 64))},64,{int(tag_cfg.get('margin_top', 250))},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events: list[str] = []
    end_all = clip_duration if clip_duration else (max((float(w["end"]) for w in words_rel), default=3.0) + 0.5)

    if hook and hook_cfg.get("enabled", True):
        events.append(f"Dialogue: 2,{ass_time(0)},{ass_time(float(hook_cfg.get('duration_s', 2.6)))},Hook,,0,0,0,,"
                      f"{{\\pos(540,{pos['hook']})}}{fmt(hook)}")
    if disclosure:
        events.append(f"Dialogue: 3,{ass_time(0)},{ass_time(end_all)},Tag,,0,0,0,,{fmt(disclosure)}")

    if sub_cfg.get("enabled", True):
        chunks = chunk_words(words_rel, int(sub_cfg.get("words_per_chunk", 3)), int(sub_cfg.get("max_chars", 18)),
                             float(sub_cfg.get("max_gap_s", 0.6)))
        for chunk in chunks:
            chunk_end = max(float(chunk[-1]["end"]), float(chunk[-1]["start"]) + 0.2)
            for i, w in enumerate(chunk):
                start = float(w["start"])
                end = float(chunk[i + 1]["start"]) if i + 1 < len(chunk) else chunk_end
                if end - start < 0.05:
                    end = start + 0.05
                parts = [f"{{\\c{hl}}}{fmt(x['word'])}{{\\c{white}}}" if j == i else fmt(x["word"])
                         for j, x in enumerate(chunk)]
                events.append(f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Sub,,0,0,0,,"
                              f"{{\\pos(540,{pos['sub']})}}{' '.join(parts)}")
    return "\n".join(header + events) + "\n"


def clip_words(all_words: list[dict], start: float, end: float) -> list[dict]:
    """Words inside [start, end], re-timed relative to the clip start."""
    out = []
    for w in all_words:
        ws, we = float(w["start"]), float(w["end"])
        if we <= start or ws >= end:
            continue
        out.append({"start": max(0.0, ws - start), "end": min(end - start, we - start), "word": w["word"]})
    return out
