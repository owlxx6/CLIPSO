import pytest

from clipper import subtitles, variants


def _words(text, start=0.0, step=0.4):
    return [{"start": start + i * step, "end": start + i * step + 0.3, "word": w} for i, w in enumerate(text.split())]


def test_ass_time():
    assert subtitles.ass_time(1.5) == "0:00:01.50"
    assert subtitles.ass_time(3725.25) == "1:02:05.25"


def test_chunks_break_on_count_punctuation_and_gaps():
    words = _words("oh mon dieu! il a fait quoi la")
    words[-1]["start"] += 2.0
    words[-1]["end"] += 2.0
    chunks = [[w["word"] for w in c] for c in subtitles.chunk_words(words, max_words=3)]
    assert chunks[0] == ["oh", "mon", "dieu!"]
    assert chunks[-1] == ["la"]
    assert all(len(c) <= 3 for c in chunks)


def test_ass_has_styles_events_and_escapes(cfg):
    words = _words("salut {les} gens")
    ass = subtitles.build_ass(words, cfg, "stacked", "#FFE500", hook="quelle dinguerie", disclosure="PUBLICITÉ",
                              clip_duration=5)
    assert "Style: Sub,Anton" in ass and "Style: Hook" in ass and "Style: Tag" in ass
    assert ass.count("Dialogue:") == 3 + 1 + 1  # one event per word, hook, tag
    assert "{LES}" not in ass and "(LES)" in ass
    assert "QUELLE DINGUERIE" in ass
    assert "\\c&H00E5FF&" in ass


def test_clip_words_are_relative_and_clipped():
    words = _words("a b c d e f", start=10.0, step=1.0)  # each word lasts 0.3 s
    rel = subtitles.clip_words(words, 11.2, 13.5)
    assert [w["word"] for w in rel] == ["b", "c", "d"]
    assert rel[0]["start"] == 0.0 and rel[-1]["end"] <= 2.3


def test_hex_colours():
    assert subtitles.hex_to_ass("#FFE500") == "&H0000E5FF"
    assert subtitles.hex_to_inline("#FFE500") == "&H00E5FF&"


def test_variants_have_distinct_cuts_and_fit_targets(cfg):
    plans = variants.plan_variants(7, 100.0, 125.0, 112.0, 600.0, ["tiktok", "instagram", "youtube"], cfg)
    assert [v.platform for v in plans] == ["tiktok", "instagram", "youtube"]
    for i, a in enumerate(plans):
        assert 15 <= a.end - a.start <= 34 and a.start < 110 and a.end > 114
        for b in plans[i + 1:]:
            assert abs(a.start - b.start) >= 0.4 and abs(a.end - b.end) >= 0.4
    again = variants.plan_variants(7, 100.0, 125.0, 112.0, 600.0, ["tiktok", "instagram", "youtube"], cfg)
    assert [(v.start, v.end) for v in again] == [(v.start, v.end) for v in plans]


def test_variants_never_reuse_cut_points(cfg):
    first = variants.plan_variants(7, 100.0, 125.0, 112.0, 600.0, ["tiktok"], cfg)[0]
    used = {"tiktok": [(first.start, first.end)]}
    second = variants.plan_variants(7, 100.0, 125.0, 112.0, 600.0, ["tiktok"], cfg, used)[0]
    assert (abs(second.start - first.start) >= 0.25) or (abs(second.end - first.end) >= 0.25)


def test_unknown_layout_is_rejected(cfg):
    from clipper import render

    with pytest.raises(ValueError):
        render.video_filter("diagonal", None, cfg)


@pytest.mark.parametrize("cam_x", [0.725, 0.02])
def test_stacked_gameplay_does_not_show_the_facecam_twice(cfg, tmp_path, cam_x):
    import subprocess

    import cv2
    import numpy as np

    from clipper import media, render

    cam = {"x": cam_x, "y": 0.0333, "w": 0.25, "h": 0.3333}
    img = np.full((720, 1280, 3), 128, np.uint8)
    x0, x1 = round(cam["x"] * 1280), round((cam["x"] + cam["w"]) * 1280)
    img[24:264, x0:x1] = (255, 0, 0)  # pure blue facecam (BGR)
    src, out = tmp_path / "frame.png", tmp_path / "out.png"
    cv2.imwrite(str(src), img)
    graph = render.video_filter("stacked", cam, cfg, zoom=1.0, out="v")
    subprocess.run([media.ffmpeg_bin(), "-v", "error", "-y", "-loop", "1", "-t", "0.2", "-i", str(src),
                    "-filter_complex", graph, "-map", "[v]", "-frames:v", "1", "-update", "1", str(out)], check=True)
    frame = cv2.imread(str(out)).astype(int)
    band = int(cfg["render"]["facecam_band_h"])
    blue = (frame[:, :, 0] - frame[:, :, 2]) > 60
    assert blue[:band].mean() > 0.9          # the facecam fills the top band
    assert blue[band:].sum() == 0            # and does not show up again in the gameplay
