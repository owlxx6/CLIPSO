import importlib.util
import subprocess

import numpy as np
import pytest

from clipper import fixture, layout, media


def test_refine_box_snaps_to_persistent_border(cfg):
    h, w = 360, 640
    edge_x = np.random.default_rng(0).uniform(0, 2, (h, w - 1)).astype(np.float32)
    edge_y = np.random.default_rng(1).uniform(0, 2, (h - 1, w)).astype(np.float32)
    edge_x[15:165, 469] = 60  # overlay left border before column 470
    edge_x[15:165, 619] = 60  # right border after column 619
    edge_y[14, 470:620] = 60  # top border before row 15
    edge_y[164, 470:620] = 60  # bottom border after row 164
    face = (520 / w, 38 / h, 38 / w, 44 / h)
    box = layout.refine_box(face, edge_x, edge_y, cfg["layout"])
    assert round(box["x"] * w) == 470 and round((box["x"] + box["w"]) * w) == 620
    assert round(box["y"] * h) == 15 and round((box["y"] + box["h"]) * h) == 165


def test_refine_box_falls_back_to_expansion_without_border(cfg):
    edge_x = np.ones((360, 639), np.float32)
    edge_y = np.ones((359, 640), np.float32)
    box = layout.refine_box((0.5, 0.3, 0.05, 0.1), edge_x, edge_y, cfg["layout"])
    assert box["w"] == pytest.approx(0.05 * cfg["layout"]["facecam_expand_w"], abs=0.01)


def test_parse_box_validates():
    assert layout.parse_box("0.72,0.04,0.26,0.30") == {"x": 0.72, "y": 0.04, "w": 0.26, "h": 0.30}
    with pytest.raises(ValueError):
        layout.parse_box("72,4,26,30")


def test_no_face_means_no_facecam(cfg, fx):
    assert layout.detect_facecam(fx["video"], 180, (640, 360), cfg["layout"]) is None


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("skimage") is None, reason="scikit-image provides a public-domain face")
def test_detects_real_face_in_overlay(cfg, tmp_path):
    import cv2
    import skimage.data

    face_png = tmp_path / "face.png"
    cv2.imwrite(str(face_png), skimage.data.astronaut()[:, :, ::-1])  # NASA portrait, public domain
    texture = fixture.make_texture(tmp_path / "texture.png", (640, 360))
    video = tmp_path / "stream.mp4"
    subprocess.run([media.ffmpeg_bin(), "-v", "error", "-y", "-i", str(texture), "-loop", "1", "-i", str(face_png),
                    "-filter_complex",
                    "[0:v]loop=loop=-1:size=1:start=0,setpts=N/30/TB,"
                    "crop=w=640:h=360:x='640*(1+sin(t*0.3))':y='360*(1+sin(t*0.2+1))'[g];"
                    "[1:v]crop=w=400:h=400:x='56+50*sin(t)':y='40+30*cos(t)',scale=150:150[f];"
                    "[g][f]overlay=470:15:shortest=1[v]", "-map", "[v]", "-t", "12", "-c:v", "libx264",
                    "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(video)], check=True)
    found = layout.detect_facecam(video, 12, (640, 360), cfg["layout"])
    assert found is not None
    fc = found["facecam"]
    assert abs(fc["x"] * 640 - 470) <= 3 and abs((fc["x"] + fc["w"]) * 640 - 620) <= 3
    assert abs(fc["y"] * 360 - 15) <= 3 and abs((fc["y"] + fc["h"]) * 360 - 165) <= 3
