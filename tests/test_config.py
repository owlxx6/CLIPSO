import shutil
from pathlib import Path

import yaml

from clipper import config

REPO = Path(__file__).resolve().parent.parent
FREE_SECTIONS = {"streamer"}  # profile-only settings that have no default


def _paths(tree: dict, prefix: str = "") -> list[str]:
    out = []
    for key, value in tree.items():
        path = f"{prefix}{key}"
        out += _paths(value, path + ".") if isinstance(value, dict) else [path]
    return out


def test_example_files_only_use_known_settings():
    defaults = yaml.safe_load(config.DEFAULTS_FILE.read_text(encoding="utf-8"))
    for name in ("config/streamers/example.yaml", "config/local.example.yaml"):
        data = yaml.safe_load((REPO / name).read_text(encoding="utf-8"))
        for path in _paths(data):
            if path.split(".")[0] in FREE_SECTIONS:
                continue
            assert config.get(defaults, path) is not None, f"{name}: unknown setting {path}"


def test_streamer_profile_and_tuned_weights_are_layered(cfg, tmp_path):
    (tmp_path / "config" / "streamers").mkdir(parents=True)
    shutil.copy(REPO / "config" / "streamers" / "example.yaml", tmp_path / "config" / "streamers" / "demo.yaml")
    (tmp_path / "config" / "tuned").mkdir()
    (tmp_path / "config" / "tuned" / "demo.yaml").write_text("detection:\n  weights:\n    chat: 2.5\n",
                                                            encoding="utf-8")
    c = config.load_config(streamer="demo")
    assert config.get(c, "streamer.display_name") == "Example"
    assert config.get(c, "render.disclosure.enabled") is True
    assert config.get(c, "detection.weights.chat") == 2.5
    assert config.get(c, "detection.weights.audio") == 1.0  # untouched defaults stay
    assert "exampleLUL" in config.get(c, "signals.chat.emotes")
