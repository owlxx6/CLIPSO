"""Configuration: packaged defaults, then config/local.yaml, streamer and tuned overlays, then env/CLI overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULTS_FILE = PACKAGE_DIR / "defaults.yaml"
FONTS_DIR = PACKAGE_DIR / "assets" / "fonts"


def project_root() -> Path:
    """Folder holding config/ and data/. CLIPPER_HOME wins, then the repo checkout, then the working directory."""
    env = os.environ.get("CLIPPER_HOME")
    if env:
        return Path(env).resolve()
    repo = PACKAGE_DIR.parent
    if (repo / "config").is_dir():
        return repo
    return Path.cwd()


def deep_merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def streamer_file(name: str | None) -> Path | None:
    if not name:
        return None
    path = project_root() / "config" / "streamers" / f"{name}.yaml"
    return path if path.exists() else None


def tuned_file(name: str | None) -> Path:
    return project_root() / "config" / "tuned" / f"{name or 'default'}.yaml"


def load_config(
    streamer: str | None = None,
    extra: str | Path | None = None,
    overrides: dict | None = None,
) -> dict:
    cfg = load_yaml(DEFAULTS_FILE)
    layers: list[Path | None] = [
        project_root() / "config" / "local.yaml",
        streamer_file(streamer),
        tuned_file(streamer),
        Path(os.environ["CLIPPER_CONFIG"]) if os.environ.get("CLIPPER_CONFIG") else None,
        Path(extra) if extra else None,
    ]
    for layer in layers:
        if layer is not None and Path(layer).exists():
            cfg = deep_merge(cfg, load_yaml(layer))
    if os.environ.get("CLIPPER_DATA_DIR"):
        cfg["data_dir"] = os.environ["CLIPPER_DATA_DIR"]
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def data_dir(cfg: dict) -> Path:
    path = Path(cfg.get("data_dir", "data"))
    if not path.is_absolute():
        path = project_root() / path
    path.mkdir(parents=True, exist_ok=True)
    return path
