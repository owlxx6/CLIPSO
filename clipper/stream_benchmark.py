"""Replay three detector runs against one frozen set of local human labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import zipfile
from contextlib import closing
from pathlib import Path

from clipper import config, db, evaluate, media, pipeline


SIDECARS = ("transcript.bound.json", "transcript.generated.json", "layout.json",
            "chat.csv", "chat.json", "viewer_clips.json")


def _library(cfg):
    root = Path(cfg.get("data_dir", "data"))
    path = (config.project_root() / root / "clipper.sqlite").resolve()
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn, path.parent


def _write_new(path, data):
    with path.open("x", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def _check_labels(labels, duration):
    def valid(row):
        return (isinstance(row, dict) and row.get("label") in (0, 1)
                and all(type(row.get(key)) in (int, float) for key in ("start_s", "end_s"))
                and 0 <= row["start_s"] < row["end_s"] <= duration)

    if not isinstance(labels, list) or not labels or not all(valid(row) for row in labels):
        raise ValueError("Provide human labels with valid intervals inside this recording.")


def freeze(source_id, output):
    conn, _ = _library(config.load_config())
    with closing(conn):
        source = db.get_source(conn, source_id)
        labels = [{key: row[key] for key in ("start_s", "end_s", "label")}
                  for row in db.list_labels(conn, source_id)]
    _check_labels(labels, source["duration"])
    _write_new(output, {"version": 1, "source_sha256": media.sha256_file(source["path"]),
                        "duration": source["duration"], "labels": labels})


def _checkout(repo, rev, directory):
    sha = subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options", rev + "^{commit}"],
                                  cwd=repo, text=True).strip()
    archive = directory.with_suffix(".zip")
    subprocess.run(["git", "archive", "-o", str(archive), sha, "clipper"], cwd=repo, check=True)
    with zipfile.ZipFile(archive) as files:
        files.extractall(directory)
    return sha


def _run(checkout, folder, inputs, settings, source, mode):
    data = folder / "data"
    work = data / "sources" / str(source["id"])
    shutil.copytree(inputs, work)
    with closing(db.connect(dict(data_dir=str(data)))) as conn:
        db.add_source(conn, **source)
    env = dict(os.environ, CLIPPER_HOME=str(folder), CLIPPER_DATA_DIR=str(data), CLIPPER_CONFIG=str(settings),
               HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
    subprocess.run([sys.executable, "-m", "clipper", "analyze", str(source["id"]), "--mode", mode,
                    "--no-transcribe", "--top", "0"], cwd=checkout, env=env, check=True)
    with closing(db.connect(dict(data_dir=str(data)))) as conn:
        return db.list_candidates(conn, source["id"], mode)


def _measure(rows, labels, k):
    positive = evaluate.truth_from_labels(labels)
    negative = [(r["start_s"], r["end_s"]) for r in labels if r["label"] == 0]
    top = [{"start": r["start_s"], "end": r["end_s"], "peak": r["peak_s"], "score": r["score"],
            "channel_scores": r["flags"].get("channel_scores")} for r in rows[:k]]
    metrics = evaluate.evaluate(top, positive, k)
    for prediction, hit in zip(top, metrics["hits"]):
        prediction["positive_match"] = None if hit is None else positive[hit]
        if hit is not None:
            left, right = positive[hit]
            prediction["start_error_s"] = round(prediction["start"] - left, 3)
            prediction["end_error_s"] = round(prediction["end"] - right, 3)
        prediction["overlapping_negative_labels"] = [interval for interval in negative
            if max(prediction["start"], interval[0]) < min(prediction["end"], interval[1])]
    # Unmatched predictions may be unlabeled. Report counts and labeled recall even on sparse annotations.
    return {"total_candidates": len(rows), "returned_at_k": len(top), "matched_positives": sum(
            hit is not None for hit in metrics["hits"]), "labeled_recall": metrics["recall"] if positive else None,
            "top": top}


def compare(source_id, frozen, output, baseline="791188c", corrected="HEAD", k=10):
    if k <= 0 or output.exists():
        raise ValueError("Use a positive k and a new report path.")
    manifest_bytes = frozen.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    conn, data = _library(config.load_config())
    with closing(conn):
        source = db.get_source(conn, source_id)
    digest = media.sha256_file(source["path"])
    if (manifest.get("version") != 1 or manifest.get("source_sha256") != digest
            or manifest.get("duration") != source["duration"]):
        raise ValueError("Frozen labels belong to a different recording or manifest version.")
    labels = manifest["labels"]
    _check_labels(labels, source["duration"])
    cfg = pipeline.source_config(source)
    cfg.pop("data_dir", None)
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="clipso-benchmark-") as folder:
        tmp = Path(folder)
        old, new = tmp / "baseline", tmp / "corrected"
        baseline_sha, corrected_sha = _checkout(repo, baseline, old), _checkout(repo, corrected, new)
        inputs = tmp / "inputs"
        inputs.mkdir()
        hashes = {}
        for name in SIDECARS:
            original = data / "sources" / str(source_id) / name
            if original.is_file():
                copied = shutil.copyfile(original, inputs / name)
                hashes[name] = media.sha256_file(copied)
        card = cfg["stream_highlights"]["result_card"]
        if card["template"]:
            template = config.project_root() / card["template"]
            snapshot = tmp / ("result-card" + template.suffix)
            shutil.copyfile(template, snapshot)
            hashes["result_card_template"] = media.sha256_file(snapshot)
            card["template"] = snapshot.name  # deterministic config digest
        settings = tmp / "settings.json"
        config_digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        if card["template"]:
            card["template"] = str(snapshot)
        _write_new(settings, cfg)  # JSON also loads as YAML
        report = {"source_sha256": digest, "labels_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                  "config_sha256": config_digest, "input_files": hashes, "k": k, "labels": labels, "runs": {}}
        for name, mode, checkout, sha in (("classic", "classic", old, baseline_sha),
                ("audited_stream", "stream_highlights", old, baseline_sha),
                ("corrected_stream", "stream_highlights", new, corrected_sha)):
            rows = _run(checkout, tmp / name, inputs, settings, source, mode)
            report["runs"][name] = dict(revision=sha, **_measure(rows, labels, k))
        if media.sha256_file(source["path"]) != digest:
            raise RuntimeError("The recording changed during the comparison.")
    _write_new(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("freeze", help="freeze the source's existing human labels")
    replay = commands.add_parser("compare", help="run three local analyses with frozen labels")
    for sub in (capture, replay):
        sub.add_argument("source", type=int)
        sub.add_argument("labels", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--baseline", default="791188c")
    replay.add_argument("--corrected", default="HEAD")
    replay.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            freeze(args.source, args.labels)
        else:
            compare(args.source, args.labels, args.output, args.baseline, args.corrected, args.k)
    except (OSError, sqlite3.Error, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")


if __name__ == "__main__":
    main()
