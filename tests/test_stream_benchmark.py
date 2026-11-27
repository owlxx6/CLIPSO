import json

import pytest

from clipper import db, stream_benchmark as benchmark


def test_frozen_labels_require_valid_intervals_and_matching_recording(source, tmp_path):
    conn, sid = source
    labels = tmp_path / "labels.json"
    with pytest.raises(ValueError, match="human labels"):
        benchmark.freeze(sid, labels)
    db.add_label(conn, sid, 10, 30, 1)
    benchmark.freeze(sid, labels)
    with pytest.raises(FileExistsError):
        benchmark.freeze(sid, labels)
    data = json.loads(labels.read_text())
    data["source_sha256"] = "different"
    labels.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different recording"):
        benchmark.compare(sid, labels, tmp_path / "result.json")


@pytest.mark.slow
def test_three_detector_runs_reuse_frozen_labels_and_leave_library_intact(source, tmp_path):
    conn, sid = source
    labels, result = tmp_path / "labels.json", tmp_path / "result.json"
    db.add_label(conn, sid, 35, 54, 1)
    db.add_label(conn, sid, 90, 94, 0)
    benchmark.freeze(sid, labels)
    frozen = labels.read_bytes()
    db.add_label(conn, sid, 140, 160, 1)
    report = benchmark.compare(sid, labels, result, baseline="HEAD", k=4)
    assert labels.read_bytes() == frozen and len(report["labels"]) == 2
    assert len(db.list_labels(conn, sid)) == 3 and not db.list_candidates(conn, sid, "stream_highlights")
    assert not db.list_candidates(conn, sid) and not db.list_exports(conn)
    classic, before, after = (report["runs"][name] for name in ("classic", "audited_stream", "corrected_stream"))
    assert before == after and classic["top"] and before["top"]
    assert classic["returned_at_k"] == classic["total_candidates"] == 3 and classic["matched_positives"] == 1
    hit = next(p for p in classic["top"] if p["positive_match"])
    assert hit["end_error_s"] == pytest.approx(hit["end"] - 54, abs=.001)
    assert json.loads(result.read_text())["labels_sha256"] == report["labels_sha256"]
