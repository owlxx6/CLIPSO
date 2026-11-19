from clipper import db, detect, evaluate, fixture, pipeline


def test_detection_finds_the_three_highlights(cfg, source, fx):
    conn, sid = source
    rows = pipeline.analyze(conn, cfg, sid, transcribe=False)
    top = rows[:3]
    for b in fixture.BURSTS:
        assert any(abs(r["peak"] - b) <= 6 for r in top), f"no top-3 candidate near {b}s: {[r['peak'] for r in top]}"
    for r in rows:
        assert 15 - 0.01 <= r["end"] - r["start"] <= 34 + 0.01
        assert r["start"] < r["peak"] < r["end"]
    m = evaluate.evaluate(rows, evaluate.load_truth(fx["truth"]), k=3)
    assert m["precision"] == 1.0 and m["recall"] == 1.0


def test_candidates_do_not_overlap_much(cfg, source):
    conn, sid = source
    rows = pipeline.analyze(conn, cfg, sid, transcribe=False)
    cands = [detect.Candidate(r["start"], r["end"], r["peak"], r["score"]) for r in rows]
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            assert detect._overlap(a, b) <= cfg["detection"]["max_overlap"] + 1e-6


def test_labels_survive_reanalysis(cfg, source):
    conn, sid = source
    pipeline.analyze(conn, cfg, sid, transcribe=False)
    first = db.list_candidates(conn, sid)[0]
    db.add_label(conn, sid, first["start_s"], first["end_s"], 1)
    pipeline.analyze(conn, cfg, sid, transcribe=False)
    again = db.list_candidates(conn, sid)[0]
    assert db.label_for(db.list_labels(conn, sid), again["start_s"], again["end_s"]) == 1


def test_tune_never_does_worse_than_current_weights(cfg, source, fx):
    conn, sid = source
    src = db.get_source(conn, sid)
    sig, tdata = pipeline.compute_signals(cfg, src, transcribe=False)
    res = evaluate.tune(sig, cfg, src["duration"], tdata, evaluate.load_truth(fx["truth"]), k=3, trials=40)
    assert res["metrics"]["f1"] >= res["baseline"]["f1"]
    assert set(res["signals"]) >= {"audio", "chat", "viewer_clips", "keywords"}


def test_fit_bounds_respects_target():
    s, e = detect.fit_bounds(100, 104, 102, 600, 15, 34)
    assert 15 <= e - s <= 34 and s <= 102 <= e
    s, e = detect.fit_bounds(50, 120, 100, 600, 15, 34)
    assert e - s <= 34.01 and s <= 100 <= e
