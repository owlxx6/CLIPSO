import sqlite3

import pytest

from clipper import config, db


def test_migrate_orphaned_export_and_keep_post(cfg):
    path = config.data_dir(cfg) / "clipper.sqlite"
    with sqlite3.connect(path) as old:
        old.executescript("""
            CREATE TABLE sources(id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT, rights TEXT);
            CREATE TABLE candidates(id INTEGER PRIMARY KEY AUTOINCREMENT, source_id INTEGER, rank INTEGER,
                                    start_s REAL, end_s REAL, peak_s REAL, score REAL, hook TEXT);
            CREATE TABLE labels(id INTEGER PRIMARY KEY AUTOINCREMENT, source_id INTEGER, start_s REAL,
                                end_s REAL, label INTEGER);
            CREATE TABLE exports(id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id INTEGER, source_id INTEGER,
                                 platform TEXT, path TEXT, start_s REAL, end_s REAL);
            CREATE TABLE posts(id INTEGER PRIMARY KEY AUTOINCREMENT, export_id INTEGER, platform TEXT);
            INSERT INTO sources VALUES (1,'fixture','permission');
            INSERT INTO exports VALUES (9,42,1,'tiktok','fixture.mp4',5,19);
            INSERT INTO posts VALUES (17,9,'tiktok');
        """)
    old.close()
    conn = db.connect(cfg)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert db.post_for(conn, 9)["id"] == 17
    assert db.get_candidate(conn, 42)["rank"] is None
    ids = db.replace_candidates(conn, 1, [dict(start=4, end=20, peak=12, score=1)])
    assert ids == [42]
    with pytest.raises(sqlite3.IntegrityError), conn:
        conn.execute("DELETE FROM exports WHERE id=9")
    conn.close()


def test_unknown_database_version_remains_intact(cfg):
    path = config.data_dir(cfg) / "clipper.sqlite"
    with sqlite3.connect(path) as old:
        old.execute("PRAGMA user_version=42")
    old.close()
    with pytest.raises(ValueError, match="42"):
        db.connect(cfg)
    with sqlite3.connect(path) as check:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 42
    check.close()

def test_imported_transcript_survives_force(source, cfg):
    from clipper import pipeline

    conn, sid = source
    src = db.get_source(conn, sid)
    imported = pipeline.get_transcript(cfg, src, transcribe=False)
    assert imported and pipeline.get_transcript(cfg, src, force=True) == imported


def test_campaign_disclosure_override(source, cfg):
    from clipper import pipeline

    conn, sid = source
    conn.execute("UPDATE sources SET rights='content-rewards:fixture' WHERE id=?", (sid,))
    conn.commit()
    pipeline.analyze(conn, cfg, sid, transcribe=False)
    cid = db.list_candidates(conn, sid)[0]["id"]
    result, = pipeline.render_candidates(conn, cfg, sid, [cid], ["tiktok"], disclosure=False)
    assert result["disclosure"] == "PUBLICITÉ"
