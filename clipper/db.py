"""SQLite store: sources (with rights), candidates, human labels, exports (with duplicate signatures), posts."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable

from clipper import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    title TEXT,
    streamer TEXT,
    language TEXT,
    rights TEXT NOT NULL,
    campaign TEXT,
    duration REAL,
    width INTEGER,
    height INTEGER,
    fps REAL,
    has_audio INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    rank INTEGER,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    peak_s REAL NOT NULL,
    score REAL NOT NULL,
    signals TEXT,
    flags TEXT,
    hook_edited INTEGER NOT NULL DEFAULT 0,
    text TEXT,
    hook TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES sources(id),
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    label INTEGER NOT NULL,
    note TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS exports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER REFERENCES candidates(id),
    source_id INTEGER NOT NULL REFERENCES sources(id),
    platform TEXT NOT NULL,
    path TEXT NOT NULL,
    start_s REAL,
    end_s REAL,
    hook TEXT,
    caption TEXT,
    signature TEXT,
    blocked_reason TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    export_id INTEGER NOT NULL REFERENCES exports(id),
    platform TEXT NOT NULL,
    account TEXT,
    url TEXT,
    campaign TEXT,
    submitted_at TEXT,
    status TEXT,
    views INTEGER,
    earnings REAL
);
CREATE INDEX IF NOT EXISTS idx_candidates_source ON candidates(source_id);
CREATE INDEX IF NOT EXISTS idx_labels_source ON labels(source_id);
CREATE INDEX IF NOT EXISTS idx_exports_platform ON exports(platform);
"""

JSON_FIELDS = ("signals", "flags", "signature")


def connect(cfg: dict) -> sqlite3.Connection:
    path = config.data_dir(cfg) / "clipper.sqlite"
    conn = sqlite3.connect(path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version != 1:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                if version == 0:
                    _initialize(conn)
                elif version != 1:
                    raise ValueError(f"Unsupported database version {version}")
    except Exception:
        conn.close()
        raise
    return conn


def _initialize(conn: sqlite3.Connection) -> None:
    tables = ("sources", "candidates", "labels", "exports", "posts")
    legacy = conn.execute("SELECT 1 FROM sqlite_master WHERE name='sources'").fetchone()
    if legacy:
        sequences = dict(conn.execute("SELECT name, seq FROM sqlite_sequence"))
        for table in tables:
            conn.execute(f"ALTER TABLE {table} RENAME TO legacy_{table}")
    for statement in SCHEMA.split(";"):
        if statement.strip():
            conn.execute(statement)
    if legacy:
        for table in tables:
            columns = ", ".join(row["name"] for row in conn.execute(f"PRAGMA table_info(legacy_{table})"))
            if table == "exports":
                # Old reanalysis deleted candidates. Keep the original export and post IDs.
                conn.execute("""INSERT INTO candidates (id, source_id, start_s, end_s, peak_s, score)
                    SELECT candidate_id, source_id, MIN(start_s), MAX(end_s),
                           (MIN(start_s)+MAX(end_s))/2, 0 FROM legacy_exports
                    WHERE candidate_id IS NOT NULL AND candidate_id NOT IN (SELECT id FROM candidates)
                    GROUP BY candidate_id""")
            conn.execute(f"INSERT INTO {table} ({columns}) SELECT {columns} FROM legacy_{table}")
            if table in sequences:
                conn.execute("UPDATE sqlite_sequence SET seq=MAX(seq, ?) WHERE name=?", (sequences[table], table))
        conn.execute("UPDATE candidates SET hook_edited=1 WHERE hook IS NOT NULL")
        for table in reversed(tables):
            conn.execute(f"DROP TABLE legacy_{table}")
        for statement in SCHEMA.split(";"):
            if statement.strip().startswith("CREATE INDEX"):
                conn.execute(statement)
    conn.execute("PRAGMA user_version = 1")


def _row(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    out = dict(row)
    for key in JSON_FIELDS:
        if key in out and isinstance(out[key], str):
            try:
                out[key] = json.loads(out[key])
            except json.JSONDecodeError:
                pass
    return out


def _rows(rows: Iterable[sqlite3.Row]) -> list[dict]:
    return [_row(r) for r in rows]  # type: ignore[misc]


# --- sources -----------------------------------------------------------------------------------------------------

def add_source(conn: sqlite3.Connection, **fields: Any) -> int:
    fields["rights"] = (fields.get("rights") or "").strip()
    if not fields["rights"]:
        raise ValueError("rights is required: say where your permission to clip this video comes from")
    keys = ",".join(fields)
    marks = ",".join("?" for _ in fields)
    cur = conn.execute(f"INSERT INTO sources ({keys}) VALUES ({marks})", tuple(fields.values()))
    conn.commit()
    return int(cur.lastrowid)


def get_source(conn: sqlite3.Connection, source_id: int) -> dict:
    row = _row(conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone())
    if row is None:
        raise KeyError(f"no source #{source_id}")
    return row


def list_sources(conn: sqlite3.Connection) -> list[dict]:
    return _rows(conn.execute("SELECT * FROM sources ORDER BY id DESC").fetchall())


# --- candidates --------------------------------------------------------------------------------------------------

def replace_candidates(conn: sqlite3.Connection, source_id: int, candidates: list[dict]) -> list[int]:
    """Reuse matching moments and retain historical rows referenced by exports or edited hooks."""
    ids = []
    with conn:
        conn.execute("BEGIN IMMEDIATE")
        previous = _rows(conn.execute("SELECT * FROM candidates WHERE source_id=? ORDER BY id", (source_id,)))
        conn.execute("UPDATE candidates SET rank=NULL WHERE source_id=?", (source_id,))
        for rank, cand in enumerate(candidates, start=1):
            matches = [row for row in previous
                       if row["start_s"] <= cand["peak"] <= row["end_s"]
                       and cand["start"] <= row["peak_s"] <= cand["end"]]
            values = (rank, cand["start"], cand["end"], cand["peak"], cand["score"],
                      json.dumps(cand.get("signals", {})), json.dumps(cand.get("flags", {})), cand.get("text", ""))
            if matches:
                row = min(matches, key=lambda item: abs(item["peak_s"] - cand["peak"]))
                previous.remove(row)
                cid = row["id"]
                conn.execute("UPDATE candidates SET rank=?, start_s=?, end_s=?, peak_s=?, score=?, signals=?, "
                             "flags=?, text=?, hook=CASE WHEN hook_edited THEN hook ELSE ? END WHERE id=?",
                             (*values, cand.get("hook"), cid))
            else:
                cur = conn.execute("INSERT INTO candidates (rank,start_s,end_s,peak_s,score,signals,flags,text,"
                                   "hook,source_id) VALUES (?,?,?,?,?,?,?,?,?,?)", (*values, cand.get("hook"), source_id))
                cid = cur.lastrowid
            ids.append(cid)
        conn.execute("DELETE FROM candidates WHERE source_id=? AND rank IS NULL AND hook_edited=0 AND "
                     "NOT EXISTS (SELECT 1 FROM exports WHERE candidate_id=candidates.id)", (source_id,))
    return ids


def list_candidates(conn: sqlite3.Connection, source_id: int) -> list[dict]:
    return _rows(conn.execute("SELECT * FROM candidates WHERE source_id = ? AND rank IS NOT NULL ORDER BY rank", (source_id,)).fetchall())


def get_candidate(conn: sqlite3.Connection, candidate_id: int) -> dict:
    row = _row(conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone())
    if row is None:
        raise KeyError(f"no candidate #{candidate_id}")
    return row


def set_hook(conn: sqlite3.Connection, candidate_id: int, hook: str | None) -> None:
    conn.execute("UPDATE candidates SET hook = ?, hook_edited=1 WHERE id = ?", (hook or None, candidate_id))
    conn.commit()


# --- labels (ground truth for tuning) ----------------------------------------------------------------------------

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    shorter = max(1e-6, min(a1 - a0, b1 - b0))
    return inter / shorter


def add_label(conn: sqlite3.Connection, source_id: int, start: float, end: float, label: int,
              note: str | None = None) -> int:
    """Store or update a human judgement for a time window (1 = good highlight, 0 = not a highlight)."""
    for row in list_labels(conn, source_id):
        if _overlap(start, end, row["start_s"], row["end_s"]) >= 0.5:
            conn.execute("UPDATE labels SET label = ?, note = COALESCE(?, note), start_s = ?, end_s = ? WHERE id = ?",
                         (int(label), note, start, end, row["id"]))
            conn.commit()
            return int(row["id"])
    cur = conn.execute("INSERT INTO labels (source_id, start_s, end_s, label, note) VALUES (?,?,?,?,?)",
                       (source_id, start, end, int(label), note))
    conn.commit()
    return int(cur.lastrowid)


def clear_label(conn: sqlite3.Connection, source_id: int, start: float, end: float) -> None:
    for row in list_labels(conn, source_id):
        if _overlap(start, end, row["start_s"], row["end_s"]) >= 0.5:
            conn.execute("DELETE FROM labels WHERE id = ?", (row["id"],))
    conn.commit()


def list_labels(conn: sqlite3.Connection, source_id: int) -> list[dict]:
    return _rows(conn.execute("SELECT * FROM labels WHERE source_id = ? ORDER BY start_s", (source_id,)).fetchall())


def label_for(labels: list[dict], start: float, end: float) -> int | None:
    best, best_ov = None, 0.0
    for row in labels:
        ov = _overlap(start, end, row["start_s"], row["end_s"])
        if ov >= 0.5 and ov > best_ov:
            best, best_ov = int(row["label"]), ov
    return best


# --- exports -----------------------------------------------------------------------------------------------------

def add_export(conn: sqlite3.Connection, *, commit: bool = True, **fields: Any) -> int:
    if "signature" in fields and not isinstance(fields["signature"], str):
        fields["signature"] = json.dumps(fields["signature"])
    keys = ",".join(fields)
    marks = ",".join("?" for _ in fields)
    cur = conn.execute(f"INSERT INTO exports ({keys}) VALUES ({marks})", tuple(fields.values()))
    if commit:
        conn.commit()
    return int(cur.lastrowid)


def list_exports(conn: sqlite3.Connection, platform: str | None = None, source_id: int | None = None,
                 candidate_id: int | None = None) -> list[dict]:
    sql, args = "SELECT * FROM exports WHERE 1=1", []
    if platform:
        sql += " AND platform = ?"
        args.append(platform)
    if source_id is not None:
        sql += " AND source_id = ?"
        args.append(source_id)
    if candidate_id is not None:
        sql += " AND candidate_id = ?"
        args.append(candidate_id)
    return _rows(conn.execute(sql + " ORDER BY id", args).fetchall())


def add_post(conn: sqlite3.Connection, export_id: int, url: str, account: str | None = None) -> int:
    """Record that an export was posted by hand (its URL). Posted exports are never replaced or deleted."""
    exp = conn.execute("SELECT e.platform, s.campaign FROM exports e JOIN sources s ON s.id = e.source_id "
                       "WHERE e.id = ?", (export_id,)).fetchone()
    if exp is None:
        raise ValueError(f"no export #{export_id}")
    old = conn.execute("SELECT id FROM posts WHERE export_id = ?", (export_id,)).fetchone()
    if old:
        conn.execute("UPDATE posts SET url = ?, account = COALESCE(?, account) WHERE id = ?", (url, account, old[0]))
        conn.commit()
        return int(old[0])
    cur = conn.execute("INSERT INTO posts (export_id, platform, account, url, campaign, submitted_at, status) "
                       "VALUES (?, ?, ?, ?, ?, datetime('now'), 'posted')", (export_id, exp[0], account, url, exp[1]))
    conn.commit()
    return int(cur.lastrowid)


def post_for(conn: sqlite3.Connection, export_id: int) -> dict | None:
    rows = _rows(conn.execute("SELECT * FROM posts WHERE export_id = ? ORDER BY id DESC LIMIT 1",
                              (export_id,)).fetchall())
    return rows[0] if rows else None
