"""Append-only pipeline ledger, physically separate from production memory.sqlite3."""

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def code_version():
    """Fingerprint actual source bytes, including uncommitted changes, not just Git HEAD."""
    root = Path(__file__).resolve().parents[1]
    return digest(
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))
        }
    )


class Ledger:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS artifacts(
                    seq INTEGER PRIMARY KEY, kind TEXT NOT NULL, id TEXT UNIQUE NOT NULL,
                    created_at REAL NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS artifacts_kind ON artifacts(kind,seq);
                CREATE INDEX IF NOT EXISTS annotation_record ON artifacts(kind,json_extract(payload,'$.record_id'),seq);
                CREATE TABLE IF NOT EXISTS cursors(source TEXT PRIMARY KEY, position INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS split_groups(key TEXT PRIMARY KEY, split TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS policies(record_id TEXT PRIMARY KEY, allowed INTEGER NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / "pipeline.sqlite3", timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def append(self, kind, value, artifact_id=None, db=None):
        identity = artifact_id or kind + ":" + uuid.uuid4().hex
        payload = dict(value, id=identity)
        if db is None:
            with self.connect() as connection:
                return self.append(kind, value, identity, connection)
        previous = db.execute("SELECT payload FROM artifacts WHERE id=?", (identity,)).fetchone()
        if previous:
            if previous[0] != canonical(payload):
                raise ValueError("An immutable artifact ID was reused with different content")
            return payload
        db.execute(
            "INSERT INTO artifacts(kind,id,created_at,payload) VALUES (?,?,?,?)",
            (kind, identity, time.time(), canonical(payload)),
        )
        return payload

    def get(self, identity, kind=None):
        with self.connect() as db:
            row = db.execute("SELECT kind,payload FROM artifacts WHERE id=?", (identity,)).fetchone()
        if not row or (kind and row["kind"] != kind):
            raise KeyError(identity)
        return json.loads(row["payload"])

    def list(self, kind, limit=None, offset=0):
        with self.connect() as db:
            return [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT payload FROM artifacts WHERE kind=? ORDER BY seq DESC LIMIT ? OFFSET ?",
                    (kind, limit if limit is not None else -1, offset),
                )
            ]

    def allowed(self, identity):
        with self.connect() as db:
            row = db.execute("SELECT allowed FROM policies WHERE record_id=?", (identity,)).fetchone()
        return not row or bool(row[0])

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self.append(
                "failure",
                dict(
                    stage=name,
                    error_type=type(exc).__name__,
                    message=str(exc)[:500],
                    elapsed_seconds=time.perf_counter() - start,
                ),
            )
            raise
