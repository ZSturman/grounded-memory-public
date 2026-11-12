"""Transactional world storage and non-destructive migration of the visual-memory database."""

import json
import sqlite3
import threading
import time
from contextlib import contextmanager

from ..store import uid

TABLES = (
    "episodes",
    "observations",
    "beliefs",
    "revisions",
    "teachings",
    "questions",
    "events",
    "predictions",
    "playback",
    "exposures",
)


def get(db, table, key, default=None):
    row = db.execute(f"SELECT payload FROM world_{table} WHERE id=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def put(db, table, value):
    db.execute(
        f"INSERT INTO world_{table}(id,payload) VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
        (value["id"], json.dumps(value, allow_nan=False)),
    )
    return value


def rows(db, table, where=None, params=(), limit=None):
    sql = f"SELECT payload FROM world_{table}"
    if where:
        sql += " WHERE " + where
    sql += " ORDER BY rowid DESC" if limit else " ORDER BY rowid"
    if limit:
        sql += " LIMIT ?"
        params = (*params, limit)
    result = [json.loads(r[0]) for r in db.execute(sql, params)]
    return list(reversed(result)) if limit else result


def meta(db, key, default=None):
    row = db.execute("SELECT payload FROM world_meta WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else default


def set_meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO world_meta VALUES (?,?)", (key, json.dumps(value)))


def new_belief(db, *, subject, predicate, value, source, evidence, context=None, belief_id=None, **details):
    belief = dict(
        id=belief_id or uid(),
        subject=subject,
        predicate=predicate,
        value=value,
        source=source,
        context=context,
        status="active",
        certainty="tentative",
        version=1,
        supporting=list(evidence),
        conflicting=[],
        history=[],
        created_at=time.time(),
        **details,
    )
    return put(db, "beliefs", belief)


def change_belief(db, belief, *, operation, reason, evidence, value=None, revision_id=None, source=None):
    before = {k: belief[k] for k in ("value", "status", "certainty", "version")}
    if operation == "correct":
        belief["value"] = value
        belief["status"] = "active"
        belief["certainty"] = "taught"
    elif operation == "retract":
        belief["status"] = "retracted"
    elif operation == "restore":
        belief["status"] = "active"
        belief["certainty"] = "tentative"
    elif operation == "contradict":
        belief["status"] = "contested"
        belief["certainty"] = "tentative"
    belief["version"] += 1
    target = "conflicting" if operation == "contradict" else "supporting"
    belief[target] = list(dict.fromkeys(belief[target] + evidence))
    revision = dict(
        id=revision_id or uid(),
        belief_id=belief["id"],
        operation=operation,
        reason=reason,
        evidence=evidence,
        source=source or {"kind": "observation"},
        before=before,
        after={k: belief[k] for k in before},
        created_at=time.time(),
    )
    belief["history"].append(revision["id"])
    put(db, "revisions", revision)
    put(db, "beliefs", belief)
    # Previously issued forecasts keep their history, but no longer count as tests of this version.
    for prediction in rows(
        db,
        "predictions",
        "json_extract(payload, '$.belief_id')=? AND json_extract(payload, '$.status')='pending'",
        (belief["id"],),
    ):
        if prediction.get("belief_id") == belief["id"] and prediction["status"] == "pending":
            prediction["status"] = "invalidated"
            put(db, "predictions", prediction)
    return belief


class WorldRepository:
    def __init__(self, memory):
        self.memory = memory
        self.lock = threading.RLock()
        with memory.connect() as db:
            version = db.execute("SELECT max(version) FROM schema_version").fetchone()[0]
            if version < 2:
                backup = memory.root / "backups" / "pre-world-v1.sqlite3"
                backup.parent.mkdir(exist_ok=True)
                if not backup.exists():
                    with sqlite3.connect(backup) as target:
                        db.backup(target)
                db.execute("BEGIN IMMEDIATE")
                for table in TABLES:
                    db.execute(f"CREATE TABLE world_{table}(id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                db.execute("CREATE TABLE world_meta(key TEXT PRIMARY KEY, payload TEXT NOT NULL)")
                db.execute(
                    "CREATE TABLE world_receipts(source_key TEXT PRIMARY KEY, observation_id TEXT NOT NULL)"
                )
                db.execute("INSERT INTO schema_version VALUES(2)")
                set_meta(db, "learner_id", uid())
                set_meta(db, "last_asked", -1e12)
                for name, value in (
                    ("continuity", "Nearby observations may describe the same thing"),
                    ("persistence", "Brief occlusion need not mean disappearance"),
                    ("temporal_order", "An effect follows its proposed cause"),
                    ("geometry", "Spatial relations require a shared coordinate frame"),
                ):
                    new_belief(
                        db,
                        subject="world",
                        predicate="prior",
                        value=value,
                        source={"kind": "prior", "id": name},
                        evidence=[],
                        belief_id="prior:" + name,
                    )
            # Added after schema v2; created in place so existing learners keep their history.
            db.execute(
                "CREATE TABLE IF NOT EXISTS world_place_regions(id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS world_place_regions_place ON world_place_regions(json_extract(payload, '$.place_id'))"
            )
            for table in ("observations", "questions", "events", "predictions"):
                db.execute(
                    f"CREATE INDEX IF NOT EXISTS world_{table}_episode ON world_{table}(json_extract(payload, '$.episode_id'))"
                )
            db.execute(
                "CREATE INDEX IF NOT EXISTS world_question_fingerprint ON world_questions(json_extract(payload, '$.fingerprint'))"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS world_question_status ON world_questions(json_extract(payload, '$.status'))"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS world_exposure_context ON world_exposures(json_extract(payload, '$.modality'), json_extract(payload, '$.place_id'))"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS world_prediction_belief ON world_predictions(json_extract(payload, '$.belief_id'), json_extract(payload, '$.status'))"
            )
            self.import_legacy(db)

    @contextmanager
    def transaction(self):
        with self.lock, self.memory.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            yield db

    def import_legacy(self, db):
        """Stable IDs preserve revoked claims and make legacy writes visible during transition."""
        for row in db.execute("SELECT * FROM claims").fetchall():
            key = "legacy:" + row["id"]
            belief = get(db, "beliefs", key)
            if not belief:
                teaching = db.execute(
                    "SELECT text,payload FROM utterances WHERE id=?", (row["utterance_id"],)
                ).fetchone()
                tid = "legacy:" + row["utterance_id"]
                if teaching:
                    put(
                        db,
                        "teachings",
                        dict(
                            id=tid,
                            text=teaching[0],
                            legacy=json.loads(teaching[1]),
                            source={"kind": "human", "id": "legacy-human"},
                        ),
                    )
                belief = new_belief(
                    db,
                    subject=row["subject"],
                    predicate=row["predicate"],
                    value=row["object"],
                    source={"kind": "human", "id": "legacy-human"},
                    evidence=[tid],
                    belief_id=key,
                )
                belief["certainty"] = "taught"
                belief["status"] = "retracted" if row["revoked"] else "active"
                put(db, "beliefs", belief)
            elif row["revoked"] and belief["status"] != "retracted":
                change_belief(
                    db, belief, operation="retract", reason="Legacy claim revoked by teacher", evidence=[key]
                )
