"""Transactional memory. Names label entities; they never identify them."""

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

RELATIONS = ("on", "above", "inside", "next_to", "part_of", "located_in", "belongs_to")


def uid():
    return uuid.uuid4().hex


class MemoryRepository:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "assets").mkdir(exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS schema_version(version INTEGER PRIMARY KEY);
            INSERT OR IGNORE INTO schema_version VALUES(1);
            CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS entities(id TEXT PRIMARY KEY, kind TEXT, label TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS claims(id TEXT PRIMARY KEY, subject TEXT, predicate TEXT,
              object TEXT, utterance_id TEXT NOT NULL, revoked INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS utterances(id TEXT PRIMARY KEY, text TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY, session_id TEXT,
              source_id TEXT, media_time REAL, payload TEXT);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, session_id TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS sensors(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS pending_questions(id TEXT PRIMARY KEY, session_id TEXT, status TEXT,
              created_at REAL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observed_relations(id TEXT PRIMARY KEY, session_id TEXT, subject TEXT,
              predicate TEXT, object TEXT, support INTEGER, payload TEXT);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / "memory.sqlite3", timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def save_session(self, session):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?)", (session["id"], json.dumps(session)))
        return session

    def session(self, sid):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row:
            raise KeyError(sid)
        return json.loads(row[0])

    def sessions(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT payload FROM sessions ORDER BY rowid DESC")]

    def event(self, sid, kind, **values):
        with self.connect() as db:
            db.execute(
                "INSERT INTO events(session_id,payload) VALUES (?,?)",
                (sid, json.dumps(dict(kind=kind, received_at=time.time(), **values))),
            )

    def observe(self, envelope):
        with self.connect() as db:
            cur = db.execute(
                "INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?)",
                (
                    envelope["id"],
                    envelope["session_id"],
                    envelope["source_id"],
                    envelope["media_time"],
                    json.dumps(envelope),
                ),
            )
            return cur.rowcount == 1

    def observations(self, sid):
        with self.connect() as db:
            return [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT payload FROM observations WHERE session_id=? ORDER BY media_time", (sid,)
                )
            ]

    def register_sensor(self, descriptor):
        """Stable id for a capture device description. Identity fields only; per-frame values excluded."""
        identity = {k: v for k, v in sorted(descriptor.items()) if k not in ("intrinsics", "transform")}
        sid = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO sensors VALUES (?,?)", (sid, json.dumps(identity)))
        return sid

    def sensors(self):
        with self.connect() as db:
            return {r[0]: json.loads(r[1]) for r in db.execute("SELECT id, payload FROM sensors")}

    def observe_relation(self, session_id, subject, predicate, obj, evidence):
        """Count one more frame supporting a geometric relation; never a human claim."""
        rid = hashlib.sha256(f"{session_id}:{subject}:{predicate}:{obj}".encode()).hexdigest()[:16]
        with self.connect() as db:
            row = db.execute("SELECT payload FROM observed_relations WHERE id=?", (rid,)).fetchone()
            payload = json.loads(row[0]) if row else {"evidence": []}
            payload["evidence"] = (payload["evidence"] + [evidence])[-5:]
            db.execute(
                "INSERT INTO observed_relations VALUES (?,?,?,?,?,1,?) ON CONFLICT(id) DO UPDATE SET "
                "support=support+1, payload=excluded.payload",
                (rid, session_id, subject, predicate, obj, json.dumps(payload)),
            )
        return rid

    def observed_relations(self, session_id=None):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM observed_relations" + (" WHERE session_id=?" if session_id else ""),
                (session_id,) if session_id else (),
            ).fetchall()
        return [dict(r, payload=json.loads(r["payload"])) for r in rows]

    def confirm_relation(self, relation_id, text):
        """The person confirmed an observed relation; only now does it become a claim."""
        with self.connect() as db:
            row = db.execute("SELECT * FROM observed_relations WHERE id=?", (relation_id,)).fetchone()
            if not row:
                raise KeyError(relation_id)
            utterance_id, claim_id = uid(), uid()
            source = {"source": "human", "confirmed_observation": relation_id}
            db.execute("INSERT INTO utterances VALUES (?,?,?)", (utterance_id, text, json.dumps(source)))
            db.execute(
                "INSERT INTO claims VALUES (?,?,?,?,?,0)",
                (claim_id, row["subject"], row["predicate"], row["object"], utterance_id),
            )
        return claim_id

    def defer_question(self, session_id, question, status, reason):
        """Keep an unanswered question: `open` to answer later, `dismissed` as not important."""
        record = dict(question, status=status, reason=reason, session_id=session_id)
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO pending_questions VALUES (?,?,?,?,?)",
                (question["id"], session_id, status, time.time(), json.dumps(record)),
            )

    def pending_questions(self, statuses=("open",), place_id=None):
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload FROM pending_questions WHERE status IN (%s) ORDER BY created_at DESC"
                % ",".join("?" * len(statuses)),
                tuple(statuses),
            ).fetchall()
        items = [json.loads(r[0]) for r in rows]
        return [q for q in items if place_id is None or q["context"]["place_id"] == place_id]

    def pending_question(self, question_id):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM pending_questions WHERE id=?", (question_id,)).fetchone()
        if not row:
            raise KeyError(question_id)
        return json.loads(row[0])

    def settle_question(self, question_id, status, **details):
        record = dict(self.pending_question(question_id), status=status, **details)
        with self.connect() as db:
            db.execute(
                "UPDATE pending_questions SET status=?, payload=? WHERE id=?",
                (status, json.dumps(record), question_id),
            )

    def entities(self, kind=None):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM entities" + (" WHERE kind=?" if kind else ""), (kind,) if kind else ()
            ).fetchall()
            claims = [dict(r) for r in db.execute("SELECT * FROM claims WHERE revoked=0")]
        return [
            dict(
                id=r["id"],
                kind=r["kind"],
                label=r["label"],
                **json.loads(r["payload"]),
                claims=[c for c in claims if c["subject"] == r["id"]],
            )
            for r in rows
        ]

    def teach(
        self,
        *,
        kind,
        text,
        label,
        evidence,
        resolution=None,
        previous_id=None,
        entity_id=None,
        categories=(),
        aliases=(),
        relations=(),
    ):
        # Parsed model output alone is insufficient. Every new term must occur in human evidence.
        words = [label, *categories, *aliases]
        if (
            kind not in ("object", "place")
            or not label.strip()
            or any(not w.strip() or w.casefold() not in text.casefold() for w in words)
        ):
            raise ValueError("Every name/category must be explicitly present in your answer.")
        for relation in relations:
            if relation["predicate"] not in RELATIONS:
                raise ValueError("Unsupported relationship")
            if relation["quote"].casefold() not in text.casefold() or not relation["quote"].strip():
                raise ValueError("A relationship requires quoted human evidence.")
        matches = [e for e in self.entities(kind) if e["label"].casefold() == label.casefold()]
        if matches and not resolution and not entity_id:
            return dict(needs_clarification=True, matches=matches)
        if resolution not in (None, "same", "same_kind", "subtype", "unrelated"):
            raise ValueError("Unknown identity choice")
        if resolution == "same":
            if previous_id not in [e["id"] for e in self.entities(kind)]:
                raise ValueError("Select an existing entity of this kind.")
            entity_id = previous_id
        entity_id = entity_id or uid()
        utterance_id = uid()
        with self.connect() as db:
            existing = db.execute("SELECT payload,kind FROM entities WHERE id=?", (entity_id,)).fetchone()
            if existing and existing["kind"] != kind:
                raise ValueError("Entity kind cannot change")
            payload = json.loads(existing["payload"]) if existing else {"examples": []}
            if evidence not in payload["examples"]:
                payload["examples"].append(evidence)
            db.execute(
                "INSERT INTO utterances VALUES (?,?,?)",
                (utterance_id, text, json.dumps({"source": "human", "evidence": evidence})),
            )
            db.execute(
                "INSERT OR REPLACE INTO entities VALUES (?,?,?,?)",
                (entity_id, kind, label, json.dumps(payload)),
            )
            facts = [
                ("named", label),
                *[("alias", a) for a in aliases],
            ]

            def category(term):
                row = db.execute(
                    "SELECT id FROM entities WHERE kind=? AND label=?", (kind + "_category", term)
                ).fetchone()
                if row:
                    return row[0]
                cid = uid()
                db.execute(
                    "INSERT INTO entities VALUES (?,?,?,?)",
                    (cid, kind + "_category", term, json.dumps({"examples": []})),
                )
                return cid

            for term in categories:
                facts.append(("instance_of", category(term)))
            if resolution in ("same_kind", "subtype"):
                previous = db.execute(
                    "SELECT label FROM entities WHERE id=? AND kind=?", (previous_id, kind)
                ).fetchone()
                if not previous:
                    raise ValueError("Select a previous entity to establish the shared category.")
                shared = category(previous[0])
                if previous[0].casefold() not in text.casefold():
                    raise ValueError("Include the shared category name in your answer.")
                db.execute(
                    "INSERT INTO claims VALUES (?,?,?,?,?,0)",
                    (uid(), previous_id, "instance_of", shared, utterance_id),
                )
                facts.append(("instance_of", shared))
                if resolution == "subtype" and label.casefold() != previous[0].casefold():
                    specific = category(label)
                    facts.append(("instance_of", specific))
                    db.execute(
                        "INSERT INTO claims VALUES (?,?,?,?,?,0)",
                        (uid(), specific, "subtype_of", shared, utterance_id),
                    )
            if resolution == "same":
                facts.append(("identity_confirmed", previous_id))
            facts.extend((r["predicate"], r["target"]) for r in relations)
            for predicate, obj in facts:
                db.execute(
                    "INSERT INTO claims VALUES (?,?,?,?,?,0)",
                    (uid(), entity_id, predicate, obj, utterance_id),
                )
        return dict(entity_id=entity_id, utterance_id=utterance_id, needs_clarification=False)

    def revoke(self, claim_id):
        with self.connect() as db:
            claim = db.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
            if not claim:
                raise KeyError(claim_id)
            if not db.execute("UPDATE claims SET revoked=1 WHERE id=?", (claim_id,)).rowcount:
                raise KeyError(claim_id)
            if claim["predicate"] == "named":
                remaining = db.execute(
                    "SELECT object FROM claims WHERE subject=? AND predicate='named' AND revoked=0 ORDER BY rowid DESC LIMIT 1",
                    (claim["subject"],),
                ).fetchone()
                db.execute(
                    "UPDATE entities SET label=? WHERE id=?",
                    (remaining[0] if remaining else "Unlabeled entity", claim["subject"]),
                )
            if claim["predicate"] == "identity_confirmed":
                utterance = db.execute(
                    "SELECT payload FROM utterances WHERE id=?", (claim["utterance_id"],)
                ).fetchone()
                entity = db.execute("SELECT payload FROM entities WHERE id=?", (claim["subject"],)).fetchone()
                if utterance and entity:
                    evidence = json.loads(utterance[0])["evidence"]
                    payload = json.loads(entity[0])
                    payload["examples"] = [e for e in payload["examples"] if e != evidence]
                    db.execute(
                        "UPDATE entities SET payload=? WHERE id=?", (json.dumps(payload), claim["subject"])
                    )
