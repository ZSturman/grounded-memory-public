"""Collection never invokes interpretation, inference, or the online learner."""

import json
import sqlite3
from pathlib import Path

from .contracts import Collected
from .storage import digest


def lineage(db, tables, namespace, table, identity, data):
    """Follow teaching evidence back to its capture groups; don't split replies from a session."""
    groups = set()

    def add(payload, sample_id=None):
        group = payload.get("episode_id") or payload.get("session_id")
        if group:
            groups.add(f"{namespace}:group:{group}")
        if payload.get("source_id"):
            groups.add(f"{namespace}:source:{payload['source_id']}")
        if sample_id:
            groups.add(f"{namespace}:sample:{sample_id}")
        evidence = payload.get("evidence")
        evidence = evidence if isinstance(evidence, dict) else {}
        refs = list(payload.get("evidence_refs", []))
        refs += [
            payload.get("evidence_ref"),
            (payload.get("perception") or {}).get("frame"),
            evidence.get("asset"),
            evidence.get("frame"),
        ]
        groups.update(f"{namespace}:asset:{ref}" for ref in refs if ref)

    add(data, identity if table in ("observations", "world_observations") else None)
    if table == "world_teachings":
        if "world_questions" in tables:
            question = db.execute(
                "SELECT payload FROM world_questions WHERE id=?", (data.get("question_id"),)
            ).fetchone()
            if question:
                add(json.loads(question[0]))
        for observation_id in data.get("evidence", []):
            groups.add(f"{namespace}:sample:{observation_id}")
            if "world_observations" in tables:
                observation = db.execute(
                    "SELECT payload FROM world_observations WHERE id=?", (observation_id,)
                ).fetchone()
                if observation:
                    add(json.loads(observation[0]))
    if table == "utterances" and not (data.get("session_id") or data.get("episode_id")):
        # Older teaching often has no session ID. Reserve this unresolved family together.
        groups.add(f"{namespace}:unresolved-legacy-teaching")
    return sorted(groups)


def collect(ledger, record: Collected):
    with ledger.stage("collection"):
        value = record.model_dump()
        value["content_digest"] = digest(record.payload)
        return ledger.append("raw", value, record.id)


SOURCE_KINDS = ("observations", "world_observations", "world_teachings", "utterances")


def import_memory(ledger, source_root, limit=500, sources=None):
    """Incrementally copy retained input records using SQLite read-only mode.

    Existing observation payloads contain derived perception fields; these remain explicitly
    identified as source snapshots, never interpreted as human-approved memory.
    """
    with ledger.stage("collection"):
        path = (Path(source_root) / "memory.sqlite3").resolve()
        namespace = digest(str(path))[:16]
        imported = 0
        db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            db.execute("BEGIN")
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in SOURCE_KINDS:
                if sources is not None and table not in sources:
                    continue
                if table not in tables or imported >= limit:
                    continue
                cursor_key = namespace + ":" + table
                with ledger.connect() as target:
                    row = target.execute(
                        "SELECT position FROM cursors WHERE source=?", (cursor_key,)
                    ).fetchone()
                position = row[0] if row else 0
                for rowid, identity, payload in db.execute(
                    f"SELECT rowid,id,payload FROM {table} WHERE rowid>? ORDER BY rowid LIMIT ?",
                    (position, limit - imported),
                ):
                    data = json.loads(payload)
                    # Legacy utterance text is stored separately from its metadata.
                    if table == "utterances":
                        data = dict(
                            data,
                            text=db.execute("SELECT text FROM utterances WHERE id=?", (identity,)).fetchone()[
                                0
                            ],
                        )
                    group = data.get("episode_id") or data.get("session_id")
                    source = data.get("source_id") or group or identity
                    origin = "human" if table in ("world_teachings", "utterances") else "sensor"
                    if data.get("origin") == "model_hypothesis":
                        origin = "model"
                    if data.get("origin") == "system_playback" or data.get("excluded"):
                        origin = "derived"
                    record = Collected(
                        id=f"{cursor_key}:{identity}",
                        source_id=f"{namespace}:{source}",
                        source_kind=table,
                        source_timestamp=data.get(
                            "timestamp", data.get("media_time", data.get("created_at", 0))
                        ),
                        group_ids=sorted(
                            set(
                                [f"{namespace}:group:{group or source}", f"{namespace}:source:{source}"]
                                + lineage(db, tables, namespace, table, identity, data)
                            )
                        ),
                        origin=origin,
                        payload=data,
                    )
                    # Artifact and checkpoint advance in the same transaction; failures are retryable.
                    with ledger.connect() as target:
                        ledger.append(
                            "raw", dict(record.model_dump(), content_digest=digest(data)), record.id, target
                        )
                        target.execute(
                            "INSERT INTO cursors VALUES (?,?) ON CONFLICT(source) DO UPDATE SET position=max(position,excluded.position)",
                            (cursor_key, rowid),
                        )
                    imported += 1
        finally:
            db.close()
        result = dict(imported=imported, limit=limit, more_may_remain=imported == limit)
        ledger.append(
            "collection_run", dict(**result, sources=list(sources if sources is not None else SOURCE_KINDS))
        )
        return result
