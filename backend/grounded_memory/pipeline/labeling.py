"""Versioned annotations do not mutate their source or assert machine certainty."""

import json

from .contracts import Annotation
from .storage import canonical, digest


def latest(ledger, record_id):
    with ledger.connect() as db:
        row = db.execute(
            "SELECT payload FROM artifacts WHERE kind='annotation' "
            "AND json_extract(payload,'$.record_id')=? ORDER BY seq DESC LIMIT 1",
            (record_id,),
        ).fetchone()
    return json.loads(row[0]) if row else None


def annotate(ledger, record_id, annotation: Annotation):
    with ledger.stage("labeling"):
        ledger.get(record_id, "raw")
        with ledger.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload FROM artifacts WHERE kind='annotation' "
                "AND json_extract(payload,'$.record_id')=? ORDER BY seq DESC LIMIT 1",
                (record_id,),
            ).fetchone()
            old = json.loads(row[0]) if row else None
            if annotation.previous_revision != (old["id"] if old else None):
                raise ValueError("Annotation changed; read its latest revision before saving")
            value = dict(record_id=record_id, **annotation.model_dump())
            return ledger.append("annotation", value, "annotation:" + digest(value), db=db)


def suggest(ledger, record_id):
    """Cheap, deterministic metadata/text extraction. A suggestion is never reviewed truth."""
    raw = ledger.get(record_id, "raw")
    old = latest(ledger, record_id)
    if old:
        return old
    text = raw["payload"].get("text", "")
    return annotate(
        ledger,
        record_id,
        Annotation(
            actor="metadata-v1",
            method="deterministic",
            text=text if isinstance(text, str) else "",
            notes="Source snapshot " + raw["source_kind"] + "; human review and fact keys required.",
        ),
    )


def set_policy(ledger, record_id, allowed, actor):
    with ledger.stage("privacy"):
        ledger.get(record_id, "raw")
        with ledger.connect() as db:
            ledger.append("policy", dict(record_id=record_id, allowed=allowed, actor=actor), db=db)
            db.execute(
                "INSERT INTO policies VALUES (?,?) ON CONFLICT(record_id) DO UPDATE SET allowed=excluded.allowed",
                (record_id, int(allowed)),
            )


def usable(ledger, dataset):
    """Fail closed for stale labels or exclusions, including historical experiment snapshots."""
    for record in dataset["records"]:
        current = latest(ledger, record["id"])
        if not ledger.allowed(record["id"]) or not current or current["id"] != record["annotation_id"]:
            raise ValueError("Dataset has excluded or revised sources; prepare and evaluate a new version")


def normalized_text(text):
    return " ".join(text.casefold().split())


def group_keys(raw, label):
    return sorted(
        set(
            raw["group_ids"]
            + [
                "source:" + raw["source_id"],
                "content:" + raw["content_digest"],
                "text:" + digest(normalized_text(label["text"])),
            ]
            + ["fact:" + canonical(key) for key in label["fact_keys"]]
        )
    )
