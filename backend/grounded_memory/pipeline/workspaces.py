"""Guided app read models and workflow helpers. No world-learning side effects."""

import json
import sqlite3
from pathlib import Path

from . import collection, deployment, labeling, service
from .storage import code_version, digest

# Latest annotations and triage are joined before filtering/pagination, so completed
# records never crowd pending work out of a limited client-side list.
RECORDS = """
 SELECT r.seq, r.created_at, r.payload,
        a.payload AS annotation, t.payload AS triage, coalesce(p.allowed,1) AS allowed
 FROM artifacts r
 LEFT JOIN artifacts a ON a.seq=(SELECT max(x.seq) FROM artifacts x
   WHERE x.kind='annotation' AND json_extract(x.payload,'$.record_id')=r.id)
 LEFT JOIN artifacts t ON t.seq=(SELECT max(x.seq) FROM artifacts x
   WHERE x.kind='triage' AND json_extract(x.payload,'$.record_id')=r.id)
 LEFT JOIN policies p ON p.record_id=r.id
 WHERE r.kind='raw'
"""


def records(ledger, state="pending", source="", search="", order="priority", limit=30, offset=0):
    filters, args = [], []
    reviewed = "coalesce(json_extract(a.payload,'$.reviewed'),0)"
    triage = "coalesce(json_extract(t.payload,'$.action'),'')"
    if state == "pending":
        filters += [f"{reviewed}=0", f"{triage}!='skip'"]
    elif state == "reviewed":
        filters += [f"{reviewed}=1"]
    elif state == "skipped":
        filters += [f"{triage}='skip'"]
    elif state == "flagged":
        filters += [f"{triage}='flag'"]
    elif state == "excluded":
        filters += ["(coalesce(p.allowed,1)=0 OR json_extract(a.payload,'$.status')='excluded')"]
    elif state == "eligible":
        filters += [
            f"{reviewed}=1",
            "coalesce(p.allowed,1)=1",
            "json_extract(a.payload,'$.status')!='excluded'",
            "json_extract(r.payload,'$.origin') IN ('human','sensor')",
        ]
    elif state != "all":
        raise ValueError("Unknown review filter")
    if source:
        filters.append("json_extract(r.payload,'$.source_kind')=?")
        args.append(source)
    if search:
        filters.append("(r.id LIKE ? OR json_extract(r.payload,'$.payload') LIKE ? OR a.payload LIKE ?)")
        args += ["%" + search + "%"] * 3
    sort = {
        "newest": "r.seq DESC",
        "oldest": "r.seq ASC",
        "priority": f"({triage}='flag') DESC, (json_extract(a.payload,'$.status')='contested') DESC, r.seq ASC",
    }.get(order)
    if not sort:
        raise ValueError("Unknown review order")
    query = RECORDS + (" AND " + " AND ".join(filters) if filters else "")
    with ledger.connect() as db:
        total = db.execute("SELECT count(*) FROM (" + query + ")", args).fetchone()[0]
        rows = db.execute(query + " ORDER BY " + sort + " LIMIT ? OFFSET ?", [*args, limit, offset])
        items = []
        for row in rows:
            raw = json.loads(row["payload"])
            annotation = json.loads(row["annotation"]) if row["annotation"] else None
            text = raw["payload"].get("text", "")
            items.append(
                dict(
                    **{k: v for k, v in raw.items() if k != "payload"},
                    preview=(annotation or {}).get("text") or (text if isinstance(text, str) else ""),
                    annotation=annotation,
                    triage=json.loads(row["triage"]) if row["triage"] else None,
                    allowed=bool(row["allowed"]),
                    collected_at=row["created_at"],
                )
            )
    return dict(items=items, total=total, limit=limit, offset=offset)


def triage(ledger, record_ids, action, actor, notes=""):
    # One transaction, and validate every ID before appending any batch decision.
    for identity in record_ids:
        ledger.get(identity, "raw")
    with ledger.connect() as db:
        return [
            ledger.append("triage", dict(record_id=identity, action=action, actor=actor, notes=notes), db=db)
            for identity in dict.fromkeys(record_ids)
        ]


def source_settings(ledger):
    rows = ledger.list("collection_settings", 1)
    return rows[0]["sources"] if rows else list(collection.SOURCE_KINDS)


def configure_sources(ledger, sources):
    if not set(sources) <= set(collection.SOURCE_KINDS):
        raise ValueError("Unknown saved-input source")
    return ledger.append("collection_settings", dict(sources=sorted(set(sources))))


def collection_status(ledger, source_root):
    path = Path(source_root) / "memory.sqlite3"
    enabled = source_settings(ledger)
    sources, source_error = [], None
    namespace = digest(str(path.resolve()))[:16]
    with ledger.connect() as db:
        cursors = {r[0]: r[1] for r in db.execute("SELECT source,position FROM cursors")}
    try:
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for kind in collection.SOURCE_KINDS:
                pending = (
                    db.execute(
                        f"SELECT count(*) FROM {kind} WHERE rowid>?",
                        (cursors.get(namespace + ":" + kind, 0),),
                    ).fetchone()[0]
                    if kind in tables
                    else 0
                )
                sources.append(
                    dict(kind=kind, available=kind in tables, enabled=kind in enabled, pending=pending)
                )
        finally:
            db.close()
    except sqlite3.Error as exc:
        source_error = str(exc)
    with ledger.connect() as db:
        activity = [
            dict(kind=r[0], created_at=r[1], id=r[2])
            for r in db.execute("SELECT kind,created_at,id FROM artifacts ORDER BY seq DESC LIMIT 12")
        ]
    return dict(
        sources=sources,
        source_error=source_error,
        ledger_bytes=sum(p.stat().st_size for p in ledger.root.glob("pipeline.sqlite3*") if p.is_file()),
        storage_scope="Pipeline ledger only; original recordings stay in capture storage.",
        recent_runs=ledger.list("collection_run", 10),
        activity=activity,
        **service.status(ledger),
    )


def dataset_plan(ledger, record_ids):
    """Group connected evidence; respect prior split ownership. Never invent gold answers."""
    groups = []
    with ledger.connect() as db:
        owners = dict(db.execute("SELECT key,split FROM split_groups"))
    for identity in dict.fromkeys(record_ids):
        raw, label = ledger.get(identity, "raw"), labeling.latest(ledger, identity)
        if (
            not label
            or not label["reviewed"]
            or label["status"] == "excluded"
            or not ledger.allowed(identity)
            or raw["origin"] not in ("human", "sensor")
        ):
            raise ValueError("Select only eligible human-reviewed records")
        keys = set(labeling.group_keys(raw, label))
        merged = dict(ids=[identity], keys=keys)
        others = []
        for group in groups:
            if group["keys"] & merged["keys"]:
                merged["ids"] += group["ids"]
                merged["keys"] |= group["keys"]
            else:
                others.append(group)
        groups = others + [merged]
    result = []
    for group in groups:
        splits = {owners[k] for k in group["keys"] if k in owners}
        if len(splits) > 1:
            raise ValueError("These records bridge previously separated splits; inspect their lineage")
        result.append(
            dict(record_ids=sorted(group["ids"]), split=next(iter(splits), "validation"), locked=bool(splits))
        )
    return dict(
        groups=result,
        explanation="Related sources, facts and duplicates stay together. New groups default to validation. Write independent questions and expected evidence before preparing.",
    )


def readiness(ledger, experiment_id):
    evaluations = [e for e in ledger.list("evaluation") if e["experiment_id"] == experiment_id]
    chosen = {}
    for result in evaluations:
        if result["split"] in ("test", "regression"):
            chosen.setdefault(result["split"], result["id"])
    ids = list(chosen.values())
    comparisons = [
        dict(split=e["split"], candidate=e["candidate"]["metrics"], baseline=e["baseline"]["metrics"])
        for e in evaluations
        if e["id"] in ids
    ]
    try:
        deployment.validate_promotion(ledger, experiment_id, ids)
        return dict(
            ready=True,
            evaluation_ids=ids,
            comparisons=comparisons,
            reason="Both final suites pass. Explicit approval is required.",
        )
    except (ValueError, KeyError) as exc:
        return dict(ready=False, evaluation_ids=ids, comparisons=comparisons, reason=str(exc))


def health(ledger):
    current = deployment.active(ledger)
    if not current:
        return dict(healthy=False, reason="No reviewed-recall deployment yet.")
    try:
        if current["code_version"] != code_version():
            raise ValueError("Code changed. Create and evaluate a new candidate.")
        labeling.usable(ledger, ledger.get(current["dataset_id"], "dataset"))
        return dict(healthy=True, reason="Reviewed recall is available.")
    except (ValueError, KeyError) as exc:
        return dict(healthy=False, reason=str(exc))
