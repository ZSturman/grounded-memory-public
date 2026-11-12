"""Workflow visibility without starting background work or reading private media."""

import json

from .deployment import active
from .labeling import latest

KINDS = {
    "raw",
    "annotation",
    "dataset",
    "experiment",
    "evaluation",
    "deployment",
    "failure",
    "policy",
    "collection_run",
    "triage",
}


def inventory(ledger, kind, limit=25, offset=0):
    if kind not in KINDS:
        raise ValueError("Unknown pipeline stage")
    items = ledger.list(kind, limit, offset)
    if kind == "raw":
        return [
            dict(
                id=r["id"],
                source_id=r["source_id"],
                source_kind=r["source_kind"],
                origin=r["origin"],
                source_timestamp=r["source_timestamp"],
                annotation=latest(ledger, r["id"]),
                allowed=ledger.allowed(r["id"]),
            )
            for r in items
        ]
    if kind == "dataset":
        return [
            dict(
                id=r["id"],
                name=r["spec"]["name"],
                purpose=r["spec"]["purpose"],
                records=len(r["records"]),
                cases=len(r["spec"]["cases"]),
                warnings=r["warnings"],
                splits={
                    split: sum(item["split"] == split for item in r["records"])
                    for split in ("train", "validation", "test", "regression")
                },
            )
            for r in items
        ]
    if kind == "evaluation":
        return [
            {
                k: v
                for k, v in r.items()
                if k not in ("candidate", "baseline", "deployment_candidate", "deployment_baseline")
            }
            | dict(
                candidate=r["candidate"]["metrics"],
                baseline=r["baseline"]["metrics"],
                deployment_candidate=r.get("deployment_candidate", {}).get("metrics"),
            )
            for r in items
        ]
    return items


def status(ledger):
    with ledger.connect() as db:
        counts = {r[0]: r[1] for r in db.execute("SELECT kind,count(*) FROM artifacts GROUP BY kind")}
        reviewed = db.execute("""
            SELECT count(*) FROM artifacts AS a WHERE a.kind='annotation'
            AND json_extract(a.payload,'$.reviewed')=1 AND a.seq=(
              SELECT max(b.seq) FROM artifacts AS b WHERE b.kind='annotation'
              AND json_extract(b.payload,'$.record_id')=json_extract(a.payload,'$.record_id'))
        """).fetchone()[0]
        cursors = [dict(r) for r in db.execute("SELECT * FROM cursors ORDER BY source")]
        failures = [
            json.loads(r[0])
            for r in db.execute(
                "SELECT payload FROM artifacts WHERE kind='failure' ORDER BY seq DESC LIMIT 5"
            )
        ]
    return dict(
        counts=counts,
        awaiting_review=counts.get("raw", 0) - reviewed,
        reviewed=reviewed,
        deployed=active(ledger),
        cursors=cursors,
        recent_failures=failures,
        scope="Reviewed recall; live world learning and perception retain their existing configuration.",
        automation="Collection and labeling run explicitly; no background training or promotion.",
    )
