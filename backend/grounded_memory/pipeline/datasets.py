"""Reproducible, reviewed snapshots with persistent split ownership."""

from .contracts import DatasetSpec
from .labeling import group_keys, latest, normalized_text
from .storage import digest

TRANSFORM_VERSION = "reviewed-memory-v1"


def prepare(ledger, spec: DatasetSpec):
    with ledger.stage("dataset"):
        records, groups, warnings = [], {}, []

        def assign(keys, split):
            for key in keys:
                if key in groups and groups[key] != split:
                    raise ValueError(
                        "Leakage: a source, conversation, document, duplicate, or fact crosses splits"
                    )
                groups[key] = split

        texts, facts = {}, {}
        for identity, split in sorted(spec.assignments.items()):
            raw = ledger.get(identity, "raw")
            label = latest(ledger, identity)
            if not ledger.allowed(identity) or raw["origin"] not in ("human", "sensor"):
                raise ValueError(
                    "Excluded or model-derived sources cannot supply independent memory evidence"
                )
            if not label or not label["reviewed"] or label["status"] == "excluded":
                raise ValueError("Every dataset record needs a current human review")
            assign(group_keys(raw, label), split)
            text = normalized_text(label["text"])
            if text in texts:
                warnings.append(dict(kind="duplicate_text", record_ids=[texts[text], identity]))
            texts[text] = identity
            for fact in label["fact_keys"]:
                for previous in facts.get(fact, []):
                    if previous["text"] != text:
                        warnings.append(
                            dict(
                                kind="possible_conflict_or_change",
                                fact_key=fact,
                                record_ids=[previous["id"], identity],
                            )
                        )
                facts.setdefault(fact, []).append(dict(id=identity, text=text))
            records.append(
                dict(
                    id=identity,
                    source_id=raw["source_id"],
                    source_kind=raw["source_kind"],
                    source_timestamp=raw["source_timestamp"],
                    source_digest=raw["content_digest"],
                    annotation_id=label["id"],
                    split=split,
                    **{k: label[k] for k in ("text", "fact_keys", "status", "valid_from", "valid_until")},
                )
            )
        case_ids = set()
        for case in spec.cases:
            if case.id in case_ids:
                raise ValueError("Duplicate evaluation case ID")
            case_ids.add(case.id)
            assign(case.group_ids + ["query:" + digest(normalized_text(case.query))], case.split)
            if len(set(case.expected_ids)) != len(case.expected_ids):
                raise ValueError("Duplicate expected record ID")
            for identity in case.expected_ids:
                if spec.assignments.get(identity) != case.split:
                    raise ValueError("Expected evidence must belong to the query's split")
            if not any(r["split"] == case.split for r in records):
                raise ValueError("Each evaluated split needs a reviewed corpus, including negative cases")
        value = dict(
            schema_version=1,
            transform_version=TRANSFORM_VERSION,
            spec=spec.model_dump(),
            records=records,
            warnings=warnings,
        )
        identity = "dataset:" + digest(value)
        with ledger.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for key, split in sorted(groups.items()):
                old = db.execute("SELECT split FROM split_groups WHERE key=?", (key,)).fetchone()
                if old and old[0] != split:
                    raise ValueError(
                        "Leakage: this group was reserved for another split in an earlier dataset"
                    )
                db.execute("INSERT OR IGNORE INTO split_groups VALUES (?,?)", (key, split))
            return ledger.append("dataset", value, identity, db)
