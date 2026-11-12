"""Deterministic, extractive recall. Never turns a ranking score into confidence."""

import re

from .contracts import Strategy
from .labeling import normalized_text


def tokens(text):
    return set(re.findall(r"\w+", text.casefold()))


def retrieve(records, query, config: Strategy, *, as_of=None, allowed_sources=None, include_history=False):
    words = tokens(query)
    ranked = []
    for record in records:
        if allowed_sources is not None and record["source_id"] not in allowed_sources:
            continue
        if record["status"] == "excluded":
            continue
        if as_of is not None:
            if record["valid_from"] is not None and as_of < record["valid_from"]:
                continue
            if record["valid_until"] is not None and as_of >= record["valid_until"]:
                continue
            if record["status"] == "historical" and record["valid_until"] is None:
                continue
        elif record["status"] == "historical" and not include_history:
            continue
        other = tokens(record["text"])
        common = len(words & other)
        divisor = len(words | other) if config.algorithm == "overlap" else len(words)
        score = common / max(1, divisor)
        if score >= config.min_score:
            ranked.append((score, record))
    ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
    # Consolidate exact redundant statements while retaining every original citation.
    consolidated = {}
    for score, record in ranked:
        key = (normalized_text(record["text"]), record["status"], record["valid_from"], record["valid_until"])
        if key not in consolidated:
            consolidated[key] = dict(
                text=record["text"],
                status=record["status"],
                score=score,
                evidence_ids=[],
                annotation_ids=[],
                source_ids=[],
                fact_keys=[],
                valid_from=record["valid_from"],
                valid_until=record["valid_until"],
            )
        hit = consolidated[key]
        hit["evidence_ids"].append(record["id"])
        hit["annotation_ids"].append(record["annotation_id"])
        hit["source_ids"] = sorted(set(hit["source_ids"] + [record["source_id"]]))
        hit["fact_keys"] = sorted(set(hit["fact_keys"] + record["fact_keys"]))
    hits = list(consolidated.values())[: config.top_k]
    # Consult the full eligible set for conflicts, even when top_k hides an alternative.
    uncertain = any(h["status"] == "contested" for h in hits)
    for hit in hits:
        if any(
            set(hit["fact_keys"]) & set(other["fact_keys"]) and hit["text"] != other["text"]
            for other in consolidated.values()
        ):
            uncertain = True
    return dict(
        hits=hits,
        insufficient_evidence=not hits,
        uncertain=uncertain,
        answer_kind="extractive",
        score_meaning="lexical similarity, not verified correctness",
    )
