"""Persistent question scheduling with a bounded human interruption budget."""

import hashlib

from ..store import uid
from .contracts import Question
from .repository import get, meta, put, rows, set_meta


def offer(
    db,
    episode_id,
    *,
    kind,
    text,
    reason,
    observation,
    proposition=None,
    region=None,
    priority=1,
    discriminator="",
):
    proposition = proposition or {}
    fingerprint = hashlib.sha256(
        repr(
            (
                episode_id,
                kind,
                proposition.get("place_id"),
                proposition.get("event_key"),
                proposition.get("entity_id"),
                proposition.get("belief_id"),
                discriminator,
            )
        ).encode()
    ).hexdigest()
    if rows(db, "questions", "json_extract(payload, '$.fingerprint')=?", (fingerprint,)):
        return
    q = Question(
        id=uid(),
        episode_id=episode_id,
        kind=kind,
        text=text,
        reason=reason,
        evidence_refs=observation.evidence_refs,
        observation_ids=[observation.id],
        proposition=proposition,
        region=region,
        priority=priority,
        fingerprint=fingerprint,
        created_at=meta(db, "now", 0),
    ).model_dump()
    put(db, "questions", q)


def schedule(db, episode_id, now):
    episode = get(db, "episodes", episode_id)
    # Deferred questions accumulate over long sessions; only active and askable ones are decoded.
    questions = rows(db, "questions", "json_extract(payload, '$.status')='active'")
    for q in questions:
        if now - (q.get("touched_at") or q["shown_at"]) >= 20:
            q["status"] = "open"
            q["deferred"] = True
            put(db, "questions", q)
    active = next((q for q in questions if q["status"] == "active"), None)
    if active:
        return active if active["episode_id"] == episode_id else None
    if (
        episode.get("paused")
        or episode.get("status") == "stopped"
        or now - meta(db, "last_asked", -1e12) < 60
    ):
        return None
    pending = rows(
        db,
        "questions",
        "json_extract(payload, '$.episode_id')=? AND json_extract(payload, '$.status')='open'"
        " AND NOT COALESCE(json_extract(payload, '$.deferred'), 0)",
        (episode_id,),
    )
    if not pending:
        return None
    question = max(pending, key=lambda q: (q["priority"], -q["created_at"]))
    question.update(status="active", shown_at=now)
    put(db, "questions", question)
    set_meta(db, "last_asked", now)
    return question
