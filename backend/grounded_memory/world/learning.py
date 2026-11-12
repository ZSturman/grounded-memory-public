"""Prequential event learning: assess each experience before updating its expectations."""

import hashlib

from .attention import offer
from .repository import change_belief, get, new_belief, put, rows


def key(*parts):
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def event_modality(event_key):
    if event_key.startswith("sound:"):
        return "audio"
    if event_key.startswith(("object:", "appeared:", "disappeared:", "change:", "region:")):
        return "vision"
    return "event"


def record_exposure(db, observation, place_id):
    # One evidence unit per original source, modality, place, and ten-second interval.
    # Reopening a recording and receiving many adjacent frames do not add independent units.
    eid = key(observation.source_id, observation.modality, place_id, int(observation.timestamp // 10))
    exposure = get(
        db,
        "exposures",
        eid,
        dict(
            id=eid,
            source_id=observation.source_id,
            place_id=place_id,
            modality=observation.modality,
            events=[],
            duration=0,
            observation_ids=[],
        ),
    )
    exposure["duration"] += observation.duration
    exposure["observation_ids"].append(observation.id)
    put(db, "exposures", exposure)
    return exposure


def record_intervention(db, belief, prediction, outcome):
    """An intervention test is kept apart from passive co-occurrence on the belief it tested."""
    if not prediction.get("intervention"):
        return
    belief.setdefault("interventions", []).append(
        dict(prediction_id=prediction["id"], outcome=outcome, evidence=prediction.get("outcome_evidence", []))
    )
    if outcome == "effect followed" and belief["status"] == "active":
        belief["certainty"] = "tested"
    put(db, "beliefs", belief)


def evaluate_predictions(db, observation, events):
    event_keys = {e["key"] for e in events}
    for prediction in rows(
        db,
        "predictions",
        "json_extract(payload, '$.episode_id')=? AND json_extract(payload, '$.status')='pending'",
        (observation.episode_id,),
    ):
        if prediction["episode_id"] != observation.episode_id or prediction["status"] != "pending":
            continue
        if prediction["clock_id"] != observation.clock_id:
            continue
        belief = get(db, "beliefs", prediction["belief_id"])
        if (
            not belief
            or belief["status"] not in ("active", "contested")
            or belief["version"] != prediction["belief_version"]
        ):
            prediction["status"] = "invalidated"
        elif (
            prediction["start"] < observation.timestamp <= prediction["due"]
            and prediction["event_key"] in event_keys
        ):
            prediction.update(status="observed", outcome_evidence=[observation.id])
            record_intervention(db, belief, prediction, "effect followed")
        elif observation.timestamp >= prediction["due"]:
            intervals = []
            for item in rows(
                db,
                "observations",
                "json_extract(payload, '$.episode_id')=? AND json_extract(payload, '$.timestamp') BETWEEN ? AND ?",
                (observation.episode_id, prediction["start"] - 60, prediction["due"]),
            ):
                if (
                    item["episode_id"] == observation.episode_id
                    and item["clock_id"] == observation.clock_id
                    and item["modality"] == prediction["modality"]
                    and item["origin"] == "sensor"
                    and item["quality"] >= 0.5
                    and not item.get("excluded")
                ):
                    lo = max(prediction["start"], item["timestamp"])
                    hi = min(prediction["due"], item["timestamp"] + item["duration"])
                    if hi > lo:
                        intervals.append((lo, hi))
            covered, end = 0, prediction["start"]
            for lo, hi in sorted(intervals):
                covered += max(0, hi - max(end, lo))
                end = max(end, hi)
            coverage = covered / (prediction["due"] - prediction["start"])
            prediction.update(
                status="not_observed" if coverage >= 0.7 else "unobserved",
                coverage=coverage,
                outcome_evidence=[observation.id],
            )
            if coverage >= 0.7:
                record_intervention(db, belief, prediction, "effect absent")
                # The source's statement remains intact; only its current reliability is revised.
                change_belief(
                    db,
                    belief,
                    operation="contradict",
                    reason="Expected event was not observed with sufficient sensor coverage",
                    evidence=[observation.id],
                )
                offer(
                    db,
                    observation.episode_id,
                    kind="clarify",
                    text=f"I expected {prediction['label']}, but didn't observe it. What might explain that?",
                    reason="Experience conflicts with an earlier expectation",
                    observation=observation,
                    proposition=dict(
                        belief_id=belief["id"],
                        event_key=prediction["event_key"],
                        place_id=belief.get("context"),
                    ),
                    priority=5,
                    discriminator=str(belief["version"]),
                )
        put(db, "predictions", prediction)


def promote_new_region(db, place_id, region):
    """Something first asked about as unfamiliar has persisted where an earlier visit saw nothing."""
    from .localization import similarity

    for q in rows(
        db,
        "questions",
        "json_extract(payload, '$.status')='open' AND json_extract(payload, '$.proposition.place_id') IS ?",
        (place_id,),
    ):
        if (
            q.get("region")
            and q.get("shown_at") is None
            and similarity(q["region"].get("features", []), region.get("features", [])) >= 0.85
        ):
            q.update(
                text="This wasn't here before. What is it?",
                reason="New since an earlier visit to this place",
                priority=max(q["priority"], 2),
            )
            put(db, "questions", q)


def learn_event(db, episode, observation, event, exposure):
    place_id = episode["location"].get("place_id")
    event_key = event["key"]
    if event.get("unnamed") and (event.get("region") or {}).get("familiarity") == "new_here":
        promote_new_region(db, place_id, event["region"])
    base = "FROM world_exposures WHERE json_extract(payload, '$.modality')=? AND id<>?"
    hit = "SUM(EXISTS(SELECT 1 FROM json_each(json_extract(world_exposures.payload, '$.events')) WHERE value=?))"
    global_hits = (
        db.execute(
            "SELECT " + hit + " " + base, (event_key, observation.modality, exposure["id"])
        ).fetchone()[0]
        or 0
    )
    count, local_hits = db.execute(
        "SELECT count(*), " + hit + " " + base + " AND json_extract(payload, '$.place_id') IS ?",
        (event_key, observation.modality, exposure["id"], place_id),
    ).fetchone()
    local_hits = local_hits or 0
    probability = (local_hits + 1) / (count + 2)
    surprising = bool(place_id and count >= 3 and global_hits >= 3 and probability <= 0.25)
    # Repeated frames extend one event episode; they do not generate independent evidence.
    recent = episode.setdefault("ongoing_events", {})
    previous = recent.get(event_key)
    if (
        previous
        and observation.timestamp - previous["last"] <= 5
        and previous["clock"] == observation.clock_id
    ):
        previous["last"] = observation.timestamp
        existing = get(db, "events", previous["id"])
        if existing:
            existing["end"] = observation.timestamp + observation.duration
            put(db, "events", existing)
        return
    event_id = key(exposure["id"], event_key)
    if get(db, "events", event_id):
        return
    record = dict(
        event,
        id=event_id,
        episode_id=observation.episode_id,
        observation_id=observation.id,
        source_id=observation.source_id,
        place_id=place_id,
        start=observation.timestamp,
        end=observation.timestamp + observation.duration,
        heard_at=place_id if observation.modality == "audio" else None,
        source_location=None,
        prior_frequency=probability if place_id and count else None,
        surprising=surprising,
        evidence_refs=observation.evidence_refs,
    )
    put(db, "events", record)
    recent[event_key] = dict(id=event_id, last=observation.timestamp, clock=observation.clock_id)
    exposure["events"] = list(dict.fromkeys(exposure["events"] + [event_key]))
    put(db, "exposures", exposure)
    familiarity = (event.get("region") or {}).get("familiarity")
    if event.get("unnamed") and familiarity != "fixture":  # a fixture is part of the place, not a question
        region = event.get("region")
        # Spatially/visually similar unlabelled regions reuse a question within this place.
        from .localization import similarity

        already = any(
            q.get("region")
            and region
            and similarity(q["region"].get("features", []), region.get("features", [])) >= 0.85
            for q in rows(
                db,
                "questions",
                "json_extract(payload, '$.region') IS NOT NULL"
                " AND json_extract(payload, '$.proposition.place_id') IS ?",
                (place_id,),
            )
        )
        new_here = familiarity == "new_here"
        if not already:
            candidate = next(iter((region or {}).get("matches") or []), None)
            candidate = candidate if candidate and candidate["appearance"] >= 0.7 else None
            offer(
                db,
                observation.episode_id,
                kind="identify",
                text="What is that sound?"
                if event["kind"] == "sound"
                else f"Is this {candidate['label']}?"
                if candidate
                else "This wasn't here before. What is it?"
                if new_here
                else "What is this?",
                reason="New since an earlier visit to this place"
                if new_here
                else "An unfamiliar sensory pattern",
                observation=observation,
                proposition=dict(
                    event_key=event_key,
                    modality=observation.modality,
                    place_id=place_id,
                    entity_id=candidate["id"] if candidate else None,
                    label=candidate["label"] if candidate else None,
                ),
                region=region,
                priority=3 if event["kind"] == "sound" else 2 if new_here else 1,
            )
    elif surprising or event["kind"] == "change":
        offer(
            db,
            observation.episode_id,
            kind="explain",
            text=f"Why did I observe {event['label']} here?",
            reason="Unexpected in this place" if surprising else "A change attracted attention",
            observation=observation,
            proposition=dict(event_key=event_key, label=event["label"], place_id=place_id),
            region=event.get("region"),
            priority=4 if surprising else 2,
        )
    if place_id and not event.get("unnamed"):
        bid = "occurrence:" + key(place_id, event_key)
        belief = get(db, "beliefs", bid)
        if not belief:
            belief = new_belief(
                db,
                subject=place_id,
                predicate="occurrence",
                value=event_key,
                source={"kind": "observation"},
                evidence=[],
                context=place_id,
                belief_id=bid,
                label=event["label"],
            )
        belief["supporting"] = list(dict.fromkeys(belief["supporting"] + [exposure["id"]]))
        belief["certainty"] = "tentative" if len(belief["supporting"]) < 3 else "repeated"
        put(db, "beliefs", belief)
    last = episode.get("last_event")
    if (
        last
        and last["key"] != event_key
        and 0 < observation.timestamp - last["time"] <= 30
        and last["place_id"] == place_id
    ):
        bid = "transition:" + key(place_id, last["key"], event_key)
        belief = get(db, "beliefs", bid) or new_belief(
            db,
            subject=last["key"],
            predicate="transition",
            value=event_key,
            source={"kind": "observation"},
            evidence=[],
            context=place_id,
            belief_id=bid,
            label=event["label"],
            horizon_seconds=30,
            cause_event=last["key"],
            expected_event=event_key,
        )
        belief["supporting"] = list(dict.fromkeys(belief["supporting"] + [exposure["id"]]))
        put(db, "beliefs", belief)
        if len(belief["supporting"]) >= 3 and not belief.get("interventions"):
            # Repetition is correlation. Only doing the cause and watching for the effect tests it.
            cause = last.get("label") or last["key"]
            offer(
                db,
                observation.episode_id,
                kind="intervene",
                text=f"{event['label']} has followed {cause} {len(belief['supporting'])} times. "
                f"Could you make {cause} happen again so I can see whether {event['label']} follows?",
                reason="Repeated order is not yet evidence of cause",
                observation=observation,
                proposition=dict(belief_id=belief["id"], event_key=last["key"], place_id=place_id),
                priority=2,
                discriminator=str(belief["version"]),
            )
    if not event.get("unnamed"):
        episode["last_event"] = dict(
            key=event_key, label=event["label"], time=observation.timestamp, place_id=place_id
        )
    for belief in rows(db, "beliefs"):
        if (
            belief.get("cause_event") != event_key
            or not belief.get("expected_event")
            or belief.get("context") != place_id
            or belief["status"] != "active"
            or (belief["source"]["kind"] == "observation" and len(belief["supporting"]) < 3)
        ):
            continue
        pid = key(belief["id"], belief["version"], event_id)
        request = belief.get("intervention_request")
        intervention = bool(
            request
            and request["episode_id"] == observation.episode_id
            and request["after"] <= observation.timestamp <= request["expires"]
        )
        if intervention:
            belief.pop("intervention_request")
            put(db, "beliefs", belief)
        if not get(db, "predictions", pid):
            put(
                db,
                "predictions",
                dict(
                    id=pid,
                    episode_id=observation.episode_id,
                    clock_id=observation.clock_id,
                    belief_id=belief["id"],
                    belief_version=belief["version"],
                    event_key=belief["expected_event"],
                    label=belief.get("expected_label", belief.get("label", belief["expected_event"])),
                    modality=event_modality(belief["expected_event"]),
                    start=observation.timestamp,
                    due=observation.timestamp + belief["horizon_seconds"],
                    status="pending",
                    evidence=[observation.id],
                    intervention=intervention,
                ),
            )
