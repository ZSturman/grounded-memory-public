"""Single ordered owner of world updates. All state transitions commit atomically."""

import hashlib
import json
import re
import time
from types import SimpleNamespace

from ..store import uid
from . import attention, executive, stability
from .contracts import LocationEstimate, ObservationEnvelope, WorldState
from .learning import evaluate_predictions, key, learn_event, record_exposure
from .localization import add_prototype, anchor, locate, similarity
from .repository import WorldRepository, change_belief, get, meta, new_belief, put, rows, set_meta
from .sensing import sensory_events


class Conflict(ValueError):
    pass


def entities(db, canonical=False):
    found = [
        dict(id=r[0], kind=r[1], label=r[2], **json.loads(r[3]))
        for r in db.execute("SELECT id,kind,label,payload FROM entities")
    ]
    if not canonical:
        return found
    redirects = {
        b["subject"]: b["value"]
        for b in rows(db, "beliefs")
        if b["predicate"] == "same_as" and b["status"] == "active"
    }
    by_id = {e["id"]: e for e in found}
    for entity in found:
        destination, seen = entity["id"], set()
        while destination in redirects and destination not in seen:
            seen.add(destination)
            destination = redirects[destination]
        if destination != entity["id"] and destination in by_id:
            target = by_id[destination]
            target["examples"] = target.get("examples", []) + entity.get("examples", [])
            target.setdefault("identity_sources", []).append(entity["id"])
    return [entity for entity in found if entity["id"] not in redirects]


def clean_region(region):
    return {k: v for k, v in region.items() if k != "features"} if region else None


class WorldEngine:
    def __init__(self, memory, clock=time.time):
        self.repository = WorldRepository(memory)
        self.memory = memory
        self.clock = clock
        with self.repository.transaction() as db:
            # Never assume the physical body stayed put while the brain was stopped.
            for episode in rows(db, "episodes"):
                if episode["status"] == "observing":
                    episode["restart_prior"] = episode["location"]
                    episode["location"] = LocationEstimate(
                        previous_place_id=episode["location"].get("place_id")
                    ).model_dump()
                    episode["location_votes"] = []
                    episode["status"] = "interrupted"
                    put(db, "episodes", episode)

    def create_episode(self, episode_id=None, mode="live"):
        episode_id = episode_id or uid()
        with self.repository.transaction() as db:
            existing = get(db, "episodes", episode_id)
            if existing:
                return existing
            episode = dict(
                id=episode_id,
                mode=mode,
                status="observing",
                created_at=self.clock(),
                paused=False,
                location=LocationEstimate().model_dump(),
                sensors={},
                last_times={},
                latest=None,
                restart_prior=meta(db, "last_location"),
            )
            put(db, "episodes", episode)
            return episode

    def recognizable_entities(self, kind):
        with self.repository.transaction() as db:
            return [e for e in entities(db, canonical=True) if e["kind"] == kind]

    def observation(self, observation_id):
        with self.repository.transaction() as db:
            return get(db, "observations", observation_id)

    def explicit_question(self, episode_id, question):
        from .contracts import Question

        with self.repository.transaction() as db:
            episode = get(db, "episodes", episode_id)
            observation = get(db, "observations", episode.get("latest_vision") or episode["latest"])
            for existing in rows(db, "questions"):
                if existing["status"] == "active":
                    existing.update(status="open", deferred=True)
                    put(db, "questions", existing)
            record = Question(
                id=question["id"],
                episode_id=episode_id,
                kind="identify",
                text="What is this?",
                reason="You directed my attention here",
                evidence_refs=[question["frame"], question["region"]["asset"]],
                observation_ids=[observation["id"]],
                region=question["region"],
                proposition=dict(
                    place_id=episode["location"].get("place_id"),
                    event_key="region:" + question["region"]["id"],
                ),
                status="active",
                created_at=self.clock(),
                shown_at=self.clock(),
                priority=10,
                fingerprint="direct:" + question["id"],
            ).model_dump()
            put(db, "questions", record)

    def withdraw_question(self, question_id, reason):
        with self.repository.transaction() as db:
            record = get(db, "questions", question_id)
            if record:
                record.update(status="dismissed", reason=reason)
                put(db, "questions", record)

    def refine_question(self, question):
        with self.repository.transaction() as db:
            record = get(db, "questions", question["id"])
            if record:
                record["region"] = question["region"]
                record["version"] += 1
                record["evidence_refs"] = list(
                    dict.fromkeys(record["evidence_refs"] + [question["region"]["asset"]])
                )
                record["touched_at"] = self.clock()
                put(db, "questions", record)

    def sync_legacy(self):
        with self.repository.transaction() as db:
            self.repository.import_legacy(db)

    def set_place(self, episode_id, place_id, label):
        with self.repository.transaction() as db:
            episode = get(db, "episodes", episode_id)
            if not episode:
                raise KeyError(episode_id)
            episode["location"] = (
                LocationEstimate(place_id=place_id, label=label, certainty="taught").model_dump()
                if place_id
                else LocationEstimate().model_dump()
            )
            episode["location_votes"] = []
            episode["ongoing_events"] = {}
            episode.pop("continuity", None)
            episode["anchor_pending"] = bool(place_id)
            put(db, "episodes", episode)
            self.repository.import_legacy(db)

    def system_output(self, duration_seconds):
        """Mark speaker output across live capture clocks, including another open review client."""
        with self.repository.transaction() as db:
            for episode in rows(db, "episodes"):
                if episode["mode"] != "live" or episode["status"] != "observing":
                    continue
                for sensor in episode["sensors"].values():
                    if sensor["modality"] != "audio":
                        continue
                    elapsed = self.clock() - sensor.get("received_at", self.clock())
                    if elapsed > 10:
                        continue
                    start = max(0, sensor["timestamp"] + elapsed - 2)
                    end = start + duration_seconds + 4
                    put(
                        db,
                        "playback",
                        dict(
                            id=uid(),
                            episode_id=episode["id"],
                            clock_id=sensor["clock_id"],
                            start=start,
                            end=end,
                            source_id=None,
                            kind="system_output",
                        ),
                    )

    def playback(self, episode_id, clock_id, start, end, source_id=None):
        if end < start or end - start > 300:
            raise ValueError("Playback interval must be ordered and at most five minutes")
        with self.repository.transaction() as db:
            if not get(db, "episodes", episode_id):
                raise KeyError(episode_id)
            put(
                db,
                "playback",
                dict(
                    id=key(episode_id, clock_id, start, end),
                    episode_id=episode_id,
                    clock_id=clock_id,
                    start=start,
                    end=end,
                    source_id=source_id,
                ),
            )

    def observe(self, observation: ObservationEnvelope, retain_evidence=None):
        observation = observation.model_copy(deep=True)
        with self.repository.transaction() as db:
            episode = get(db, "episodes", observation.episode_id)
            if not episode:
                raise KeyError(observation.episode_id)
            old = get(db, "observations", observation.id)
            encoded = observation.model_dump()
            input_digest = hashlib.sha256(json.dumps(encoded, sort_keys=True).encode()).hexdigest()
            if old:
                if old.get("input_digest") != input_digest:
                    raise Conflict("An observation ID cannot be reused for different evidence")
                return False
            source_key = key(
                observation.source_id, observation.sensor_id, observation.timestamp, observation.modality
            )
            if db.execute("SELECT 1 FROM world_receipts WHERE source_key=?", (source_key,)).fetchone():
                return False
            db.execute("INSERT INTO world_receipts VALUES (?,?)", (source_key, observation.id))
            excluded = observation.origin != "sensor" or observation.quality < 0.2
            if observation.modality == "vision" and observation.origin == "sensor":
                # Arrival is tracked even for unusable frames: a dark view is not a lost camera.
                episode["vision_arrived_at"] = self.clock()
            if observation.modality == "audio":
                excluded = excluded or any(
                    p["episode_id"] == observation.episode_id
                    and p["clock_id"] == observation.clock_id
                    and (p["source_id"] is None or p["source_id"] == observation.source_id)
                    and observation.timestamp < p["end"]
                    and observation.timestamp + observation.duration > p["start"]
                    for p in rows(db, "playback")
                )
            encoded.update(
                input_digest=input_digest,
                learner_id=meta(db, "learner_id"),
                received_at=self.clock(),
                excluded=excluded,
            )
            put(db, "observations", encoded)
            sensor_key = observation.clock_id + ":" + observation.sensor_id
            if observation.timestamp < episode["last_times"].get(sensor_key, -1):
                # Preserve late evidence; learning requires a deliberate ordered replay, not time travel.
                episode["late_observations"] = episode.get("late_observations", 0) + 1
                put(db, "episodes", episode)
                return True
            episode["last_times"][sensor_key] = observation.timestamp
            if excluded:
                put(db, "episodes", episode)
                return True
            if episode.get("current_clock") != observation.clock_id:
                episode.update(
                    current_clock=observation.clock_id, visual={}, ongoing_events={}, location_votes=[]
                )
                for transient in ("last_event", "continuity", "recent_views"):
                    episode.pop(transient, None)
            if (
                observation.tracking_frame_id
                and episode.get("tracking_frame_id") != observation.tracking_frame_id
            ):
                episode.update(tracking_frame_id=observation.tracking_frame_id, visual={}, missing_objects={})
            set_meta(db, "now", self.clock())
            episode.update(
                status="stopped" if episode["status"] == "stopped" else "observing", latest=observation.id
            )
            if observation.modality == "vision":
                episode["latest_vision"] = observation.id
            episode["sensors"][observation.sensor_id] = dict(
                modality=observation.modality,
                received_at=self.clock(),
                timestamp=observation.timestamp,
                clock_id=observation.clock_id,
                quality=observation.quality,
            )
            known = entities(db, canonical=True)
            previous_place = episode["location"].get("place_id")
            landmarks = []
            for region in observation.measurements.get("regions", []):
                best = next(iter(region.get("matches") or []), None)
                if best and best["appearance"] >= 0.9:
                    entity = next((e for e in known if e["id"] == best["id"]), None)
                    if entity:
                        landmarks.extend(
                            set(e["place_id"] for e in entity.get("examples", []) if e.get("place_id"))
                        )
            observation.measurements["landmark_places"] = landmarks
            episode["location"] = locate(
                episode,
                observation,
                [e for e in known if e["kind"] == "place"],
                [b for b in rows(db, "beliefs") if b["predicate"] == "place_transition"],
            )
            place = episode["location"].get("place_id")
            if place != previous_place:
                episode["location_epoch"] = episode.get("location_epoch", 0) + 1
                episode["ongoing_events"] = {}
                episode.pop("last_event", None)
                if previous_place and place:
                    bid = "place-transition:" + key(previous_place, place)
                    belief = get(db, "beliefs", bid) or new_belief(
                        db,
                        subject=previous_place,
                        predicate="place_transition",
                        value=place,
                        source={"kind": "observation"},
                        evidence=[],
                        belief_id=bid,
                    )
                    belief["supporting"] = list(dict.fromkeys(belief["supporting"] + [observation.source_id]))
                    put(db, "beliefs", belief)
            learned_view = place and self._remember_view(db, place, observation, episode["location"])
            if place:
                set_meta(db, "last_location", episode["location"])
            elif observation.modality == "vision":
                candidates = episode["location"]["candidates"]
                proposition = dict(place_id=None, candidates=candidates)
                if candidates and candidates[0]["similarity"] >= 0.8:
                    proposition.update(entity_id=candidates[0]["id"], label=candidates[0]["label"])
                attention.offer(
                    db,
                    observation.episode_id,
                    kind="locate",
                    text=f"Is this {proposition['label']}?"
                    if proposition.get("label")
                    else "Where am I? You can show me while I keep observing.",
                    reason="Location changes what I expect",
                    observation=observation,
                    proposition=proposition,
                    priority=3,
                    discriminator=str(episode.get("location_epoch", 0)),
                )
            stability.annotate(db, episode, observation, episode["location"])
            detected = sensory_events(observation, episode, known)
            evaluate_predictions(db, observation, detected)
            exposure = record_exposure(db, observation, place)
            for event in detected:
                learn_event(db, episode, observation, event, exposure)
            if retain_evidence:
                questions = rows(
                    db, "questions", "json_extract(payload, '$.episode_id')=?", (observation.episode_id,)
                )
                owned = [q for q in questions if observation.id in q["observation_ids"]]
                event_evidence = rows(
                    db, "events", "json_extract(payload, '$.observation_id')=?", (observation.id,)
                )
                periodic = observation.timestamp - episode.get("retained_at", -1e12) >= 5
                keep = bool(owned or event_evidence or periodic or learned_view)
                refs = list(observation.evidence_refs) if keep else []
                for q in owned:
                    if q.get("region"):
                        refs.append(q["region"]["asset"])
                for event in event_evidence:
                    if event.get("region"):
                        refs.append(event["region"]["asset"])
                if keep:
                    episode["retained_at"] = observation.timestamp
                    geometry = observation.measurements.get("geometry")
                    if geometry and geometry.get("depth"):
                        refs.append(geometry["depth"])
                retain_evidence(list(dict.fromkeys(refs)))
                encoded["evidence_refs"] = observation.evidence_refs if keep else []
                # The recording and source timestamp remain the authoritative reference for other samples.
                encoded["recording_reference"] = dict(
                    source_id=observation.source_id, timestamp=observation.timestamp
                )
                put(db, "observations", encoded)
            put(db, "episodes", episode)
            attention.schedule(db, observation.episode_id, self.clock())
        return True

    def _remember_view(self, db, place_id, observation, estimate):
        """Store a new appearance of a held place on its own record, with the observation as provenance."""
        row = db.execute(
            "SELECT label, payload FROM entities WHERE id=? AND kind='place'", (place_id,)
        ).fetchone()
        if not row:
            return None
        place = dict(id=place_id, label=row[0], **json.loads(row[1]))
        prototype = add_prototype(place, observation, estimate)
        if prototype:
            prototype.update(
                asset=next(iter(observation.evidence_refs), None),
                sensor_id=observation.sensor_id,
                tracking_frame_id=observation.tracking_frame_id,
                timestamp=observation.timestamp,
            )
            payload = {k: v for k, v in place.items() if k not in ("id", "label")}
            db.execute("UPDATE entities SET payload=? WHERE id=?", (json.dumps(payload), place_id))
        return prototype

    def _exclude_views(self, db, question, reply):
        """A rejected place guess stops trusting the learned view that suggested it; the view stays stored."""
        for candidate in question["proposition"].get("candidates", [])[:1]:
            if candidate.get("example_source") != "experience" or not candidate.get("example_observation_id"):
                continue
            row = db.execute("SELECT payload FROM entities WHERE id=?", (candidate["id"],)).fetchone()
            if not row:
                continue
            payload = json.loads(row[0])
            for example in payload.get("examples", []):
                if example.get("observation_id") == candidate["example_observation_id"]:
                    example.update(excluded=True, excluded_by=reply.id)
            db.execute("UPDATE entities SET payload=? WHERE id=?", (json.dumps(payload), candidate["id"]))

    def _name(self, db, question, reply, observation):
        kind = (
            "place"
            if question["kind"] == "locate"
            else "sound"
            if observation["modality"] == "audio"
            else "object"
        )
        text = reply.text.strip()
        label = reply.label or (question["proposition"].get("label") if reply.response == "yes" else None)
        label = label or re.sub(
            r"^(?:that is|this is|it is|that's|it's|we are in|we're in)\s+", "", text, flags=re.I
        ).rstrip(".")
        if not label or len(label) > 200:
            raise ValueError("Give a short name, or select a name from your answer")
        existing = entities(db)
        entity_id = reply.entity_id or (
            question["proposition"].get("entity_id") if reply.response == "yes" else None
        )
        named = [e for e in existing if e["kind"] == kind and e["label"].casefold() == label.casefold()]
        if named and not entity_id:
            # A duplicate name is not identity evidence. Return choices without committing the reply.
            return dict(
                needs_clarification=True, matches=[{k: e[k] for k in ("id", "label", "kind")} for e in named]
            )
        entity = next((e for e in existing if e["id"] == entity_id and e["kind"] == kind), None)
        if entity_id and not entity:
            if entity_id != "new":
                raise ValueError("Choose an existing entity of the same kind")
            entity_id = None
        entity_id = entity_id or uid()
        evidence = dict(
            question.get("region") or {},
            features=(question.get("region") or {}).get("features", observation["features"]),
            encoder=(question.get("region") or {}).get("encoder", observation["encoder"]),
            asset=(question.get("region") or {}).get("asset") or next(iter(question["evidence_refs"]), None),
            observation_id=observation["id"],
            sensor_id=observation["sensor_id"],
            projection=observation["measurements"].get("projection", "unknown"),
            place_id=question["proposition"].get("place_id"),
        )
        examples = entity.get("examples", []) if entity else []
        if not any(e.get("observation_id") == observation["id"] for e in examples):
            examples.append(evidence)
        db.execute(
            "INSERT INTO entities VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET label=excluded.label,payload=excluded.payload",
            (entity_id, kind, label, json.dumps(dict(examples=examples))),
        )
        belief = new_belief(
            db,
            subject=entity_id,
            predicate="named",
            value=label,
            source={"kind": "human", "id": reply.teacher_id},
            evidence=[reply.id, observation["id"]],
        )
        belief["certainty"] = "taught"
        put(db, "beliefs", belief)
        if kind == "place":
            stability.record_visit(db, entity_id, question["episode_id"])
            episode = get(db, "episodes", question["episode_id"])
            latest = get(db, "observations", episode.get("latest_vision") or episode.get("latest"))
            if latest and (
                latest["id"] == observation["id"]
                or (
                    latest["clock_id"] == observation["clock_id"]
                    and 0 <= latest["timestamp"] - observation["timestamp"] <= 5
                    and latest["tracking_frame_id"] == observation["tracking_frame_id"]
                    and latest["encoder"] == observation["encoder"]
                    and similarity(latest["features"], observation["features"]) >= 0.88
                )
            ):
                episode["location"] = LocationEstimate(
                    place_id=entity_id, label=label, certainty="taught", evidence_ids=[reply.id]
                ).model_dump()
                episode["location_votes"] = []
                episode["ongoing_events"] = {}
                # Continuity starts from the view the teacher named, not whatever arrives next.
                view = SimpleNamespace(**latest)
                anchor(episode, view, entity_id)
                episode["recent_views"] = [dict(features=view.features, encoder=view.encoder)]
                put(db, "episodes", episode)
        return dict(entity_id=entity_id, belief_id=belief["id"], needs_clarification=False)

    def answer(self, question_id, reply):
        original_reply = reply.model_dump()
        with self.repository.transaction() as db:
            old = get(db, "teachings", reply.id)
            if old:
                if old.get("reply") != original_reply or old.get("question_id") != question_id:
                    raise Conflict("Reply ID already belongs to a different answer")
                return old["result"]
            q = get(db, "questions", question_id)
            if not q:
                raise KeyError(question_id)
            if q["version"] != reply.question_version or q["status"] == "answered":
                raise Conflict(
                    "Review the current question; this proposition has already changed or been answered"
                )
            if reply.response == "answer":
                short = reply.text.strip().lower().rstrip(".!?")
                meanings = {
                    "yes": "yes",
                    "no": "no",
                    "i don't know": "unknown",
                    "i don’t know": "unknown",
                    "not sure": "unknown",
                    "later": "later",
                    "not important": "dismiss",
                }
                if short in meanings:
                    reply = reply.model_copy(update={"response": meanings[short]})
            observation = get(db, "observations", q["observation_ids"][0])
            result = dict(saved=True)
            if q["kind"] == "intervene" and reply.response in ("no", "yes", "answer"):
                # Agreeing schedules a test; declining is not evidence against the expectation.
                if reply.response != "no":
                    belief = get(db, "beliefs", q["proposition"]["belief_id"])
                    episode = get(db, "episodes", q["episode_id"])
                    after = max(episode["last_times"].values(), default=0)
                    belief["intervention_request"] = dict(
                        reply_id=reply.id, episode_id=q["episode_id"], after=after, expires=after + 120
                    )
                    put(db, "beliefs", belief)
                q["status"] = "answered" if reply.response != "no" else "dismissed"
            elif reply.response in ("unknown", "later", "dismiss"):
                q.update(status="dismissed" if reply.response == "dismiss" else "open", deferred=True)
            elif reply.response == "no":
                if q["kind"] == "locate":
                    self._exclude_views(db, q, reply)
                if q["proposition"].get("belief_id"):
                    belief = get(db, "beliefs", q["proposition"]["belief_id"])
                    change_belief(
                        db,
                        belief,
                        operation="contradict",
                        reason=reply.text or "Teacher disagreed with the proposition",
                        evidence=[reply.id],
                        source={"kind": "human", "id": reply.teacher_id},
                    )
                q["status"] = "answered"
                follow = dict(
                    q,
                    id=uid(),
                    version=1,
                    status="open",
                    shown_at=None,
                    touched_at=None,
                    text="What should I understand instead?",
                    fingerprint=q["fingerprint"] + ":correction:" + reply.id,
                )
                follow["proposition"] = {
                    k: v for k, v in q["proposition"].items() if k not in ("entity_id", "label")
                }
                put(db, "questions", follow)
            elif q["kind"] in ("locate", "identify"):
                result = self._name(db, q, reply, observation)
                if result.get("needs_clarification"):
                    return result
                q["status"] = "answered"
            else:
                if reply.response == "yes" and not q["proposition"].get("belief_id"):
                    raise ValueError("This question needs an explanation, not a yes/no answer")
                statement = reply.text or "Teacher confirmed the displayed proposition"
                belief = new_belief(
                    db,
                    subject=q["proposition"].get("event_key", "world"),
                    predicate="explanation",
                    value=statement,
                    source={"kind": "human", "id": reply.teacher_id},
                    evidence=[reply.id, observation["id"]],
                    context=q["proposition"].get("place_id"),
                )
                if reply.expected_event:
                    event_records = rows(db, "events")

                    def resolve(label):
                        return next(
                            (
                                e["key"]
                                for e in reversed(event_records)
                                if e["label"].casefold() == label.casefold()
                            ),
                            "event:" + label,
                        )

                    belief.update(
                        cause_event=resolve(reply.cause_event)
                        if reply.cause_event
                        else q["proposition"].get("event_key"),
                        expected_event=resolve(reply.expected_event),
                        expected_label=reply.expected_event,
                        horizon_seconds=reply.horizon_seconds,
                    )
                belief["certainty"] = "taught"
                put(db, "beliefs", belief)
                q["status"] = "answered"
                result["belief_id"] = belief["id"]
            put(db, "questions", q)
            put(
                db,
                "teachings",
                dict(
                    id=reply.id,
                    question_id=question_id,
                    question_version=q["version"],
                    proposition=q["proposition"],
                    text=reply.text,
                    reply=original_reply,
                    evidence=q["observation_ids"],
                    source={"kind": "human", "id": reply.teacher_id},
                    result=result,
                    created_at=self.clock(),
                ),
            )
            return result

    def revise(self, belief_id, revision):
        with self.repository.transaction() as db:
            existing = get(db, "revisions", revision.id)
            if existing:
                if existing.get("request") != revision.model_dump() or existing["belief_id"] != belief_id:
                    raise Conflict("Revision ID already used")
                return get(db, "beliefs", belief_id)
            belief = get(db, "beliefs", belief_id)
            if not belief:
                raise KeyError(belief_id)
            if belief["version"] != revision.expected_version:
                raise Conflict("This belief changed; review its current version")
            if revision.operation == "correct" and not revision.value:
                raise ValueError("A correction requires a replacement statement")
            if revision.operation == "identity":
                known = entities(db)
                subject = next((e for e in known if e["id"] == belief["subject"]), None)
                target = next((e for e in known if e["id"] == revision.target_entity_id), None)
                if (
                    not subject
                    or not target
                    or subject["kind"] != target["kind"]
                    or subject["id"] == target["id"]
                ):
                    raise ValueError("Identity repair requires two distinct entities of the same kind")
                redirects = {
                    b["subject"]: b["value"]
                    for b in rows(db, "beliefs")
                    if b["predicate"] == "same_as" and b["status"] == "active"
                }
                cursor, seen = target["id"], {subject["id"]}
                while cursor in redirects:
                    if cursor in seen:
                        raise ValueError("Identity corrections cannot form a cycle")
                    seen.add(cursor)
                    cursor = redirects[cursor]
                if cursor == subject["id"]:
                    raise ValueError("Identity corrections cannot form a cycle")
                new_belief(
                    db,
                    subject=subject["id"],
                    predicate="same_as",
                    value=target["id"],
                    source={"kind": "human", "id": revision.teacher_id},
                    evidence=[revision.id],
                )
                operation = "contradict"
            else:
                operation = revision.operation
            change_belief(
                db,
                belief,
                operation=operation,
                reason=revision.reason,
                value=revision.value,
                evidence=[revision.id],
                revision_id=revision.id,
                source={"kind": "human", "id": revision.teacher_id},
            )
            if belief["predicate"] == "named":
                if revision.operation == "correct":
                    db.execute("UPDATE entities SET label=? WHERE id=?", (revision.value, belief["subject"]))
                elif revision.operation == "retract":
                    others = [
                        b
                        for b in rows(db, "beliefs")
                        if b["subject"] == belief["subject"]
                        and b["predicate"] == "named"
                        and b["status"] == "active"
                    ]
                    db.execute(
                        "UPDATE entities SET label=? WHERE id=?",
                        (others[-1]["value"] if others else "Unlabeled entity", belief["subject"]),
                    )
            record = get(db, "revisions", revision.id)
            record["request"] = revision.model_dump()
            put(db, "revisions", record)
            return belief

    def control(self, episode_id, action):
        with self.repository.transaction() as db:
            episode = get(db, "episodes", episode_id)
            if not episode:
                raise KeyError(episode_id)
            if action not in ("pause", "resume", "stop"):
                raise ValueError("Use pause, resume, or stop")
            episode["paused"] = action == "pause"
            if action == "stop":
                episode["status"] = "stopped"
            put(db, "episodes", episode)
            for q in rows(db, "questions"):
                if q["episode_id"] == episode_id and q["status"] == "active" and action != "resume":
                    q.update(status="open", deferred=True)
                    put(db, "questions", q)

    def touch(self, question_id):
        with self.repository.transaction() as db:
            question = get(db, "questions", question_id)
            if not question:
                raise KeyError(question_id)
            question["touched_at"] = self.clock()
            put(db, "questions", question)

    def state(self, episode_id=None):
        with self.repository.transaction() as db:
            self.repository.import_legacy(db)
            episode = get(db, "episodes", episode_id) if episode_id else None
            if episode_id and not episode:
                raise KeyError(episode_id)
            active = attention.schedule(db, episode_id, self.clock()) if episode else None

            def public_question(q):
                from .contracts import Question

                return Question.model_validate(
                    {
                        k: v
                        for k, v in dict(q, region=clean_region(q.get("region"))).items()
                        if k in Question.model_fields
                    }
                )

            where = "json_extract(payload, '$.episode_id')=?" if episode_id else None
            params = (episode_id,) if episode_id else ()
            questions = rows(db, "questions", where, params, limit=100)
            events = rows(db, "events", where, params, limit=30)
            predictions = rows(db, "predictions", where, params, limit=30)
            beliefs = rows(db, "beliefs")
            names = {e["id"]: e["label"] for e in entities(db)}
            if episode and episode["location"].get("place_id") in names:
                episode["location"]["label"] = names[episode["location"]["place_id"]]
            for belief in beliefs:
                belief["subject_label"] = names.get(belief["subject"], belief["subject"])
                belief["context_label"] = names.get(belief.get("context"))
            control = (
                executive.assess(
                    episode,
                    self.clock(),
                    [q for q in questions if q["episode_id"] == episode_id],
                    events,
                    predictions,
                )
                if episode
                else {}
            )
            return WorldState(
                learner_id=meta(db, "learner_id"),
                episode_id=episode_id,
                executive=control,
                location=LocationEstimate.model_validate(episode["location"])
                if episode
                else LocationEstimate(),
                attention=public_question(active) if active else None,
                questions=[public_question(q) for q in questions[-100:]],
                beliefs=beliefs[-200:],
                predictions=predictions[-30:],
                recent_events=[dict(e, region=clean_region(e.get("region"))) for e in events[-30:]],
                revisions=rows(db, "revisions", limit=30),
                sensors=episode["sensors"] if episode else {},
                status=episode["status"] if episode else "observing",
                limitations=[
                    "Similarity is not calibrated certainty",
                    "A microphone does not establish a sound's source location",
                ],
            )
