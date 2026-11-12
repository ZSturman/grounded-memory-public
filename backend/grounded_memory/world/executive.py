"""Fixed control modes and a bounded working set, derived from the learner's current state.

Modes are the learner's own policy, not learned meaning, and they are recomputed from the evidence
each time rather than stored as a separate state machine. A body can act on `request` (today a
person moving the phone, later a robot's gaze or wheels); nothing here decides what is true.
"""

WORKING_SET = 12
EVENT_TTL = 30
STALE_VISION = 10


def assess(episode, now, questions, events, predictions):
    location = episode["location"]
    candidates = location.get("candidates") or []
    best = candidates[0] if candidates else None
    vision = [s for s in episode["sensors"].values() if s["modality"] == "vision"]
    asking = next(
        (
            q
            for q in questions
            if q["kind"] == "locate" and q["status"] in ("active", "open") and not q.get("deferred")
        ),
        None,
    )
    if episode["status"] == "stopped":
        mode, reason, request = "STOPPED", "Observation has ended", None
    elif vision and now - max(s["received_at"] for s in vision) > STALE_VISION:
        if now - episode.get("vision_arrived_at", 0) <= STALE_VISION:
            mode, reason = "RECOVER", "Camera frames are arriving but are too dark or blurred to use"
            request = "Uncover the camera, add light, or hold it steadier"
        else:
            mode, reason = "RECOVER", "Camera evidence stopped arriving"
            request = "Point the camera at the scene again; I keep what I knew only as history"
    elif location.get("place_id"):
        basis = "an unbroken view since" if location.get("continuous") else location["certainty"]
        mode, reason, request = "CONFIRMED", f"{location['label']}: {basis}", None
    elif best and best["similarity"] >= 0.65:
        mode, reason = "VERIFY", f"This view partly resembles {best['label']}"
        request = f"Turn slowly so I can compare more of this place with {best['label']}"
    elif location.get("previous_place_id") or episode.get("restart_prior"):
        mode, reason = "RELOCALIZE", "My earlier location is only history until fresh views match"
        request = "Look around slowly so I can find where I am"
    elif asking:
        mode, reason, request = "NOVEL", "Nothing I remember matches this place", "Tell me where we are"
    else:
        mode, reason, request = "EXPLORE", "I have no place to compare yet", "Look around slowly"

    latest = max(episode.get("last_times", {}).values(), default=0)
    by_id = {e["id"]: e for e in events}
    items = []
    if location.get("place_id"):
        items.append(dict(kind="place", id=location["place_id"], label=location["label"], basis=reason))
    elif best:
        items.append(dict(kind="place_candidate", id=best["id"], label=best["label"], basis=reason))
    if asking:
        items.append(dict(kind="question", id=asking["id"], label=asking["text"], basis=asking["reason"]))
    for p in predictions:
        if p["status"] == "pending":
            items.append(
                dict(kind="expectation", id=p["id"], label=p["label"], basis="due", expires=p["due"])
            )
    ongoing = sorted(episode.get("ongoing_events", {}).values(), key=lambda e: -e["last"])
    for event in ongoing:
        record = by_id.get(event["id"])
        if (
            record
            and event.get("clock") == episode.get("current_clock")
            and latest - event["last"] <= EVENT_TTL
        ):
            items.append(
                dict(
                    kind="event",
                    id=record["id"],
                    label=record["label"],
                    basis="surprising" if record.get("surprising") else "observed",
                    expires=event["last"] + EVENT_TTL,
                )
            )
    for key, seen in (episode.get("visual") or {}).get("objects", {}).items():
        items.append(dict(kind="object", id=key.split(":", 1)[-1], label=seen["label"], basis="in view"))
    return dict(mode=mode, reason=reason, request=request, working_set=items[:WORKING_SET])
