"""Conservative, multi-view semantic localization. Geometry never crosses tracking frames."""

import numpy as np

from .contracts import LocationEstimate


def similarity(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not a.size or a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# Consecutive views at least this similar belong to one continuous experience (real 1 fps
# recordings: 5th percentile 0.91). Continuity only extends an anchor; it never creates a place.
CONTINUITY = 0.85
ANCHOR_SECONDS = 30
ANCHOR_METRES = 3.0
RECENT_VIEWS = 3


def position(observation):
    pose = observation.measurements.get("pose")
    return list(pose[12:15]) if pose and len(pose) == 16 else None


def anchor(episode, observation, place_id):
    """A teaching or recognition: the start of a continuity chain for that place."""
    episode["continuity"] = dict(
        place_id=place_id,
        timestamp=observation.timestamp,
        position=position(observation),
        tracking_frame_id=getattr(observation, "tracking_frame_id", None),
        observation_id=observation.id,
    )


def continuous(episode, observation, place_id):
    """True while views stay unbroken since the place's anchor and the body has not gone far."""
    chain = episode.get("continuity")
    if not chain or chain["place_id"] != place_id:
        return False
    here, start = position(observation), chain.get("position")
    if here and start and getattr(observation, "tracking_frame_id", None) == chain.get("tracking_frame_id"):
        # Measured self-motion: turning the phone changes the view, not the place.
        return float(np.linalg.norm(np.subtract(here, start))) <= ANCHOR_METRES
    # Comparing with the last few views lets one blurred or swung frame pass; a doorway does not.
    recent = [v for v in episode.get("recent_views", []) if v.get("encoder") == observation.encoder]
    if not recent or max(similarity(v["features"], observation.features) for v in recent) < CONTINUITY:
        return False
    return observation.timestamp - chain["timestamp"] <= ANCHOR_SECONDS


def add_prototype(place, observation, estimate, cap=24, novelty=0.92):
    """Keep a view of a recognized or continuously held place when no stored example resembles it."""
    held = estimate.get("certainty") == "recognized" or estimate.get("continuous")
    if not held or estimate.get("place_id") != place["id"]:
        return None
    examples = place.setdefault("examples", [])
    same = [e for e in examples if e.get("encoder") == observation.encoder]
    if any(similarity(observation.features, e.get("features", [])) >= novelty for e in same):
        return None
    prototype = dict(
        features=list(observation.features),
        encoder=observation.encoder,
        projection=observation.measurements.get("projection", "unknown"),
        observation_id=observation.id,
        source="experience",
        basis="continuity" if estimate.get("continuous") else "recognition",
        evidence_ids=list(estimate.get("evidence_ids", [])),
    )
    examples.append(prototype)
    experienced = [e for e in examples if e.get("source") == "experience"]
    if len(experienced) > cap:
        examples.remove(experienced[0])
    return prototype


def locate(episode, observation, places, transitions=()):
    current = episode.get("location", LocationEstimate().model_dump())
    if observation.modality != "vision" or not observation.features or observation.quality < 0.5:
        return current
    place_id = current.get("place_id")
    if episode.pop("anchor_pending", False) and place_id:
        # A place named without a frozen view (an explicit context) starts its chain here.
        anchor(episode, observation, place_id)
        held = True
    else:
        held = bool(place_id) and continuous(episode, observation, place_id)
    if not held:
        episode.pop("continuity", None)  # one break ends the chain until the next teaching or recognition
    view = dict(features=list(observation.features), encoder=observation.encoder)
    episode["recent_views"] = (episode.get("recent_views") or [])[-(RECENT_VIEWS - 1) :] + [view]
    candidates = []
    for place in places:
        projection = observation.measurements.get("projection", "unknown")
        scores = [
            (similarity(observation.features, example.get("features", [])), example)
            for example in place.get("examples", [])
            if example.get("encoder") == observation.encoder
            and not example.get("excluded")
            and (projection == "unknown" or example.get("projection", "unknown") in ("unknown", projection))
        ]
        if scores:
            landmarks = observation.measurements.get("landmark_places", [])
            transition_support = sum(
                len(b["supporting"])
                for b in transitions
                if b["subject"] == current.get("place_id")
                and b["value"] == place["id"]
                and b["status"] == "active"
            )
            score, example = max(scores, key=lambda pair: pair[0])
            candidates.append(
                dict(
                    id=place["id"],
                    label=place["label"],
                    similarity=score,
                    example_observation_id=example.get("observation_id"),
                    example_source=example.get("source", "teaching"),
                    landmark_support=landmarks.count(place["id"]),
                    transition_support=transition_support,
                )
            )
    candidates.sort(
        key=lambda c: (c["similarity"], c["landmark_support"], c["transition_support"]), reverse=True
    )
    best = candidates[0] if candidates else None
    gap = (best["similarity"] - candidates[1]["similarity"]) if len(candidates) > 1 else 1
    # Temporal votes must be separate samples, not adjacent callbacks or replay duplicates.
    votes = episode.setdefault("location_votes", [])
    if not votes or observation.timestamp - votes[-1]["timestamp"] >= 1:
        votes.append(
            dict(
                id=best["id"]
                if best
                and (
                    (best["similarity"] >= 0.88 and gap >= 0.08)
                    or (best["similarity"] >= 0.78 and best["landmark_support"] >= 2 and gap >= 0.04)
                )
                and "histogram" not in (observation.encoder or "")
                else None,
                timestamp=observation.timestamp,
                observation_id=observation.id,
            )
        )
    episode["location_votes"] = votes = votes[-5:]
    supported = best and sum(v["id"] == best["id"] for v in votes[-3:]) >= 3
    if supported:
        anchor(episode, observation, best["id"])
        return LocationEstimate(
            place_id=best["id"],
            label=best["label"],
            certainty="recognized",
            candidates=candidates[:3],
            evidence_ids=[v["observation_id"] for v in votes[-3:]],
            previous_place_id=current.get("place_id") or current.get("previous_place_id"),
        ).model_dump()
    # A freshly taught place survives weak visual matching until three independent mismatches.
    if current.get("certainty") == "taught" and len(votes) < 3:
        return dict(current, candidates=candidates[:3], continuous=bool(held))
    if current.get("place_id") and any(v["id"] == current["place_id"] for v in votes[-3:]):
        return dict(current, candidates=candidates[:3], continuous=bool(held))
    # Another place's appearance may interrupt continuity; otherwise an unbroken view holds the place.
    if held and not (best and best["id"] != current["place_id"] and best["similarity"] >= 0.88):
        return dict(current, candidates=candidates[:3], continuous=True)
    episode.pop("continuity", None)
    return LocationEstimate(
        certainty="tentative" if best and best["similarity"] >= 0.65 else "unknown",
        candidates=candidates[:3],
        previous_place_id=current.get("place_id") or current.get("previous_place_id"),
    ).model_dump()
