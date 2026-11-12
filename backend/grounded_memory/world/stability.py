"""Which unfamiliar regions belong to a place, and which are new since an earlier visit.

The first visit treats everything as part of the place. A region seen again on separate visits is
a fixture of that place. A region absent from an earlier visit that saw the same view is new here.
An earlier visit that never looked this way is missing coverage, not evidence of change.
"""

import numpy as np

from ..store import uid
from .localization import similarity
from .repository import get, meta, put, rows, set_meta

SAME_REGION = 0.85  # the existing threshold for treating two unlabelled region crops as one thing
SAME_VIEW = 0.88  # the place-recognition threshold: the earlier visit saw this viewpoint
FIXTURE_VISITS = 2
# Real replay: one-off mask fragments looked "new"; something new persists across separate frames.
NEW_SIGHTINGS = 3
LOOSE_REGION = 0.75


def visits(db, place_id):
    return meta(db, "place_visits:" + place_id, {})


def record_visit(db, place_id, episode_id):
    seen = visits(db, place_id)
    seen[episode_id] = seen.get(episode_id, 0) + 1
    set_meta(db, "place_visits:" + place_id, seen)
    return seen


def covered_by_earlier_visit(db, observation, estimate):
    """The current view matched a stored view of this place taken in a different episode."""
    best = next(iter(estimate.get("candidates") or []), None)
    if not best or best["id"] != estimate.get("place_id") or best["similarity"] < SAME_VIEW:
        return False
    source = best.get("example_observation_id")
    earlier = source and get(db, "observations", source)
    return bool(earlier) and earlier["episode_id"] != observation.episode_id


def annotate(db, episode, observation, estimate):
    """Record unfamiliar regions under the held place and label each one's familiarity there."""
    place_id = estimate.get("place_id")
    regions = observation.measurements.get("regions") or []
    if not place_id or observation.modality != "vision" or observation.quality < 0.5:
        return
    seen = record_visit(db, place_id, observation.episode_id)
    earlier_visits = [e for e in seen if e != observation.episode_id]
    covered = None
    memory = rows(db, "place_regions", "json_extract(payload, '$.place_id')=?", (place_id,))
    for region in regions:
        match = next(iter(region.get("matches") or []), None)
        features = region.get("features") or []
        if (match and match["appearance"] >= 0.8) or not features:
            continue
        scored = [
            (similarity(features, r["features"]), r) for r in memory if r["encoder"] == observation.encoder
        ]
        score, record = max(scored, key=lambda pair: pair[0], default=(0, None))
        if record and score >= SAME_REGION:
            if observation.episode_id not in record["episodes"]:
                record["episodes"].append(observation.episode_id)
            times = record.setdefault("visit_times", [])
            if record["last_seen"]["episode_id"] != observation.episode_id:
                times.clear()
            if not times or observation.timestamp - times[-1] >= 1:
                times[:] = (times + [observation.timestamp])[-NEW_SIGHTINGS:]
            if record["status"] == "new_candidate" and len(times) >= NEW_SIGHTINGS:
                record["status"] = "new_here"
            n = record["sightings"] = record["sightings"] + 1
            centroid = (np.asarray(record["features"]) * (n - 1) + np.asarray(features)) / n
            record["features"] = (centroid / (np.linalg.norm(centroid) + 1e-9)).tolist()
            record["last_seen"] = dict(episode_id=observation.episode_id, observation_id=observation.id)
            others = [e for e in record["episodes"] if e != observation.episode_id]
            region["familiarity"] = (
                "fixture" if len(record["episodes"]) >= FIXTURE_VISITS and others else record["status"]
            )
        else:
            if covered is None:
                covered = bool(earlier_visits) and covered_by_earlier_visit(db, observation, estimate)
            resembles_earlier = any(
                r is not None and score >= LOOSE_REGION and set(r["episodes"]) - {observation.episode_id}
                for score, r in scored
            )
            status = (
                "part_of_place"
                if not earlier_visits
                else "new_candidate"
                if covered and not resembles_earlier
                else "unseen"
            )
            record = dict(
                id=uid(),
                place_id=place_id,
                encoder=observation.encoder,
                features=list(features),
                episodes=[observation.episode_id],
                sightings=1,
                visit_times=[observation.timestamp],
                status=status,
                first_seen=dict(
                    episode_id=observation.episode_id,
                    observation_id=observation.id,
                    asset=region.get("asset"),
                ),
                last_seen=dict(episode_id=observation.episode_id, observation_id=observation.id),
            )
            memory.append(record)
            region["familiarity"] = status
        record["stability"] = round(len(record["episodes"]) / max(len(seen), 1), 3)
        region["place_region_id"] = record["id"]
        put(db, "place_regions", record)
