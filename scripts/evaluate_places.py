"""Replay real labeled capture sessions through the place localizer and report what it recognizes.

Reads the data directory read-only. Each legacy capture session names its place, and its stored
observations are exactly what the live pipeline saw: DINOv2 scene features, sharpness, timing and,
for native captures, the ARKit camera pose. Nothing is written to the data directory.

    uv run --no-sync python scripts/evaluate_places.py [--json build/place-eval/summary.json]

Each place is named once, `--teach-at` seconds into its first session; that session then continues
untaught. Later sessions of the same place are replayed with no teaching at all, and every place is
also replayed while held out entirely to count false recognitions. Results are frame counts from a
few sessions recorded on one afternoon; adjacent frames are not independent, and nothing here is a
calibrated accuracy.
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from grounded_memory.world import localization  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    default=os.getenv("GM_DATA_DIR", str(Path.home() / "Library/Application Support/Grounded Memory")),
)
parser.add_argument("--json", help="also write the summary here")
parser.add_argument(
    "--teach-at",
    default="5,10,20,40",
    help="seconds into a place's first session; each listed moment is a separate trial",
)
parser.add_argument(
    "--walk",
    action="append",
    default=[],
    help="session ID prefix whose label is not per-frame truth (it moved between places); reported separately",
)
args = parser.parse_args()


def sessions():
    """Ordered observations per labeled session, from the read-only legacy store."""
    db = sqlite3.connect((Path(args.data) / "memory.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    found = []
    for (payload,) in db.execute("SELECT payload FROM sessions"):
        s = json.loads(payload)
        label = ((s.get("context") or {}).get("label") or "").strip().casefold()
        rows = [
            json.loads(p)
            for (p,) in db.execute(
                "SELECT payload FROM observations WHERE session_id=? ORDER BY json_extract(payload, '$.received_at')",
                (s["id"],),
            )
        ]
        rows = [r for r in rows if r.get("modality") == "vision" and r["perception"].get("scene_features")]
        if not label or not rows:
            continue
        sources = list(dict.fromkeys(r["source_id"] for r in rows))
        start = rows[0]["received_at"]
        observations = []
        for i, r in enumerate(rows):
            perception = r["perception"]
            camera = (perception.get("geometry") or {}).get("camera") or {}
            observations.append(
                SimpleNamespace(
                    id=r["id"],
                    modality="vision",
                    features=perception["scene_features"],
                    encoder=perception["encoder"],
                    quality=min(1, perception.get("sharpness", 100) / 100),
                    tracking_frame_id=r["source_id"],
                    measurements=dict(projection="unknown", pose=camera.get("transform")),
                    # One source keeps its media clock; segmented live captures use arrival time.
                    timestamp=float(r["media_time"]) if len(sources) == 1 else r["received_at"] - start,
                )
            )
        walk = any(s["id"].startswith(prefix) for prefix in args.walk)
        found.append(
            dict(id=s["id"], start=s["created_at"], label=label, observations=observations, walk=walk)
        )
    found.sort(key=lambda s: s["start"])
    unique = []
    for s in found:
        # Importing one file twice yields the same features; it is one experience, not two.
        if any(
            u["label"] == s["label"] and u["observations"][0].features == s["observations"][0].features
            for u in unique
        ):
            print(f"skipped {s['id'][:8]}: re-import of an earlier session")
            continue
        unique.append(s)
    return unique


def replay(session, places, learn, teach_at=None, pose=True, trace=None):
    """Run locate() over one session in order. Teaching names the place once at `teach_at`."""
    episode = {}
    outcome = dict(frames=0, correct=0, wrong=0, unknown=0, added=0)
    for observation in session["observations"]:
        if not pose:
            observation = SimpleNamespace(**vars(observation))
            observation.measurements = dict(observation.measurements, pose=None)
        if teach_at is not None and observation.timestamp >= teach_at and observation.quality >= 0.5:
            place = next((p for p in places if p["id"] == session["label"]), None)
            if place is None:
                place = dict(id=session["label"], label=session["label"], examples=[])
                places.append(place)
            place["examples"].append(dict(features=observation.features, encoder=observation.encoder))
            episode["location"] = dict(
                place_id=place["id"], label=place["label"], certainty="taught", candidates=[], evidence_ids=[]
            )
            episode["location_votes"] = []
            episode["anchor_pending"] = True  # the named view itself starts the continuity chain
            teach_at = None
        estimate = localization.locate(episode, observation, places)
        episode["location"] = estimate
        place = estimate.get("place_id")
        if trace is not None:
            trace.append((observation.id, place))
        outcome["frames"] += 1
        outcome["correct" if place == session["label"] else "wrong" if place else "unknown"] += 1
        if learn and place:
            target = next(p for p in places if p["id"] == place)
            outcome["added"] += bool(localization.add_prototype(target, observation, estimate))
    return outcome


def add(total, result):
    for k, v in result.items():
        total[k] = total.get(k, 0) + v


def rate(total, k):
    return total.get(k, 0) / max(total.get("frames", 0), 1)


def main():
    everything = sessions()
    walks = [s for s in everything if s["walk"]]
    data = [s for s in everything if not s["walk"]]
    labels = list(dict.fromkeys(s["label"] for s in data))
    print("sessions:", [(s["label"], s["id"][:8], len(s["observations"])) for s in data])
    summary = {}
    for mode, learn, pose in (
        ("taught views only", False, True),
        ("continuity, no pose", True, False),
        ("continuity with pose", True, True),
    ):
        teaching, later, untaught, walked = {}, {}, {}, {}
        for teach_at in [float(t) for t in args.teach_at.split(",")]:
            places = []
            for label in labels:
                first = next(s for s in data if s["label"] == label)
                add(teaching, replay(first, places, learn, teach_at=teach_at, pose=pose))
            for s in data:
                if s is not next(x for x in data if x["label"] == s["label"]):
                    add(later, replay(s, places, learn=False, pose=pose))
            for s in walks:
                trace = []
                replay(s, places, learn=False, pose=pose, trace=trace)
                for _, place in trace:
                    walked[place or "unknown"] = walked.get(place or "unknown", 0) + 1
            for held_out in labels:
                places = []
                for label in labels:
                    if label != held_out:
                        first = next(s for s in data if s["label"] == label)
                        replay(first, places, learn, teach_at=teach_at, pose=pose)
                for s in data:
                    if s["label"] == held_out:
                        add(untaught, replay(s, places, learn=False, pose=pose))
        summary[mode] = dict(
            teaching_session=teaching, later_sessions=later, untaught_place=untaught, walks=walked
        )
        print(
            f"{mode:21s} | teaching session: place held {rate(teaching, 'correct'):5.1%}, "
            f"wrong {rate(teaching, 'wrong'):4.1%}, +{teaching['added']} views "
            f"| later sessions (n={later['frames']}): correct {rate(later, 'correct'):5.1%} "
            f"wrong {rate(later, 'wrong'):4.1%} | untaught place called known {rate(untaught, 'wrong'):4.1%} "
            f"(n={untaught['frames']})" + (f" | unlabeled walk frames: {walked}" if walks else "")
        )
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
