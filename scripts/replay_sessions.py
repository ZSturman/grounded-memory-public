"""Replay real capture sessions through the whole world engine in a disposable data directory.

The observations are real (what the live pipeline stored: DINOv2 scene features, unfamiliar
regions, sharpness, ARKit pose and arrival times); the teacher is scripted. It answers place
questions from the session's label, says "no" to a wrong place guess, and defers everything else.
A session passed with --walk moved between places, so its teacher defers every question.
The user's data directory is opened read-only and never modified.

    uv run --no-sync python scripts/replay_sessions.py --walk 145cb99a
"""

import argparse
import collections
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from grounded_memory.store import MemoryRepository  # noqa: E402
from grounded_memory.world.contracts import ObservationEnvelope, TeachingReply  # noqa: E402
from grounded_memory.world.engine import WorldEngine  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data",
    default=os.getenv("GM_DATA_DIR", str(Path.home() / "Library/Application Support/Grounded Memory")),
)
parser.add_argument("--walk", action="append", default=[], help="session ID prefix that moved between places")
parser.add_argument("--json", help="also write the per-session summary here")
args = parser.parse_args()


def sessions():
    db = sqlite3.connect((Path(args.data) / "memory.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    found = []
    for (payload,) in db.execute("SELECT payload FROM sessions"):
        s = json.loads(payload)
        label = ((s.get("context") or {}).get("label") or "").strip()
        rows = [
            json.loads(p)
            for (p,) in db.execute(
                "SELECT payload FROM observations WHERE session_id=? ORDER BY json_extract(payload, '$.received_at')",
                (s["id"],),
            )
        ]
        rows = [r for r in rows if r.get("modality") == "vision" and r["perception"].get("scene_features")]
        if label and rows:
            # One source keeps its capture clock; processing backlogs make arrival times misleading.
            single = len({r["source_id"] for r in rows}) == 1
            found.append(
                dict(id=s["id"], label=label, rows=rows, created=s["created_at"], single_source=single)
            )
    return sorted(found, key=lambda s: s["created"])


def envelope(session, r, start):
    perception = r["perception"]
    camera = (perception.get("geometry") or {}).get("camera") or {}
    regions = [
        {k: v for k, v in region.items() if k != "matches"} for region in perception.get("regions", [])
    ]
    return ObservationEnvelope(
        id=r["id"],
        episode_id=session["id"],
        source_id=r["source_id"],
        sensor_id=r.get("sensor_id") or "unknown-camera",
        clock_id=session["id"],
        tracking_frame_id=r["source_id"],
        timestamp=float(r["media_time"]) if session["single_source"] else r["received_at"] - start,
        duration=0.5,
        modality="vision",
        evidence_refs=[perception["frame"]] if perception.get("frame") else [],
        features=perception["scene_features"],
        encoder=perception["encoder"],
        quality=min(1, perception.get("sharpness", 100) / 100),
        measurements=dict(regions=regions, projection="unknown", pose=camera.get("transform")),
    )


def teach(engine, session, question, walk):
    """The scripted teacher: knows only the session's place label."""
    ident = f"{session['id'][:8]}:{question.id}"
    if walk or question.kind != "locate":
        return engine.answer(
            question.id, TeachingReply(id=ident, response="later", question_version=question.version)
        )
    proposed = question.proposition.label
    if proposed and proposed.casefold() == session["label"].casefold():
        return engine.answer(
            question.id, TeachingReply(id=ident, response="yes", question_version=question.version)
        )
    if proposed:
        return engine.answer(
            question.id, TeachingReply(id=ident, response="no", text="No", question_version=question.version)
        )
    reply = TeachingReply(id=ident, text=session["label"], question_version=question.version)
    result = engine.answer(question.id, reply)
    if result.get("needs_clarification"):
        result = engine.answer(
            question.id,
            reply.model_copy(update={"id": ident + ":same", "entity_id": result["matches"][0]["id"]}),
        )
    return result


def main():
    now = [0.0]
    with tempfile.TemporaryDirectory() as folder:
        engine = WorldEngine(MemoryRepository(Path(folder)), clock=lambda: now[0])
        summary = []
        for session in sessions():
            walk = any(session["id"].startswith(p) for p in args.walk)
            engine.create_episode(session["id"])
            start = session["rows"][0]["received_at"]
            outcome = collections.Counter()
            asked = collections.Counter()
            modes = collections.Counter()
            shown = set()
            for r in session["rows"]:
                now[0] = start + (
                    float(r["media_time"]) if session["single_source"] else r["received_at"] - start
                )
                engine.observe(envelope(session, r, start))
                state = engine.state(session["id"])
                modes[state.executive.mode] += 1
                label = state.location.label if state.location.place_id else None
                outcome[
                    "unknown"
                    if not label
                    else "correct"
                    if label.casefold() == session["label"].casefold()
                    else "wrong"
                ] += 1
                question = state.attention
                if question and question.id not in shown:
                    shown.add(question.id)
                    asked[question.kind] += 1
                    teach(engine, session, question, walk)
            with engine.repository.transaction() as db:
                places = {
                    label: sum(
                        e.get("source") == "experience" for e in json.loads(payload).get("examples", [])
                    )
                    for label, payload in db.execute("SELECT label, payload FROM entities WHERE kind='place'")
                }
                familiar = collections.Counter(
                    json.loads(payload)["status"]
                    for (payload,) in db.execute(
                        "SELECT payload FROM world_place_regions WHERE json_extract(payload, '$.first_seen.episode_id')=?",
                        (session["id"],),
                    )
                )
                fixtures = db.execute(
                    "SELECT count(*) FROM world_place_regions WHERE json_array_length(json_extract(payload, '$.episodes'))>1"
                ).fetchone()[0]
                new_here = [
                    json.loads(payload)["first_seen"]["asset"]
                    for (payload,) in db.execute(
                        "SELECT payload FROM world_place_regions WHERE json_extract(payload, '$.status')='new_here'"
                        " AND json_extract(payload, '$.first_seen.episode_id')=?",
                        (session["id"],),
                    )
                ]
            minutes = (now[0] - start) / 60
            row = dict(
                session=session["id"][:8],
                label=session["label"],
                walk=walk,
                frames=len(session["rows"]),
                minutes=round(minutes, 1),
                location=dict(outcome),
                questions=dict(asked),
                questions_per_minute=round(sum(asked.values()) / max(minutes, 1 / 60), 2),
                modes=dict(modes),
                learned_views=places,
                new_regions=dict(familiar),
                regions_seen_on_several_visits=fixtures,
                new_here_assets=new_here,
            )
            summary.append(row)
            print(json.dumps(row))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
