"""Acceptance of epistemic behavior, using disposable worlds and controllable clocks."""

import sqlite3

import pytest
from grounded_memory.store import MemoryRepository
from grounded_memory.world.contracts import BeliefRevision, ObservationEnvelope, TeachingReply
from grounded_memory.world.engine import Conflict, WorldEngine
from grounded_memory.world.repository import get, rows


@pytest.fixture
def learner(tmp_path):
    now = [100.0]
    engine = WorldEngine(MemoryRepository(tmp_path), clock=lambda: now[0])
    engine.create_episode("e")
    return engine, now


def sample(
    engine, name, t, *, episode="e", modality="vision", features=None, measurements=None, source=None, **extra
):
    obs = ObservationEnvelope(
        id=name,
        episode_id=episode,
        source_id=source or name,
        sensor_id=modality,
        clock_id="boot",
        timestamp=t,
        duration=1,
        modality=modality,
        features=features if features is not None else ([1, 0] if modality == "vision" else []),
        encoder="test",
        measurements=measurements or {},
        **extra,
    )
    engine.observe(obs)
    return obs


def teach_place(engine, label="Kitchen", features=None, **extra):
    sample(engine, "place-" + label, 0, features=features, **extra)
    q = next(q for q in engine.state("e").questions if q.kind == "locate" and q.status != "answered")
    return engine.answer(q.id, TeachingReply(id="teach-" + label, text=label))["entity_id"]


def test_first_awakening_unknown_is_not_an_entity_and_restart_relocalizes(learner, tmp_path):
    engine, _ = learner
    assert engine.state("e").location.place_id is None
    place = teach_place(engine)
    assert engine.state("e").location.place_id == place
    learner_id = engine.state().learner_id
    restarted = WorldEngine(MemoryRepository(tmp_path))
    assert restarted.state().learner_id == learner_id
    assert restarted.state("e").location.place_id is None
    for t in range(1, 4):
        sample(restarted, f"fresh{t}", t)
    assert restarted.state("e").location.place_id == place
    assert all(e["label"] != "Unknown place" for e in engine.memory.entities())


def test_similar_rooms_remain_uncertain_and_relocation_discards_old_place(learner):
    engine, _ = learner
    first = teach_place(engine)
    engine.memory.teach(
        kind="place",
        text="Other kitchen",
        label="Other kitchen",
        evidence={"features": [1, 0], "encoder": "test"},
    )
    engine.create_episode("other")
    for t in range(4):
        sample(engine, f"ambiguous{t}", t, episode="other")
    assert engine.state("other").location.place_id is None
    for t in range(1, 6):
        sample(engine, f"moved{t}", t, features=[0, 1])
    assert engine.state("e").location.place_id != first


def test_retry_and_replay_do_not_inflate_support(learner):
    engine, _ = learner
    obs = sample(engine, "same", 0, modality="event", measurements={"event": "flow"}, source="recording")
    assert engine.observe(obs) is False
    engine.create_episode("replay")
    replay = obs.model_copy(update={"id": "new-id", "episode_id": "replay"})
    assert engine.observe(replay) is False
    with engine.repository.transaction() as db:
        assert len(rows(db, "observations")) == 1
        assert len(rows(db, "events")) == 1
    with pytest.raises(Conflict):
        engine.observe(obs.model_copy(update={"measurements": {"event": "different"}}))


def test_audio_naming_and_context_surprise_are_learned(learner):
    engine, _ = learner
    kitchen = teach_place(engine)
    sample(engine, "water-first", 1, modality="audio", features=[1, 0], measurements={"rms": 0.1})
    q = next(q for q in engine.state("e").questions if q.kind == "identify")
    sound = engine.answer(q.id, TeachingReply(id="water-name", text="Running water"))["entity_id"]
    for t in (10, 20, 30, 40):
        sample(engine, f"kitchen{t}", t, modality="audio", features=[1, 0], measurements={"rms": 0.1})
    living = engine.memory.teach(kind="place", text="Living room", label="Living room", evidence={})[
        "entity_id"
    ]
    engine.set_place("e", living, "Living room")
    for t in (50, 60, 70, 80):
        sample(engine, f"quiet{t}", t, modality="audio", features=[0, 1], measurements={"rms": 0})
    sample(engine, "unexpected", 90, modality="audio", features=[1, 0], measurements={"rms": 0.1})
    event = engine.state("e").model_dump()["recent_events"][-1]
    assert event["surprising"]
    assert event["key"] == "sound:" + sound
    assert event["heard_at"] == living and event["source_location"] is None
    assert any(
        q.kind == "explain" and q.reason == "Unexpected in this place" for q in engine.state("e").questions
    )
    assert kitchen != living


def test_taught_prediction_is_contested_without_erasing_teacher(learner):
    engine, _ = learner
    teach_place(engine)
    sample(engine, "flash", 1, measurements={"brightness": 0.1})
    sample(engine, "flash2", 2, measurements={"brightness": 0.8})
    q = next(q for q in engine.state("e").questions if q.kind == "explain")
    result = engine.answer(
        q.id,
        TeachingReply(
            id="why", text="When the light changes, a bell rings", expected_event="bell", horizon_seconds=3
        ),
    )
    # Repeated brightness change triggers the taught forecast.
    sample(engine, "reset", 10, measurements={"brightness": 0.8})
    sample(engine, "again", 11, measurements={"brightness": 0.1})
    for t in (11, 12, 13, 14):
        sample(engine, f"listen{t}", t, modality="event", measurements={})
    state = engine.state("e")
    belief = next(b for b in state.model_dump()["beliefs"] if b["id"] == result["belief_id"])
    assert belief["status"] == "contested"
    assert belief["value"] == "When the light changes, a bell rings"
    assert any(q.kind == "clarify" for q in state.questions)
    with engine.repository.transaction() as db:
        assert get(db, "teachings", "why")["text"] == belief["value"]


def test_missing_sensor_does_not_disprove_prediction(learner):
    engine, _ = learner
    teach_place(engine)
    sample(engine, "base", 1, measurements={"brightness": 0})
    sample(engine, "flash", 2, measurements={"brightness": 1})
    q = next(q for q in engine.state("e").questions if q.kind == "explain")
    result = engine.answer(
        q.id, TeachingReply(id="explain", text="A bell follows", expected_event="bell", horizon_seconds=2)
    )
    sample(engine, "base2", 10, measurements={"brightness": 1})
    sample(engine, "flash2", 11, measurements={"brightness": 0})
    sample(engine, "later", 20, measurements={"brightness": 0})
    state = engine.state("e")
    assert (
        next(b for b in state.model_dump()["beliefs"] if b["id"] == result["belief_id"])["status"] == "active"
    )
    assert state.model_dump()["predictions"][-1]["status"] == "unobserved"


def test_playback_and_model_guesses_cannot_teach_themselves(learner):
    engine, _ = learner
    engine.playback("e", "boot", 0, 5)
    sample(engine, "speech", 1, modality="audio", measurements={"rms": 0.2}, features=[1, 0])
    sample(
        engine, "guess", 6, modality="event", measurements={"event": "secret fact"}, origin="model_hypothesis"
    )
    assert engine.state("e").recent_events == []
    with engine.repository.transaction() as db:
        assert len(rows(db, "observations")) == 2
        assert all(o["excluded"] for o in rows(db, "observations"))


def test_attention_budget_deferral_and_frozen_reply_idempotency(learner):
    engine, now = learner
    sample(engine, "first", 0)
    q = engine.state("e").attention
    now[0] += 21
    assert engine.state("e").attention is None
    sample(engine, "noise", 1, modality="audio", measurements={"rms": 0.1}, features=[1, 0])
    assert engine.state("e").attention is None
    now[0] += 40
    assert engine.state("e").attention.kind == "identify"
    reply = TeachingReply(id="late-answer", text="Bedroom")
    result = engine.answer(q.id, reply)
    assert engine.answer(q.id, reply) == result
    with pytest.raises(Conflict):
        engine.answer(q.id, TeachingReply(id="another", text="Kitchen"))


def test_revision_preserves_history_and_checks_concurrent_version(learner):
    engine, _ = learner
    teach_place(engine)
    belief = next(b for b in engine.state().model_dump()["beliefs"] if b["predicate"] == "named")
    revision = BeliefRevision(
        id="revision", expected_version=1, operation="correct", reason="I used the wrong name", value="Office"
    )
    result = engine.revise(belief["id"], revision)
    assert result["value"] == "Office" and result["version"] == 2
    assert engine.revise(belief["id"], revision)["version"] == 2
    with pytest.raises(Conflict):
        engine.revise(belief["id"], revision.model_copy(update={"id": "conflict"}))
    assert engine.state().model_dump()["revisions"][-1]["before"]["value"] == "Kitchen"


def test_migration_preserves_claims_media_and_pairing(tmp_path):
    memory = MemoryRepository(tmp_path)
    memory.teach(kind="object", text="Clock", label="Clock", evidence={"asset": "original.mov"})
    claim = memory.entities()[0]["claims"][0]
    memory.revoke(claim["id"])
    (tmp_path / "assets" / "original.mov").write_bytes(b"original")
    (tmp_path / "pairing.key").write_bytes(b"pairing")
    engine = WorldEngine(memory)
    assert (tmp_path / "assets" / "original.mov").read_bytes() == b"original"
    assert (tmp_path / "pairing.key").read_bytes() == b"pairing"
    assert (
        next(b for b in engine.state().model_dump()["beliefs"] if b["id"] == "legacy:" + claim["id"])[
            "status"
        ]
        == "retracted"
    )
    backup = tmp_path / "backups" / "pre-world-v1.sqlite3"
    with sqlite3.connect(backup) as db:
        assert db.execute("SELECT max(version) FROM schema_version").fetchone()[0] == 1
    WorldEngine(memory)
    with memory.connect() as db:
        assert (
            db.execute(
                "SELECT count(*) FROM world_beliefs WHERE id=?", ("legacy:" + claim["id"],)
            ).fetchone()[0]
            == 1
        )


def test_audio_does_not_prevent_answering_the_place_question(learner):
    engine, _ = learner
    sample(engine, "view", 0)
    q = engine.state("e").attention
    sample(engine, "audio", 0.5, modality="audio", features=[0, 1], measurements={"rms": 0.1})
    engine.answer(q.id, TeachingReply(id="place-answer", text="Study"))
    assert engine.state("e").location.label == "Study"


def test_free_text_unknown_is_not_learned_as_a_name(learner):
    engine, _ = learner
    sample(engine, "view", 0)
    q = engine.state("e").attention
    reply = TeachingReply(id="unknown", text="I don't know")
    assert engine.answer(q.id, reply) == engine.answer(q.id, reply)
    assert engine.memory.entities() == []
    assert engine.state("e").location.place_id is None


def test_retention_keeps_frozen_evidence_and_periodic_frames_only(learner):
    engine, _ = learner
    retained = []
    for t in range(10):
        observation = ObservationEnvelope(
            id=f"frame{t}",
            episode_id="e",
            source_id="recording",
            sensor_id="camera",
            clock_id="boot",
            timestamp=t,
            modality="vision",
            features=[1, 0],
            encoder="test",
            evidence_refs=[f"frame{t}.jpg"],
        )
        engine.observe(observation, retain_evidence=retained.extend)
        assert engine.observe(observation, retain_evidence=retained.extend) is False
    assert retained == ["frame0.jpg", "frame5.jpg"]
    assert engine.observation("frame1")["recording_reference"]["timestamp"] == 1


def test_losing_a_known_location_can_raise_a_new_place_question(learner):
    engine, now = learner
    kitchen = teach_place(engine)
    for t in (1, 2, 3, 4):
        sample(engine, f"moved{t}", t, features=[0, 1])
    # Without pose, one abrupt view change ends continuity; later similar views are not the kitchen.
    assert all(e.get("source") != "experience" for e in place_examples(engine, kitchen))
    unanswered = [q for q in engine.state("e").questions if q.kind == "locate" and q.status != "answered"]
    assert len(unanswered) == 1
    engine.answer(unanswered[0].id, TeachingReply(id="new-room", text="Bedroom"))
    assert engine.state("e").location.label == "Bedroom"


def test_identity_repair_changes_recognition_without_destroying_originals(learner):
    engine, _ = learner
    memory = engine.memory
    first = memory.teach(kind="object", text="Cup", label="Cup", evidence={"features": [1, 0]})["entity_id"]
    second = memory.teach(kind="object", text="Mug", label="Mug", evidence={"features": [0, 1]})["entity_id"]
    engine.sync_legacy()
    belief = next(b for b in engine.state().model_dump()["beliefs"] if b["subject"] == first)
    engine.revise(
        belief["id"],
        BeliefRevision(
            id="identity",
            expected_version=1,
            operation="identity",
            reason="Those were two views of the same cup",
            target_entity_id=second,
        ),
    )
    assert len(memory.entities("object")) == 2
    view = engine.recognizable_entities("object")
    assert len(view) == 1 and len(view[0]["examples"]) == 2
    identity = next(b for b in engine.state().model_dump()["beliefs"] if b["predicate"] == "same_as")
    engine.revise(
        identity["id"],
        BeliefRevision(
            id="undo-identity",
            expected_version=1,
            operation="retract",
            reason="They were different cups after all",
        ),
    )
    assert len(engine.recognizable_entities("object")) == 2


def test_late_capture_does_not_restart_a_stopped_episode(learner):
    engine, _ = learner
    sample(engine, "initial", 0)
    engine.control("e", "stop")
    sample(engine, "late", 1)
    assert engine.state("e").status == "stopped"
    assert engine.state("e").attention is None


def test_delayed_place_answer_does_not_assign_an_old_view_to_current_location(learner):
    engine, _ = learner
    sample(engine, "old-view", 0)
    question = engine.state("e").attention
    sample(engine, "later-view", 40, features=[1, 0])
    engine.answer(question.id, TeachingReply(id="late-place", text="Office"))
    assert engine.state("e").location.place_id is None
    for t in (41, 42, 43):
        sample(engine, f"confirm{t}", t, features=[1, 0])
    assert engine.state("e").location.label == "Office"


def test_review_playback_is_excluded_from_another_live_capture(learner):
    engine, now = learner
    sample(engine, "mic-before", 0, modality="audio", features=[1, 0], measurements={"rms": 0})
    engine.system_output(3)
    now[0] += 1
    sample(engine, "speaker", 1, modality="audio", features=[1, 0], measurements={"rms": 0.1})
    assert engine.observation("speaker")["excluded"]
    assert engine.state("e").recent_events == []


def pose(x):
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1]


def place_examples(engine, place_id):
    return [e for e in engine.recognizable_entities("place") if e["id"] == place_id][0]["examples"]


def teach_with_pose(engine):
    return teach_place(engine, features=[1, 0, 0], tracking_frame_id="ar", measurements={"pose": pose(0)})


def test_turning_within_a_taught_place_learns_new_views_that_later_recognize_it(learner):
    engine, _ = learner
    kitchen = teach_with_pose(engine)
    # The teacher named the first view; turning around shows an unrelated view from the same spot.
    for t in (1, 2, 3, 4):
        sample(
            engine,
            f"turned{t}",
            t,
            features=[0, 1, 0],
            tracking_frame_id="ar",
            measurements={"pose": pose(0.5)},
        )
    learned = [e for e in place_examples(engine, kitchen) if e.get("source") == "experience"]
    assert [e["observation_id"] for e in learned] == ["turned1"]
    assert learned[0]["basis"] == "continuity"
    engine.create_episode("tomorrow")
    for t in (10, 11, 12):
        sample(engine, f"back{t}", t, episode="tomorrow", features=[0, 1, 0])
    location = engine.state("tomorrow").location
    assert (location.place_id, location.certainty) == (kitchen, "recognized")
    assert location.candidates[0].example_source == "experience"


def test_continuity_does_not_carry_a_place_into_another_room(learner):
    engine, _ = learner
    kitchen = teach_place(engine, features=[1, 0, 0])
    sample(
        engine, "taught-pose", 1, features=[1, 0, 0], tracking_frame_id="ar", measurements={"pose": pose(0)}
    )
    # Walking 5 m away ends measured continuity even though the new views resemble each other.
    for t in (2, 3, 4, 5):
        sample(
            engine, f"far{t}", t, features=[0, 0, 1], tracking_frame_id="ar", measurements={"pose": pose(5)}
        )
    assert not [e for e in place_examples(engine, kitchen) if e.get("source") == "experience"]
    assert engine.state("e").location.place_id is None


def test_rejecting_a_place_guess_stops_trusting_the_learned_view_but_keeps_it(learner):
    engine, _ = learner
    kitchen = teach_with_pose(engine)
    sample(
        engine, "turned", 1, features=[0.6, 0.8, 0], tracking_frame_id="ar", measurements={"pose": pose(0)}
    )
    engine.create_episode("later")
    sample(engine, "lookalike", 20, episode="later", features=[0.6, 0.8, 0.05])
    question = next(q for q in engine.state("later").questions if q.kind == "locate")
    assert question.proposition.entity_id == kitchen
    assert question.proposition.candidates[0].example_source == "experience"
    engine.answer(question.id, TeachingReply(id="not-kitchen", response="no", text="No"))
    learned = [e for e in place_examples(engine, kitchen) if e.get("source") == "experience"]
    assert learned and learned[0]["excluded"] and learned[0]["excluded_by"] == "not-kitchen"
    for t in (21, 22, 23):
        sample(engine, f"again{t}", t, episode="later", features=[0.6, 0.8, 0.05])
    assert engine.state("later").location.place_id is None


def test_executive_mode_follows_evidence_and_working_set_stays_bounded(learner):
    engine, now = learner
    assert engine.state("e").executive.mode == "EXPLORE"
    sample(engine, "first", 0, features=[1, 0, 0])
    control = engine.state("e").executive
    assert (control.mode, control.request) == ("NOVEL", "Tell me where we are")
    q = engine.state("e").attention
    kitchen = engine.answer(q.id, TeachingReply(id="named", text="Kitchen"))["entity_id"]
    control = engine.state("e").executive
    assert control.mode == "CONFIRMED"
    assert control.working_set[0].kind == "place" and control.working_set[0].id == kitchen
    for t in (1, 2, 3, 4):
        sample(engine, f"elsewhere{t}", t, features=[0, 0, 1])
    assert engine.state("e").executive.mode == "RELOCALIZE"
    engine.create_episode("partial")
    sample(engine, "resembles", 10, episode="partial", features=[0.75, 0.66, 0])
    control = engine.state("partial").executive
    assert control.mode == "VERIFY" and "Kitchen" in control.request
    now[0] += 11
    sample(engine, "covered", 20, episode="partial", features=[0, 0, 0.1], quality=0.05)
    control = engine.state("partial").executive
    assert control.mode == "RECOVER" and "too dark or blurred" in control.reason
    now[0] += 11
    assert engine.state("partial").executive.reason == "Camera evidence stopped arriving"
    assert len(engine.state("e").executive.working_set) <= 12


def region(name, features):
    return dict(id=name, asset=name + ".png", bbox=[0.1, 0.1, 0.2, 0.2], salience=0.5, features=features)


def test_revisit_marks_returning_regions_as_fixtures_and_asks_about_what_is_new(learner):
    engine, _ = learner
    kitchen = teach_place(engine, features=[1, 0, 0])
    chair = region("chair", [0, 1, 0, 0])
    sample(engine, "first-look", 1, features=[1, 0, 0], measurements={"regions": [chair]})
    engine.create_episode("revisit")
    ball = region("ball", [0, 0, 0, 1])
    fragment = region("fragment", [0, 0, 1, 0])
    for t in (10, 11, 12, 13, 14, 15):
        regions = [region(f"chair{t}", [0, 0.99, 0.1, 0])]
        regions += [ball] if t >= 13 else [fragment] if t == 12 else []
        sample(
            engine, f"back{t}", t, episode="revisit", features=[1, 0, 0], measurements={"regions": regions}
        )
    assert engine.state("revisit").location.place_id == kitchen
    with engine.repository.transaction() as db:
        memory = {r["first_seen"]["asset"]: r for r in rows(db, "place_regions")}
    assert memory["chair.png"]["episodes"] == ["e", "revisit"]
    assert memory["chair.png"]["stability"] == 1.0
    # Persisting across separate frames is required; a one-frame mask fragment stays a candidate.
    assert memory["ball.png"]["status"] == "new_here"
    assert memory["fragment.png"]["status"] == "new_candidate"
    asked = {q.region.id: q for q in engine.state().questions if q.region}
    assert asked["ball"].text == "This wasn't here before. What is it?"
    assert asked["fragment"].text == "What is this?"
    assert asked["ball"].priority > asked["chair"].priority


def test_an_unseen_viewpoint_is_missing_coverage_not_a_change(learner):
    engine, _ = learner
    teach_place(engine, features=[1, 0, 0], tracking_frame_id="ar", measurements={"pose": pose(0)})
    engine.create_episode("revisit")
    for t in (10, 11, 12):
        sample(engine, f"back{t}", t, episode="revisit", features=[1, 0, 0], tracking_frame_id="ar2")
    # Turning to a view the first visit never saw: the place is held by measured continuity only.
    sample(
        engine,
        "new-angle",
        14,
        episode="revisit",
        features=[0, 1, 0],
        tracking_frame_id="ar2",
        measurements={"pose": pose(0.3), "regions": [region("lamp", [0, 0, 1, 0])]},
    )
    with engine.repository.transaction() as db:
        (lamp,) = rows(db, "place_regions")
    assert lamp["status"] == "unseen"


def test_movement_is_reported_while_the_camera_turns_but_not_from_blur_or_walking(learner):
    engine, _ = learner
    teach_place(engine, features=[1, 0, 0])
    box = [0.4, 0.4, 0.2, 0.2]
    turning = dict(world_motion=0.05, world_motion_box=box, global_motion=0.3, local_motion=0.4)
    sample(engine, "hand", 1, features=[1, 0, 0], measurements=turning)
    sample(engine, "blurred", 2, features=[1, 0, 0], quality=0.2, measurements=turning)
    sample(engine, "whole-frame", 3, features=[1, 0, 0], measurements=dict(turning, world_motion=0.6))
    sample(engine, "step-a", 4, features=[1, 0, 0], measurements=dict(pose=pose(0)))
    sample(engine, "step-b", 5, features=[1, 0, 0], measurements=dict(turning, pose=pose(0.5)))
    with engine.repository.transaction() as db:
        moved = [e for e in rows(db, "events") if e["key"] == "change:motion"]
    assert [(e["observation_id"], e["box"]) for e in moved] == [("hand", box)]


def event(engine, name, t, what, episode="e"):
    return sample(engine, name, t, episode=episode, modality="event", measurements={"event": what})


def test_repeated_order_asks_for_an_intervention_and_only_a_performed_cause_tests_it(learner):
    engine, _ = learner
    teach_place(engine)
    for t in (10, 40, 70):
        event(engine, f"drawer{t}", t, "drawer opened")
        event(engine, f"chime{t}", t + 2, "chime")
    question = next(q for q in engine.state("e").questions if q.kind == "intervene")
    assert question.text.startswith("chime has followed drawer opened 3 times")
    belief_id = question.proposition.belief_id
    # A passive repetition is a forecast, not an intervention.
    event(engine, "drawer100", 100, "drawer opened")
    event(engine, "chime100", 102, "chime")
    with engine.repository.transaction() as db:
        assert not get(db, "beliefs", belief_id).get("interventions")
    engine.answer(question.id, TeachingReply(id="will-do", response="yes", text="yes"))
    event(engine, "drawer-test", 130, "drawer opened")
    event(engine, "chime-test", 133, "chime")
    with engine.repository.transaction() as db:
        belief = get(db, "beliefs", belief_id)
    assert [i["outcome"] for i in belief["interventions"]] == ["effect followed"]
    assert belief["certainty"] == "tested" and "intervention_request" not in belief


def test_declining_an_intervention_is_not_evidence_against_the_expectation(learner):
    engine, _ = learner
    teach_place(engine)
    for t in (10, 40, 70):
        event(engine, f"drawer{t}", t, "drawer opened")
        event(engine, f"chime{t}", t + 2, "chime")
    question = next(q for q in engine.state("e").questions if q.kind == "intervene")
    engine.answer(question.id, TeachingReply(id="cannot", response="no", text="No"))
    with engine.repository.transaction() as db:
        belief = get(db, "beliefs", question.proposition.belief_id)
    assert belief["status"] == "active" and not belief["history"]
    assert next(q for q in engine.state("e").questions if q.id == question.id).status == "dismissed"
