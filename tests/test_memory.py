import pytest
from grounded_memory.store import MemoryRepository


def test_duplicate_names_require_identity_clarification_and_survive_restart(tmp_path):
    store = MemoryRepository(tmp_path)
    first = store.teach(kind="object", text="clock", label="clock", evidence={"asset": "digital"})
    result = store.teach(kind="object", text="clock", label="clock", evidence={"asset": "analog"})
    assert result["needs_clarification"]
    second = store.teach(
        kind="object",
        text="clock",
        label="clock",
        evidence={"asset": "analog"},
        resolution="same_kind",
        previous_id=first["entity_id"],
    )
    assert first["entity_id"] != second["entity_id"]
    assert len(MemoryRepository(tmp_path).entities("object")) == 2


def test_places_share_names_without_merging(tmp_path):
    s = MemoryRepository(tmp_path)
    a = s.teach(kind="place", text="Kitchen", label="Kitchen", evidence={})
    assert s.teach(kind="place", text="Kitchen", label="Kitchen", evidence={})["needs_clarification"]
    b = s.teach(kind="place", text="Kitchen", label="Kitchen", evidence={}, resolution="unrelated")
    assert a["entity_id"] != b["entity_id"]


def test_model_additions_are_rejected_transactionally(tmp_path):
    s = MemoryRepository(tmp_path)
    with pytest.raises(ValueError):
        s.teach(kind="object", text="clock", label="clock", categories=["timekeeping device"], evidence={})
    assert s.entities() == []


def test_replay_is_idempotent_and_claim_revocation_retains_evidence(tmp_path):
    s = MemoryRepository(tmp_path)
    e = dict(id="source:1", session_id="session", source_id="source", media_time=0.5)
    assert s.observe(e)
    assert not s.observe(e)
    s.teach(
        kind="object",
        text="digital clock, clock",
        label="digital clock",
        categories=["clock"],
        evidence={"asset": "one"},
    )
    entity = s.entities()[0]
    s.revoke(entity["claims"][0]["id"])
    entity = s.entities()[0]
    assert entity["examples"] == [{"asset": "one"}]
    assert len(entity["claims"]) == 1


def test_unknown_identity_cannot_overwrite_other_entity(tmp_path):
    s = MemoryRepository(tmp_path)
    with pytest.raises(ValueError):
        s.teach(
            kind="object", text="clock", label="clock", evidence={}, resolution="same", previous_id="invented"
        )


def test_same_identity_accumulates_views(tmp_path):
    s = MemoryRepository(tmp_path)
    first = s.teach(kind="object", text="clock", label="clock", evidence={"angle": 1})
    s.teach(
        kind="object",
        text="clock",
        label="clock",
        evidence={"angle": 2},
        resolution="same",
        previous_id=first["entity_id"],
    )
    assert len(s.entities()) == 1
    assert len(s.entities()[0]["examples"]) == 2


def test_identity_confirmation_can_be_undone_without_deleting_evidence(tmp_path):
    s = MemoryRepository(tmp_path)
    a = s.teach(kind="object", text="clock", label="clock", evidence={"asset": "first"})
    s.teach(
        kind="object",
        text="clock",
        label="clock",
        evidence={"asset": "second"},
        resolution="same",
        previous_id=a["entity_id"],
    )
    entity = s.entities()[0]
    claim = next(c for c in entity["claims"] if c["predicate"] == "identity_confirmed")
    s.revoke(claim["id"])
    assert s.entities()[0]["examples"] == [{"asset": "first"}]
    with s.connect() as db:
        assert db.execute("SELECT count(*) FROM utterances").fetchone()[0] == 2


def test_clock_subtypes_share_a_grounded_category(tmp_path):
    s = MemoryRepository(tmp_path)
    a = s.teach(kind="object", text="clock", label="clock", evidence={"asset": "digital"})
    s.teach(
        kind="object",
        text="analog clock, both are clock",
        label="analog clock",
        evidence={"asset": "analog"},
        resolution="subtype",
        previous_id=a["entity_id"],
    )
    categories = s.entities("object_category")
    clock = next(c for c in categories if c["label"] == "clock")
    analog = next(c for c in categories if c["label"] == "analog clock")
    assert any(c["predicate"] == "subtype_of" and c["object"] == clock["id"] for c in analog["claims"])
    assert len(s.entities("object")) == 2
