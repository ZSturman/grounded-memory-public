"""Application workflows, pagination, concurrent review and held-out boundaries."""

import importlib
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from grounded_memory.pipeline import collection, datasets, labeling, workspaces
from grounded_memory.pipeline.contracts import Annotation, Collected, DatasetSpec
from grounded_memory.pipeline.storage import Ledger


def note(ledger, identity, source=None):
    return collection.collect(
        ledger,
        Collected(
            id=identity,
            source_id=source or identity,
            source_kind="user-note",
            source_timestamp=10,
            group_ids=[source or identity],
            origin="human",
            payload={"text": "Evidence " + identity},
        ),
    )


def review(ledger, identity):
    return labeling.annotate(
        ledger,
        identity,
        Annotation(
            actor="person", method="human", reviewed=True, text="Memory " + identity, fact_keys=[identity]
        ),
    )


def test_queue_filters_before_pagination_and_triage_is_not_truth(tmp_path):
    ledger = Ledger(tmp_path)
    for i in range(38):
        note(ledger, str(i))
        if i < 33:
            review(ledger, str(i))
    assert workspaces.records(ledger)["total"] == 5
    assert [v["id"] for v in workspaces.records(ledger, limit=2)["items"]] == ["33", "34"]
    workspaces.triage(ledger, ["34"], "flag", "reviewer")
    assert workspaces.records(ledger)["items"][0]["id"] == "34"
    workspaces.triage(ledger, ["33", "34"], "skip", "reviewer")
    assert workspaces.records(ledger)["total"] == 3
    assert workspaces.records(ledger, state="skipped")["total"] == 2
    assert labeling.latest(ledger, "34") is None
    with pytest.raises(KeyError):
        workspaces.triage(ledger, ["35", "missing"], "flag", "reviewer")
    assert workspaces.records(ledger, state="flagged")["total"] == 0
    assert workspaces.records(ledger, search="Evidence 37")["total"] == 1


def test_import_preferences_checkpoint_history_and_read_only_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "memory.sqlite3"
    with sqlite3.connect(path) as db:
        for table in ("observations", "world_observations"):
            db.execute(f"CREATE TABLE {table}(id TEXT,payload TEXT)")
            db.execute(
                f"INSERT INTO {table} VALUES (?,?)",
                (table, json.dumps({"timestamp": 10, "session_id": "session"})),
            )
    original = path.read_bytes()
    ledger = Ledger(tmp_path / "ledger")
    workspaces.configure_sources(ledger, ["world_observations"])
    result = collection.import_memory(ledger, source, sources=workspaces.source_settings(ledger))
    assert result["imported"] == 1
    assert (
        collection.import_memory(ledger, source, sources=workspaces.source_settings(ledger))["imported"] == 0
    )
    state = workspaces.collection_status(ledger, source)
    assert next(s for s in state["sources"] if s["kind"] == "observations")["pending"] == 1
    assert len(state["recent_runs"]) == 2
    workspaces.configure_sources(ledger, list(collection.SOURCE_KINDS))
    assert (
        collection.import_memory(ledger, source, sources=workspaces.source_settings(ledger))["imported"] == 1
    )
    assert path.read_bytes() == original


def test_dataset_planner_keeps_connected_groups_and_holdouts(tmp_path):
    ledger = Ledger(tmp_path)
    for identity, source in [("a", "shared"), ("b", "shared"), ("c", "independent")]:
        note(ledger, identity, source)
        review(ledger, identity)
    plan = workspaces.dataset_plan(ledger, ["a", "b", "c"])
    assert sorted(len(g["record_ids"]) for g in plan["groups"]) == [1, 2]
    datasets.prepare(
        ledger,
        DatasetSpec(
            name="Holdout",
            assignments={"a": "test", "b": "test"},
            evaluation_time=10,
            reviewer="reviewer",
            purpose="reviewed-local",
            cases=[
                dict(
                    id="q",
                    split="test",
                    query="memory",
                    expected_ids=["a"],
                    categories=["accuracy"],
                    group_ids=["shared"],
                )
            ],
        ),
    )
    assert workspaces.dataset_plan(ledger, ["b"])["groups"] == [
        dict(record_ids=["b"], split="test", locked=True)
    ]
    labeling.set_policy(ledger, "b", False, "reviewer")
    with pytest.raises(ValueError, match="eligible"):
        workspaces.dataset_plan(ledger, ["b"])


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GM_PAIR_CODE", "123456")
    from grounded_memory import api

    importlib.reload(api)
    with TestClient(api.app) as client:
        assert client.get("/api/v2/pipeline/records").status_code == 401
        assert client.post("/api/pair", json={"code": "123456"}).status_code == 200
        yield client


def test_http_complete_journey_and_safe_readiness(client):
    prefix = "/api/v2/pipeline"
    fixture = json.loads((Path(__file__).parent / "fixtures/pipeline/regression-v1.json").read_text())
    for record in fixture["records"]:
        assert client.post(prefix + "/collect", json=record["raw"]).status_code == 200
        assert (
            client.post(
                prefix + "/annotations",
                json={"record_id": record["raw"]["id"], "annotation": record["annotation"]},
            ).status_code
            == 200
        )
    dataset = client.post(prefix + "/datasets", json=fixture["spec"]).json()
    candidate = client.post(
        prefix + "/experiments",
        json={"dataset_id": dataset["id"], "config": {"algorithm": "coverage", "min_score": 0.5}},
    ).json()
    assert not client.get(prefix + "/readiness", params={"experiment_id": candidate["id"]}).json()["ready"]
    assert client.get(prefix + "/items/deployment").json() == []
    for split in ("validation", "test", "regression"):
        result = client.post(prefix + "/evaluations", json={"experiment_id": candidate["id"], "split": split})
        assert result.status_code == 200, result.text
    readiness = client.get(prefix + "/readiness", params={"experiment_id": candidate["id"]}).json()
    assert readiness["ready"]
    assert client.get(prefix + "/items/deployment").json() == []
    deployed = client.post(
        prefix + "/promote",
        json={
            "experiment_id": candidate["id"],
            "evaluation_ids": readiness["evaluation_ids"],
            "actor": "test-human",
        },
    ).json()
    assert client.get(prefix + "/health").json()["healthy"]
    assert client.post(prefix + "/recall", json={"query": "Garage key"}).json()["hits"]
    # Restore a genuinely different previous configuration, not just the current one.
    second = client.post(
        prefix + "/experiments",
        json={"dataset_id": dataset["id"], "config": {"algorithm": "coverage", "min_score": 0.5, "top_k": 4}},
    ).json()
    for split in ("test", "regression"):
        assert (
            client.post(
                prefix + "/evaluations", json={"experiment_id": second["id"], "split": split}
            ).status_code
            == 200
        )
    second_ready = client.get(prefix + "/readiness", params={"experiment_id": second["id"]}).json()
    assert second_ready["ready"]
    assert (
        client.post(
            prefix + "/promote",
            json={
                "experiment_id": second["id"],
                "evaluation_ids": second_ready["evaluation_ids"],
                "actor": "test-human",
            },
        ).status_code
        == 200
    )
    assert (
        client.post(
            prefix + "/rollback", json={"deployment_id": deployed["id"], "actor": "test-human"}
        ).json()["action"]
        == "rollback"
    )
    assert client.get(prefix + "/status").json()["deployed"]["config"]["top_k"] == 3
    client.post(
        prefix + "/policy", json={"record_id": "test-current", "allowed": False, "actor": "test-human"}
    )
    record = client.get(prefix + "/record", params={"id": "test-current"}).json()
    assert record["allowed"] is False
    assert record["raw"]["id"] == "test-current"
    assert not client.get(prefix + "/health").json()["healthy"]
    assert not client.get(prefix + "/readiness", params={"experiment_id": candidate["id"]}).json()["ready"]
    assert (
        client.post(
            prefix + "/rollback", json={"deployment_id": deployed["id"], "actor": "test-human"}
        ).status_code
        == 422
    )
