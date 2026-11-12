"""Stage contracts, integrity boundaries, and deterministic final-evaluation gates."""

import json
import sqlite3
from pathlib import Path

import pytest
from grounded_memory.pipeline import (
    collection,
    datasets,
    deployment,
    evaluation,
    experiments,
    labeling,
    service,
)
from grounded_memory.pipeline.contracts import Annotation, Collected, DatasetSpec, Strategy
from grounded_memory.pipeline.storage import Ledger
from grounded_memory.store import MemoryRepository

FIXTURE = Path(__file__).parent / "fixtures/pipeline/regression-v1.json"


@pytest.fixture
def prepared(tmp_path):
    fixture = json.loads(FIXTURE.read_text())
    ledger = Ledger(tmp_path / "pipeline")
    for item in fixture["records"]:
        collection.collect(ledger, Collected.model_validate(item["raw"]))
        labeling.annotate(ledger, item["raw"]["id"], Annotation.model_validate(item["annotation"]))
    spec = DatasetSpec.model_validate(fixture["spec"])
    return ledger, spec, datasets.prepare(ledger, spec)


def approved(ledger, dataset):
    experiment = experiments.create(ledger, dataset["id"], Strategy(algorithm="coverage", min_score=0.5))
    results = [evaluation.evaluate(ledger, experiment["id"], split) for split in ("test", "regression")]
    return deployment.promote(ledger, experiment["id"], [r["id"] for r in results], "test-human")


def test_raw_identity_conflicts_and_independent_annotation_history(prepared):
    ledger, spec, dataset = prepared
    raw = ledger.get("test-current")
    label = labeling.latest(ledger, raw["id"])
    corrected = labeling.annotate(
        ledger,
        raw["id"],
        Annotation(
            actor="reviewer",
            method="human",
            reviewed=True,
            text="Garage key is in the red drawer.",
            fact_keys=["test-location"],
            previous_revision=label["id"],
        ),
    )
    assert ledger.get(raw["id"]) == raw
    assert ledger.get(label["id"])["text"] != corrected["text"]
    with pytest.raises(ValueError, match="latest revision"):
        labeling.annotate(ledger, raw["id"], Annotation(actor="stale", method="human"))
    with pytest.raises(ValueError, match="revised sources"):
        experiments.create(ledger, dataset["id"], Strategy())
    changed = Collected.model_validate({k: v for k, v in raw.items() if k != "content_digest"})
    changed.payload = {"text": "different raw evidence"}
    with pytest.raises(ValueError, match="immutable"):
        collection.collect(ledger, changed)
    assert ledger.get(raw["id"]) == raw
    assert len(ledger.list("failure")) == 3
    assert datasets.prepare(ledger, spec)["id"] != dataset["id"]


def test_deterministic_suggestion_needs_review(tmp_path):
    ledger = Ledger(tmp_path)
    collection.collect(
        ledger,
        Collected(
            id="r",
            source_id="s",
            source_kind="note",
            source_timestamp=1,
            group_ids=["conversation"],
            origin="human",
            payload={"text": "Window open"},
        ),
    )
    first = labeling.suggest(ledger, "r")
    assert first == labeling.suggest(ledger, "r")
    assert not first["reviewed"] and first["method"] == "deterministic"
    assert service.status(ledger)["awaiting_review"] == 1
    with pytest.raises(ValueError, match="explicit human review"):
        Annotation(actor="model", method="model", reviewed=True)
    spec = DatasetSpec(
        name="unreviewed",
        evaluation_time=1,
        reviewer="person",
        purpose="reviewed-local",
        assignments={"r": "test"},
        cases=[
            dict(
                id="q",
                query="Window",
                split="test",
                expected_ids=["r"],
                categories=["accuracy"],
                group_ids=["conversation"],
            )
        ],
    )
    with pytest.raises(ValueError, match="human review"):
        datasets.prepare(ledger, spec)
    assert not ledger.list("dataset")


@pytest.mark.parametrize("collision", ["source", "conversation", "fact", "text", "payload"])
def test_no_group_leakage_even_across_dataset_versions(prepared, collision):
    ledger, spec, _ = prepared
    original = ledger.get("train-one")
    original_label = labeling.latest(ledger, "train-one")
    raw = Collected(
        id="leak",
        source_id="independent",
        source_kind="note",
        source_timestamp=1,
        group_ids=["independent"],
        origin="human",
        payload={"text": "different"},
    )
    label = Annotation(
        actor="reviewer", method="human", reviewed=True, text="Another statement", fact_keys=["unrelated"]
    )
    if collision == "source":
        raw.source_id = original["source_id"]
    if collision == "conversation":
        raw.group_ids = original["group_ids"]
    if collision == "fact":
        label.fact_keys = original_label["fact_keys"]
    if collision == "text":
        label.text = original_label["text"].upper()
    if collision == "payload":
        raw.payload = original["payload"]
    collection.collect(ledger, raw)
    labeling.annotate(ledger, raw.id, label)
    spec = DatasetSpec(
        name="leak",
        evaluation_time=1,
        assignments={"leak": "test"},
        reviewer="person",
        purpose="reviewed-local",
        cases=[
            dict(
                id="leak-query",
                split="test",
                query="other unique",
                expected_ids=["leak"],
                categories=["retrieval"],
                group_ids=["independent"],
            )
        ],
    )
    with pytest.raises(ValueError, match="Leakage"):
        datasets.prepare(ledger, spec)
    assert len(ledger.list("dataset")) == 1


def test_dataset_repeatability_missing_gold_and_current_split_leakage(prepared):
    ledger, spec, dataset = prepared
    assert datasets.prepare(ledger, spec) == dataset
    assert dataset["warnings"]
    changed = spec.model_copy(deep=True)
    changed.cases[0].expected_ids = ["missing"]
    with pytest.raises(ValueError, match="query's split"):
        datasets.prepare(ledger, changed)
    changed = spec.model_copy(deep=True)
    changed.assignments["test-past"] = "train"
    with pytest.raises(ValueError, match="Leakage"):
        datasets.prepare(ledger, changed)
    assert ledger.get(dataset["id"]) == dataset


def test_repeatable_evaluation_final_holdouts_and_append_only_runs(prepared):
    ledger, _, dataset = prepared
    candidate = experiments.create(ledger, dataset["id"], Strategy(algorithm="coverage", min_score=0.5))
    first = evaluation.evaluate(ledger, candidate["id"], "test")
    second = evaluation.evaluate(ledger, candidate["id"], "test")
    assert first["id"] != second["id"]
    assert first["result_digest"] == second["result_digest"]
    assert first["candidate"]["metrics"]["exact"] == 1
    assert first["candidate"]["metrics"]["exact"] > first["baseline"]["metrics"]["exact"]
    assert first["elapsed_seconds"] >= 0 and first["process_peak_rss"] > 0
    with pytest.raises(ValueError, match="training examples"):
        evaluation.evaluate(ledger, candidate["id"], "train")
    with pytest.raises(ValueError, match="test AND"):
        deployment.promote(ledger, candidate["id"], [first["id"], second["id"]], "person")
    assert deployment.active(ledger) is None


def test_complete_workflow_rollback_and_live_exclusion(prepared):
    ledger, _, dataset = prepared
    first = approved(ledger, dataset)
    second = approved(ledger, dataset)
    assert second["previous_id"] == first["id"]
    rollback = deployment.rollback(ledger, first["id"], "person")
    assert rollback["previous_id"] == second["id"] and rollback["rollback_to"] == first["id"]
    recall = deployment.recall(ledger, "Garage key")
    assert recall["hits"][0]["evidence_ids"] == ["test-current"]
    assert recall["deployment_id"] == rollback["id"]
    assert not deployment.recall(ledger, "Garage key", allowed_sources=[])["hits"]
    labeling.set_policy(ledger, "test-current", False, "person")
    with pytest.raises(ValueError, match="excluded"):
        deployment.recall(ledger, "Garage key")
    with pytest.raises(ValueError, match="excluded"):
        deployment.rollback(ledger, second["id"], "person")
    assert ledger.get("test-current")["payload"]["text"]


def test_bad_candidate_cannot_replace_approved_deployment(prepared):
    ledger, _, dataset = prepared
    first = approved(ledger, dataset)
    bad = experiments.create(ledger, dataset["id"], Strategy(algorithm="coverage", min_score=1, top_k=1))
    results = [evaluation.evaluate(ledger, bad["id"], split) for split in ("test", "regression")]
    with pytest.raises(ValueError, match="all curated"):
        deployment.promote(ledger, bad["id"], [r["id"] for r in results], "person")
    assert deployment.active(ledger) == first


def test_reject_changed_code_and_model_sources(prepared, monkeypatch):
    ledger, spec, dataset = prepared
    candidate = experiments.create(ledger, dataset["id"], Strategy())
    monkeypatch.setattr(evaluation, "code_version", lambda: "changed")
    with pytest.raises(ValueError, match="Code changed"):
        evaluation.evaluate(ledger, candidate["id"], "test")
    raw = Collected(
        id="model",
        source_id="model",
        source_kind="proposal",
        source_timestamp=1,
        group_ids=["model"],
        origin="model",
        payload={"text": "model says"},
    )
    collection.collect(ledger, raw)
    labeling.annotate(
        ledger,
        raw.id,
        Annotation(actor="person", method="human", reviewed=True, text="model says", fact_keys=["model"]),
    )
    spec.assignments["model"] = "train"
    with pytest.raises(ValueError, match="independent memory evidence"):
        datasets.prepare(ledger, spec)


def test_incremental_readonly_import_and_failure_preserve_production(tmp_path):
    source = MemoryRepository(tmp_path / "production")
    source.observe(dict(id="a", session_id="session", source_id="camera", media_time=0, text="raw"))
    ledger = Ledger(tmp_path / "experiments")
    before = (source.root / "memory.sqlite3").read_bytes()
    assert collection.import_memory(ledger, source.root, 1)["imported"] == 1
    assert collection.import_memory(ledger, source.root, 1)["imported"] == 0
    assert (source.root / "memory.sqlite3").read_bytes() == before
    source.observe(dict(id="b", session_id="session", source_id="camera", media_time=1))
    assert collection.import_memory(ledger, source.root)["imported"] == 1
    assert len(ledger.list("raw")) == 2
    missing = tmp_path / "missing"
    with pytest.raises(sqlite3.OperationalError):
        collection.import_memory(ledger, missing)
    assert not missing.exists()
    assert ledger.list("failure")[0]["stage"] == "collection"


def test_import_checkpoint_and_record_are_atomic(tmp_path, monkeypatch):
    source = MemoryRepository(tmp_path / "production")
    source.observe(dict(id="a", session_id="session", source_id="camera", media_time=0))
    ledger = Ledger(tmp_path / "pipeline")
    append = ledger.append

    def fail(kind, *args, **kwargs):
        result = append(kind, *args, **kwargs)
        if kind == "raw":
            raise RuntimeError("simulated collection interruption")
        return result

    monkeypatch.setattr(ledger, "append", fail)
    with pytest.raises(RuntimeError, match="interruption"):
        collection.import_memory(ledger, source.root)
    assert not ledger.list("raw")
    assert not service.status(ledger)["cursors"]
    monkeypatch.setattr(ledger, "append", append)
    assert collection.import_memory(ledger, source.root)["imported"] == 1


def test_repeatability_across_clean_ledgers(prepared, tmp_path):
    ledger, spec, dataset = prepared
    other = Ledger(tmp_path / "independent")
    fixture = json.loads(FIXTURE.read_text())
    for item in fixture["records"]:
        collection.collect(other, Collected.model_validate(item["raw"]))
        labeling.annotate(other, item["raw"]["id"], Annotation.model_validate(item["annotation"]))
    assert datasets.prepare(other, spec) == dataset


def test_experiments_cannot_mutate_online_memory(prepared, tmp_path):
    ledger, _, dataset = prepared
    production = MemoryRepository(tmp_path / "production")
    production.teach(kind="object", text="Lamp", label="Lamp", evidence={"frame": "f"})
    before = (production.root / "memory.sqlite3").read_bytes()
    approved(ledger, dataset)
    assert (production.root / "memory.sqlite3").read_bytes() == before
    assert production.entities("object")[0]["label"] == "Lamp"


def test_teaching_import_resolves_episode_and_observation_lineage(tmp_path):
    source = MemoryRepository(tmp_path / "production")
    with source.connect() as db:
        for table in ("world_observations", "world_teachings", "world_questions"):
            db.execute(f"CREATE TABLE {table}(id TEXT PRIMARY KEY,payload TEXT)")
        observation = dict(id="o", episode_id="episode", source_id="recording", timestamp=1)
        question = dict(id="q", episode_id="episode", observation_ids=["o"])
        teaching = dict(id="answer", question_id="q", text="Desk", evidence=["o"])
        for table, item in (
            ("world_observations", observation),
            ("world_questions", question),
            ("world_teachings", teaching),
        ):
            db.execute(f"INSERT INTO {table} VALUES (?,?)", (item["id"], json.dumps(item)))
    ledger = Ledger(tmp_path / "pipeline")
    assert collection.import_memory(ledger, source.root)["imported"] == 2
    records = {r["source_kind"]: r for r in ledger.list("raw")}
    overlap = set(records["world_observations"]["group_ids"]) & set(records["world_teachings"]["group_ids"])
    assert any(g.endswith(":group:episode") for g in overlap)
    assert any(g.endswith(":source:recording") for g in overlap)
    assert any(g.endswith(":sample:o") for g in overlap)
    assert records["world_teachings"]["payload"] == teaching
