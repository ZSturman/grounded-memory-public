"""Paired console transport; domain stages also run independently from the CLI."""

from typing import Literal

from fastapi import APIRouter, Query
from pydantic import Field

from ..world.contracts import Contract
from . import collection, datasets, deployment, evaluation, experiments, labeling, service, workspaces
from .contracts import (
    Annotation,
    Collected,
    DatasetSpec,
    EvaluationInput,
    ExperimentInput,
    PromotionInput,
    RecallInput,
)


class ReviewInput(Contract):
    record_id: str
    annotation: Annotation


class RecordInput(Contract):
    record_id: str


class PolicyInput(RecordInput):
    allowed: bool
    actor: str = Field(min_length=1)


class RollbackInput(Contract):
    deployment_id: str
    actor: str = Field(min_length=1)


class SourceInput(Contract):
    sources: list[str]


class SelectionInput(Contract):
    record_ids: list[str] = Field(min_length=1, max_length=500)


class TriageInput(SelectionInput):
    action: Literal["skip", "flag", "reopen", "clear"]
    actor: str = Field(min_length=1)
    notes: str = ""


def router(ledger, source_root):
    api = APIRouter(prefix="/api/v2/pipeline", tags=["pipeline"])

    @api.get("/status")
    def status():
        return service.status(ledger)

    @api.get("/items/{kind}")
    def items(kind: str, limit: int = Query(default=25, ge=1, le=100), offset: int = Query(default=0, ge=0)):
        return service.inventory(ledger, kind, limit, offset)

    @api.get("/artifact")
    def artifact(id: str):
        return ledger.get(id)

    @api.get("/record")
    def record(id: str):
        return dict(raw=ledger.get(id, "raw"), allowed=ledger.allowed(id))

    @api.post("/collect")
    def collect(body: Collected):
        return collection.collect(ledger, body)

    @api.post("/import")
    def import_sources(limit: int = Query(default=500, ge=1, le=2000)):
        return collection.import_memory(ledger, source_root, limit, workspaces.source_settings(ledger))

    @api.get("/collection")
    def collection_state():
        return workspaces.collection_status(ledger, source_root)

    @api.post("/collection/settings")
    def collection_settings(body: SourceInput):
        return workspaces.configure_sources(ledger, body.sources)

    @api.get("/records")
    def records(
        state: str = "pending",
        source: str = "",
        search: str = "",
        order: str = "priority",
        limit: int = Query(default=30, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ):
        return workspaces.records(ledger, state, source, search, order, limit, offset)

    @api.post("/triage")
    def triage(body: TriageInput):
        return workspaces.triage(ledger, **body.model_dump())

    @api.post("/dataset-plan")
    def plan(body: SelectionInput):
        return workspaces.dataset_plan(ledger, body.record_ids)

    @api.get("/readiness")
    def readiness(experiment_id: str):
        return workspaces.readiness(ledger, experiment_id)

    @api.get("/health")
    def health():
        return workspaces.health(ledger)

    @api.post("/suggest")
    def suggest(body: RecordInput):
        return labeling.suggest(ledger, body.record_id)

    @api.post("/annotations")
    def annotate(body: ReviewInput):
        return labeling.annotate(ledger, body.record_id, body.annotation)

    @api.post("/policy")
    def policy(body: PolicyInput):
        labeling.set_policy(ledger, body.record_id, body.allowed, body.actor)
        return dict(saved=True)

    @api.post("/datasets")
    def prepare(body: DatasetSpec):
        return datasets.prepare(ledger, body)

    @api.post("/experiments")
    def experiment(body: ExperimentInput):
        return experiments.create(ledger, body.dataset_id, body.config)

    @api.post("/evaluations")
    def evaluate(body: EvaluationInput):
        return evaluation.evaluate(ledger, body.experiment_id, body.split)

    @api.post("/promote")
    def promote(body: PromotionInput):
        return deployment.promote(ledger, body.experiment_id, body.evaluation_ids, body.actor)

    @api.post("/rollback")
    def rollback(body: RollbackInput):
        return deployment.rollback(ledger, body.deployment_id, body.actor)

    @api.post("/recall")
    def recall(body: RecallInput):
        return deployment.recall(ledger, **body.model_dump())

    return api
