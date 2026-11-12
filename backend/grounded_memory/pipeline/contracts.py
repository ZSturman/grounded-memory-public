"""Inputs shared by the CLI and paired review console."""

from typing import Any, Literal

from pydantic import Field, model_validator

from ..world.contracts import Contract

Split = Literal["train", "validation", "test", "regression"]
Category = Literal[
    "accuracy", "retrieval", "temporal", "contradiction", "consolidation", "grounding", "privacy"
]


class Collected(Contract):
    id: str = Field(min_length=1, max_length=300)
    source_id: str = Field(min_length=1)
    source_kind: str = Field(min_length=1)
    source_timestamp: float
    group_ids: list[str] = Field(min_length=1)
    origin: Literal["human", "sensor", "model", "derived"]
    payload: dict[str, Any]


class Annotation(Contract):
    actor: str = Field(min_length=1)
    method: Literal["human", "deterministic", "model"]
    reviewed: bool = False
    text: str = ""
    fact_keys: list[str] = Field(default_factory=list)
    status: Literal["current", "historical", "contested", "excluded"] = "current"
    valid_from: float | None = None
    valid_until: float | None = None
    notes: str = ""
    # Optimistic concurrency prevents a stale review silently replacing a correction.
    previous_revision: str | None = None

    @model_validator(mode="after")
    def valid(self):
        if self.reviewed and self.method != "human":
            raise ValueError("Only an explicit human review may confirm an annotation")
        if self.reviewed and self.status != "excluded" and (not self.text.strip() or not self.fact_keys):
            raise ValueError("Reviewed memory needs text and underlying fact keys")
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_from >= self.valid_until
        ):
            raise ValueError("valid_until must follow valid_from")
        return self


class Case(Contract):
    id: str = Field(min_length=1)
    split: Split
    query: str = Field(min_length=1)
    expected_ids: list[str]
    categories: list[Category] = Field(min_length=1)
    expected_uncertain: bool = False
    expected_hit_count: int | None = Field(default=None, ge=0)
    as_of: float | None = None
    allowed_sources: list[str] | None = None
    include_history: bool = False
    # Include the conversation/document/fact that supplied an unanswerable query too.
    group_ids: list[str] = Field(min_length=1)


class DatasetSpec(Contract):
    name: str = Field(min_length=1)
    assignments: dict[str, Split] = Field(min_length=1)
    cases: list[Case] = Field(min_length=1)
    evaluation_time: float
    reviewer: str = Field(min_length=1)
    purpose: Literal["synthetic-regression", "reviewed-local"]


class Strategy(Contract):
    algorithm: Literal["overlap", "coverage"] = "overlap"
    top_k: int = Field(default=3, ge=1, le=20)
    min_score: float = Field(default=0.2, gt=0, le=1)


class ExperimentInput(Contract):
    dataset_id: str
    config: Strategy = Field(default_factory=Strategy)


class EvaluationInput(Contract):
    experiment_id: str
    split: Literal["validation", "test", "regression"] = "validation"


class PromotionInput(Contract):
    experiment_id: str
    evaluation_ids: list[str] = Field(min_length=2)
    actor: str = Field(min_length=1)


class RecallInput(Contract):
    query: str = Field(min_length=1)
    as_of: float | None = None
    allowed_sources: list[str] | None = None
    include_history: bool = False
