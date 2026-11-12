"""Versioned public contracts. Similarity is never presented as calibrated probability."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, json_schema_serialization_defaults_required=True
    )


class ObservationEnvelope(Contract):
    id: str = Field(min_length=1, max_length=200)
    episode_id: str
    source_id: str = Field(min_length=1, max_length=200)
    device_id: str = "unknown-device"
    sensor_id: str
    clock_id: str
    tracking_frame_id: str | None = None
    timestamp: float = Field(ge=0)
    duration: float = Field(default=0, ge=0, le=60)
    modality: Literal["vision", "audio", "pose", "depth", "event"]
    quality: float = Field(default=1, ge=0, le=1)
    evidence_refs: list[str] = Field(default_factory=list, max_length=32)
    features: list[float] = Field(default_factory=list, max_length=4096)
    encoder: str | None = None
    measurements: dict[str, Any] = Field(default_factory=dict)
    # An adapter may identify playback; it cannot assert human or independent teacher evidence.
    origin: Literal["sensor", "system_playback", "model_hypothesis"] = "sensor"


class PlaceCandidate(Contract):
    id: str
    label: str
    similarity: float
    landmark_support: int = 0
    transition_support: int = 0
    # The stored view that matched best: a teaching, or a view learned while the place was held.
    example_observation_id: str | None = None
    example_source: str = "teaching"


class RegionEvidence(BaseModel):
    model_config = ConfigDict(extra="allow", json_schema_serialization_defaults_required=True)
    id: str
    asset: str
    bbox: list[float]
    salience: float = 0


class Proposition(BaseModel):
    model_config = ConfigDict(extra="allow", json_schema_serialization_defaults_required=True)
    place_id: str | None = None
    entity_id: str | None = None
    event_key: str | None = None
    belief_id: str | None = None
    label: str | None = None
    candidates: list[PlaceCandidate] = Field(default_factory=list)


class LocationEstimate(Contract):
    place_id: str | None = None
    label: str = "Unknown place"
    certainty: Literal["unknown", "tentative", "recognized", "taught"] = "unknown"
    candidates: list[PlaceCandidate] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    previous_place_id: str | None = None
    # Held by an unbroken view sequence since the last teaching or recognition, not by appearance.
    continuous: bool = False


class Question(Contract):
    id: str
    episode_id: str
    version: int = 1
    kind: Literal["locate", "identify", "explain", "clarify", "predict", "intervene"]
    text: str
    reason: str
    evidence_refs: list[str] = Field(default_factory=list)
    observation_ids: list[str] = Field(default_factory=list)
    proposition: Proposition = Field(default_factory=Proposition)
    region: RegionEvidence | None = None
    status: Literal["active", "open", "dismissed", "answered"] = "open"
    created_at: float
    shown_at: float | None = None
    touched_at: float | None = None
    priority: float = 0
    fingerprint: str


class TeachingReply(Contract):
    id: str = Field(min_length=1, max_length=200)
    question_version: int = 1
    text: str = Field(default="", max_length=5000)
    response: Literal["answer", "yes", "no", "unknown", "later", "dismiss"] = "answer"
    teacher_id: str = Field(default="local-human", min_length=1, max_length=200)
    entity_id: str | None = None
    label: str | None = Field(default=None, max_length=200)
    # These are teacher-selected event names, not guesses extracted from model world knowledge.
    cause_event: str | None = Field(default=None, max_length=200)
    expected_event: str | None = Field(default=None, max_length=200)
    horizon_seconds: float = Field(default=30, ge=1, le=3600)
    audio_ref: str | None = None

    @model_validator(mode="after")
    def has_answer(self):
        if self.response == "answer" and not self.text.strip():
            raise ValueError("An answer needs your words.")
        if self.label and self.label.casefold() not in self.text.casefold():
            raise ValueError("A name must be present in the teacher's answer.")
        return self


class BeliefRevision(Contract):
    id: str = Field(min_length=1, max_length=200)
    operation: Literal["correct", "contradict", "retract", "restore", "identity"]
    reason: str = Field(min_length=1, max_length=5000)
    teacher_id: str = "local-human"
    value: str | None = Field(default=None, max_length=5000)
    target_entity_id: str | None = None
    expected_version: int = Field(ge=1)


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    kind: str
    id: str | None = None


class BeliefRecord(BaseModel):
    model_config = ConfigDict(extra="allow", json_schema_serialization_defaults_required=True)
    id: str
    subject: str
    subject_label: str
    predicate: str
    value: str
    source: SourceRecord
    context_label: str | None = None
    status: str
    certainty: str
    version: int
    supporting: list[str]
    conflicting: list[str]
    history: list[str]


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    label: str
    start: float
    due: float
    status: str


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    label: str
    kind: str
    surprising: bool


class RevisionState(Contract):
    value: str
    status: str
    certainty: str
    version: int


class RevisionRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    belief_id: str
    reason: str
    before: RevisionState
    after: RevisionState


class SensorState(Contract):
    received_at: float = 0
    modality: str
    timestamp: float
    clock_id: str
    quality: float


class WorkingItem(BaseModel):
    model_config = ConfigDict(extra="allow", json_schema_serialization_defaults_required=True)
    kind: Literal["place", "place_candidate", "question", "expectation", "event", "object"]
    id: str
    label: str
    basis: str
    expires: float | None = None


class Executive(Contract):
    mode: Literal["EXPLORE", "RELOCALIZE", "VERIFY", "CONFIRMED", "NOVEL", "RECOVER", "STOPPED"] = "EXPLORE"
    reason: str = ""
    # What the body should do next; a person holding the phone today, gaze or wheels later.
    request: str | None = None
    working_set: list[WorkingItem] = Field(default_factory=list, max_length=12)


class WorldState(Contract):
    learner_id: str
    episode_id: str | None = None
    location: LocationEstimate = Field(default_factory=LocationEstimate)
    executive: Executive = Field(default_factory=Executive)
    attention: Question | None = None
    questions: list[Question] = Field(default_factory=list)
    beliefs: list[BeliefRecord] = Field(default_factory=list)
    predictions: list[PredictionRecord] = Field(default_factory=list)
    recent_events: list[EventRecord] = Field(default_factory=list)
    revisions: list[RevisionRecord] = Field(default_factory=list)
    sensors: dict[str, SensorState] = Field(default_factory=dict)
    status: str = "observing"
    limitations: list[str] = Field(default_factory=list)
