"""Version two transport. Domain state is owned by WorldEngine, not request handlers."""

import hashlib
import io
import wave
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from .contracts import BeliefRevision, Contract, ObservationEnvelope, TeachingReply, WorldState
from .engine import Conflict
from .sensing import audio_features


class EpisodeInput(Contract):
    mode: Literal["live", "recall"] = "live"


class EpisodeControl(Contract):
    action: Literal["pause", "resume", "stop"]


class Playback(Contract):
    clock_id: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    source_id: str | None = None


def router(engine, memory, dialogue):
    api = APIRouter(prefix="/api/v2")

    @api.post("/episodes")
    def create(body: EpisodeInput):
        episode = engine.create_episode(mode=body.mode)
        memory.save_session(
            dict(
                id=episode["id"],
                mode=body.mode,
                status="ready",
                created_at=episode["created_at"],
                context=None,
                question=None,
                latest=None,
                recordings=[],
                paused=False,
                world_version=2,
            )
        )
        return episode

    @api.get("/episodes")
    def episodes():
        from .repository import rows

        with engine.repository.transaction() as db:
            return rows(db, "episodes")

    @api.get("/world", response_model=WorldState)
    def world(episode_id: str | None = None):
        return engine.state(episode_id)

    @api.get("/episodes/{episode_id}/state", response_model=WorldState)
    def state(episode_id: str):
        return engine.state(episode_id)

    @api.post("/episodes/{episode_id}/observations")
    def observe(episode_id: str, body: ObservationEnvelope):
        if body.episode_id != episode_id:
            raise ValueError("Episode does not match the observation")
        for ref in body.evidence_refs:
            if not ref or "/" in ref or "\\" in ref or not (memory.root / "assets" / ref).is_file():
                raise ValueError("Evidence must reference a retained local asset")
        try:
            return dict(accepted=engine.observe(body))
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc

    class InterpretationInput(Contract):
        text: str = Field(min_length=1, max_length=5000)

    @api.post("/questions/{question_id}/interpret")
    async def interpret(question_id: str, body: InterpretationInput):
        from .repository import get

        with engine.repository.transaction() as db:
            question = get(db, "questions", question_id)
        if not question:
            raise KeyError(question_id)
        try:
            return await dialogue.interpret_world(body.text, question["text"])
        except Exception as exc:
            raise HTTPException(
                503, "Local interpretation unavailable; your original answer can still be saved"
            ) from exc

    @api.post("/questions/{question_id}/answers")
    def answer(question_id: str, body: TeachingReply):
        if body.audio_ref and (
            "/" in body.audio_ref or not (memory.root / "assets" / body.audio_ref).is_file()
        ):
            raise ValueError("Voice evidence is missing")
        try:
            result = engine.answer(question_id, body)
            if not result.get("needs_clarification"):
                for session in memory.sessions():
                    if (session.get("question") or {}).get("id") == question_id:
                        session["question"] = None
                        memory.save_session(session)
            return result
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @api.post("/questions/{question_id}/touch")
    def touch(question_id: str):
        engine.touch(question_id)
        return {"saved": True}

    @api.post("/beliefs/{belief_id}/revisions")
    def revise(belief_id: str, body: BeliefRevision):
        try:
            return engine.revise(belief_id, body)
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @api.post("/episodes/{episode_id}/control")
    def control(episode_id: str, body: EpisodeControl):
        engine.control(episode_id, body.action)
        session = memory.session(episode_id)
        session["paused"] = body.action == "pause"
        if body.action == "stop":
            session["status"] = "stopped"
        memory.save_session(session)
        return engine.state(episode_id)

    class SystemOutput(Contract):
        duration_seconds: float = Field(ge=0, le=300)

    @api.post("/output")
    def output(body: SystemOutput):
        engine.system_output(body.duration_seconds)
        return {"saved": True}

    @api.post("/episodes/{episode_id}/playback")
    def playback(episode_id: str, body: Playback):
        engine.playback(episode_id, body.clock_id, body.start, body.end, body.source_id)
        return {"saved": True}

    @api.post("/episodes/{episode_id}/audio")
    async def audio(
        episode_id: str,
        request: Request,
        observation_id: str,
        source_id: str,
        clock_id: str,
        timestamp: float,
        device_id: str = "phone",
        playback: bool = False,
    ):
        data = await request.body()
        if len(data) > 2_000_000:
            raise ValueError("Audio chunk too large")
        try:
            with wave.open(io.BytesIO(data)) as wav:
                if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
                    raise ValueError("Send mono 16-bit PCM WAV")
                rate = wav.getframerate()
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(float) / 32768
        except (wave.Error, EOFError) as exc:
            raise ValueError("Invalid PCM WAV") from exc
        features, measurements, encoder = audio_features(samples, rate)
        # Content-addressing makes retries reuse the exact original evidence.
        asset = hashlib.sha256(data).hexdigest() + ".wav"
        (memory.root / "assets" / asset).write_bytes(data)
        sensor_id = memory.register_sensor(
            dict(modality="audio", source="native-pcm", device=device_id, sample_rate=rate)
        )
        envelope = ObservationEnvelope(
            id=observation_id,
            episode_id=episode_id,
            source_id=source_id,
            device_id=device_id,
            sensor_id=sensor_id,
            clock_id=clock_id,
            timestamp=timestamp,
            duration=measurements["duration"],
            modality="audio",
            evidence_refs=[asset],
            features=features,
            encoder=encoder,
            measurements=measurements,
            origin="system_playback" if playback else "sensor",
        )
        try:
            return dict(accepted=engine.observe(envelope))
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc

    return api
