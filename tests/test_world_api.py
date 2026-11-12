import importlib
import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GM_PAIR_CODE", "123456")
    from grounded_memory import api

    importlib.reload(api)
    with TestClient(api.app) as client:
        client.post("/api/pair", json={"code": "123456"})
        yield client


def test_v2_first_awake_answer_and_contract(client):
    episode = client.post("/api/v2/episodes", json={}).json()["id"]
    observation = dict(
        id="first",
        episode_id=episode,
        source_id="camera",
        sensor_id="cam",
        clock_id="boot",
        timestamp=0,
        modality="vision",
        features=[1, 0],
        encoder="test",
    )
    assert client.post(f"/api/v2/episodes/{episode}/observations", json=observation).json()["accepted"]
    state = client.get(f"/api/v2/world?episode_id={episode}").json()
    assert state["location"]["certainty"] == "unknown"
    q = state["attention"]
    assert q["kind"] == "locate"
    response = client.post(f"/api/v2/questions/{q['id']}/answers", json=dict(id="reply", text="Office"))
    assert response.status_code == 200
    assert client.get(f"/api/v2/world?episode_id={episode}").json()["location"]["label"] == "Office"
    assert (
        client.post(f"/api/v2/episodes/{episode}/observations", json=observation).json()["accepted"] is False
    )
    observation["features"] = [0, 1]
    assert client.post(f"/api/v2/episodes/{episode}/observations", json=observation).status_code == 409


def test_live_audio_evidence_and_system_speech_exclusion(client):
    episode = client.post("/api/v2/episodes", json={}).json()["id"]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes((10000 * np.sin(np.arange(32000) * 2 * np.pi * 440 / 16000)).astype("<i2").tobytes())
    params = dict(observation_id="sound", source_id="mic", clock_id="boot", timestamp=0)
    response = client.post(f"/api/v2/episodes/{episode}/audio", params=params, content=buffer.getvalue())
    assert response.status_code == 200
    state = client.get(f"/api/v2/world?episode_id={episode}").json()
    assert state["attention"]["kind"] == "identify"
    assert state["attention"]["region"] is None
    assert client.get("/api/assets/" + state["attention"]["evidence_refs"][0]).content == buffer.getvalue()
    assert (
        client.post(f"/api/v2/episodes/{episode}/audio", params=params, content=buffer.getvalue()).json()[
            "accepted"
        ]
        is False
    )
    params.update(observation_id="playback", timestamp=10, playback=True)
    client.post(f"/api/v2/episodes/{episode}/audio", params=params, content=buffer.getvalue())
    assert len(client.get(f"/api/v2/world?episode_id={episode}").json()["recent_events"]) == 1


def test_native_v2_pose_without_depth_and_retry(client, monkeypatch):
    import json

    from grounded_memory import api

    episode = client.post("/api/v2/episodes", json={}).json()["id"]

    async def inference(*args):
        return dict(regions=[], scene_features=[1, 0], encoder="test", sharpness=150)

    monkeypatch.setattr(api, "inference", inference)
    pose = [float(i % 5 == 0) for i in range(16)]
    meta = dict(
        observation_id="frame",
        recording="phone.mp4",
        clock_id="boot",
        tracking_frame_id="map1",
        media_time=0,
        sensor=dict(modality="video", source="test"),
        camera=dict(transform=pose),
    )

    def send(image):
        return client.post(
            f"/api/v2/episodes/{episode}/samples",
            data={"meta": json.dumps(meta)},
            files={"image": ("frame.jpg", image, "image/jpeg")},
        )

    assert send(b"fake-image").status_code == 200
    assert api.world.observation("frame")["measurements"]["pose"] == pose
    assert send(b"fake-image").status_code == 200
    assert send(b"changed-image").status_code == 409


def test_auth_and_missing_evidence(client):
    episode = client.post("/api/v2/episodes", json={}).json()["id"]
    invalid = dict(
        id="one",
        episode_id=episode,
        source_id="source",
        sensor_id="sensor",
        clock_id="clock",
        timestamp=0,
        modality="event",
        evidence_refs=["missing.wav"],
    )
    assert client.post(f"/api/v2/episodes/{episode}/observations", json=invalid).status_code == 422
    client.cookies.clear()
    assert client.get("/api/v2/world").status_code == 401


def test_direct_attention_works_in_unknown_place_and_refinement_binds_reply_version(client, monkeypatch):
    from grounded_memory import api

    async def inference(*args):
        return dict(regions=[], scene_features=[1.0], encoder="test", sharpness=150)

    async def worker(function, *args):
        (api.WORK / "tapped.png").write_bytes(b"crop")
        return dict(
            id="tapped",
            bbox=[0.1, 0.1, 0.5, 0.5],
            asset="tapped.png",
            salience=1,
            features=[1.0],
            encoder="test",
        )

    monkeypatch.setattr(api, "inference", inference)
    monkeypatch.setattr(api, "worker", worker)
    episode = client.post("/api/v2/episodes", json={}).json()["id"]
    client.post(
        f"/api/sessions/{episode}/frames?media_time=0&segment=original.mov", content=b"jpg"
    ).raise_for_status()
    tapped = client.post(f"/api/sessions/{episode}/point", json={"x": 0.5, "y": 0.5})
    assert tapped.status_code == 200
    question = tapped.json()["world"]["attention"]
    assert question["kind"] == "identify" and question["region"]["asset"] == "tapped.png"
    assert tapped.json()["world"]["location"]["place_id"] is None
    api.world.refine_question(
        dict(id=question["id"], region=dict(question["region"], features=[1.0], encoder="test"))
    )
    body = dict(id="tap-answer", question_version=1, text="Lamp")
    assert client.post(f"/api/v2/questions/{question['id']}/answers", json=body).status_code == 409
    body["question_version"] = 2
    response = client.post(f"/api/v2/questions/{question['id']}/answers", json=body)
    assert response.status_code == 200
    assert api.store.entities("object")[0]["examples"][0]["asset"] == "tapped.png"


def test_pipeline_paired_complete_workflow_and_existing_world(client):
    import json
    from pathlib import Path

    fixture = json.loads((Path(__file__).parent / "fixtures/pipeline/regression-v1.json").read_text())
    prefix = "/api/v2/pipeline"
    for item in fixture["records"]:
        client.post(prefix + "/collect", json=item["raw"]).raise_for_status()
        client.post(
            prefix + "/annotations", json=dict(record_id=item["raw"]["id"], annotation=item["annotation"])
        ).raise_for_status()
    response = client.post(prefix + "/datasets", json=fixture["spec"])
    response.raise_for_status()
    dataset = response.json()
    candidate = client.post(
        prefix + "/experiments",
        json=dict(dataset_id=dataset["id"], config=dict(algorithm="coverage", min_score=0.5)),
    ).json()
    evaluations = []
    for split in ("test", "regression"):
        result = client.post(prefix + "/evaluations", json=dict(experiment_id=candidate["id"], split=split))
        result.raise_for_status()
        evaluations.append(result.json()["id"])
    promoted = client.post(
        prefix + "/promote",
        json=dict(experiment_id=candidate["id"], evaluation_ids=evaluations, actor="paired-reviewer"),
    )
    promoted.raise_for_status()
    recall = client.post(prefix + "/recall", json=dict(query="Garage key")).json()
    assert recall["hits"][0]["evidence_ids"] == ["test-current"]
    assert client.get(prefix + "/status").json()["awaiting_review"] == 0
    assert len(client.get(prefix + "/items/raw?limit=2").json()) == 2
    assert client.get(prefix + "/artifact", params={"id": "test-current"}).json()["payload"]["text"]
    client.post(
        prefix + "/rollback", json=dict(deployment_id=promoted.json()["id"], actor="reviewer")
    ).raise_for_status()
    # Existing transport and world ownership remain usable after the pipeline runs.
    episode = client.post("/api/v2/episodes", json={}).json()["id"]
    assert client.get(f"/api/v2/world?episode_id={episode}").status_code == 200
    assert client.get("/api/memory").status_code == 200
    client.cookies.clear()
    for path in ("/status", "/items/raw", "/artifact?id=test-current"):
        assert client.get(prefix + path).status_code == 401
    assert client.post(prefix + "/import").status_code == 401
    assert client.post(prefix + "/recall", json=dict(query="Garage key")).status_code == 401


def test_pipeline_invalid_label_and_failed_stage_are_visible(client):
    prefix = "/api/v2/pipeline"
    assert (
        client.post(
            prefix + "/annotations",
            json=dict(record_id="missing", annotation=dict(actor="model", method="model", reviewed=True)),
        ).status_code
        == 422
    )
    assert (
        client.post(
            prefix + "/annotations",
            json=dict(record_id="missing", annotation=dict(actor="person", method="human")),
        ).status_code
        == 404
    )
    status = client.get(prefix + "/status").json()
    assert status["recent_failures"][0]["stage"] == "labeling"
    assert status["deployed"] is None
