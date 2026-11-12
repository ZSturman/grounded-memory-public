import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GM_PAIR_CODE", "123456")
    from grounded_memory import api

    importlib.reload(api)
    with TestClient(api.app) as client:
        yield client


def test_authentication_rejects_missing_forged_and_wrong_sessions(client):
    assert client.get("/api/memory").status_code == 401
    client.cookies.set("gm_session", "1.forged")
    assert client.get("/api/memory").status_code == 401
    assert client.post("/api/pair", json={"code": "wrong"}).status_code == 401
    client.cookies.clear()
    assert client.post("/api/pair", json={"code": "123456"}).status_code == 200
    assert client.get("/api/memory").json() == []


def test_context_gate_resumable_upload_and_clarification(client):
    client.post("/api/pair", json={"code": "123456"})
    session = client.post("/api/sessions", json={"mode": "recall"}).json()
    path = "/api/sessions/" + session["id"]
    assert client.post(path + "/start", json={}).status_code == 422
    assert client.put(path + "/upload?offset=0", content=b"abc").json()["next_offset"] == 3
    assert client.put(path + "/upload?offset=0", content=b"xyz").status_code == 409
    assert client.put(path + "/upload?offset=3", content=b"def").json()["next_offset"] == 6
    assert client.post(path + "/context", json={"label": "Kitchen"}).json()["context"]["label"] == "Kitchen"
    other = client.post("/api/sessions", json={"mode": "recall"}).json()
    result = client.post("/api/sessions/" + other["id"] + "/context", json={"label": "Kitchen"}).json()
    assert result["needs_clarification"]
    assert len(client.get("/api/memory").json()) == 1


def test_teaching_requires_frozen_current_question(client):
    client.post("/api/pair", json={"code": "123456"})
    session = client.post("/api/sessions", json={"mode": "recall"}).json()
    response = client.post(
        "/api/sessions/" + session["id"] + "/teach",
        json={"question_id": "invented", "text": "clock", "label": "clock"},
    )
    assert response.status_code == 409


def test_media_and_missing_session_errors_preserve_original(client):
    client.post("/api/pair", json={"code": "123456"})
    assert client.get("/api/sessions/nonexistent").status_code == 404
    session = client.post("/api/sessions", json={"mode": "recall"}).json()
    path = "/api/sessions/" + session["id"]
    client.put(path + "/upload?offset=0", content=b"not a movie")
    assert client.post(path + "/upload/complete", json={}).status_code == 422
    from grounded_memory.api import ROOT

    assert (ROOT / "assets" / (session["id"] + ".upload")).read_bytes() == b"not a movie"


def test_restart_keeps_pairing_and_marks_interrupted_sessions(client):
    client.post("/api/pair", json={"code": "123456"})
    session = client.post("/api/sessions", json={"mode": "live"}).json()
    from grounded_memory import api

    api.store.save_session(dict(session, status="live"))
    importlib.reload(api)
    with TestClient(api.app) as restarted:
        restarted.cookies = client.cookies
        assert restarted.get("/api/sessions/" + session["id"]).json()["status"] == "interrupted"


def test_transcriptions_never_overlap(client, monkeypatch):
    import asyncio
    import io
    import sys
    import types

    from fastapi import UploadFile
    from grounded_memory import api

    active = []

    def transcribe(path, **options):
        active.append(1)
        assert len(active) == 1, "concurrent Whisper call"
        import time

        time.sleep(0.2)
        active.pop()
        return {"text": "clock"}

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=transcribe))

    async def both():
        return await asyncio.gather(
            *[api.transcribe(UploadFile(io.BytesIO(b"audio"), filename="reply")) for _ in range(2)]
        )

    assert [r["text"] for r in asyncio.run(both())] == ["clock", "clock"]


def test_only_question_and_periodic_frames_are_retained(client, monkeypatch):
    import asyncio

    from grounded_memory import api

    client.post("/api/pair", json={"code": "123456"})
    session = client.post("/api/sessions", json={"mode": "live"}).json()
    client.post("/api/sessions/" + session["id"] + "/context", json={"label": "Desk"})

    async def inference(frame, previous, geometry=None):
        crop = api.WORK / ("crop-" + api.Path(frame).stem + ".png")
        crop.write_bytes(b"crop")
        region = dict(
            id=crop.stem, bbox=[0, 0, 1, 1], asset=crop.name, salience=1, features=[1.0], encoder="t"
        )
        return dict(regions=[region], scene_features=[1.0], encoder="t", segmentation="t")

    monkeypatch.setattr(api, "inference", inference)

    async def stream():
        for i in range(12):
            path = api.WORK / f"f{i}.jpg"
            path.write_bytes(b"jpg")
            await api.accept_frame(session["id"], i * 0.5, path, None, "segment")

    asyncio.run(stream())
    kept = {p.name for p in (api.ROOT / "assets").iterdir()}
    question = client.get("/api/sessions/" + session["id"]).json()["question"]
    assert {question["frame"], question["region"]["asset"]} <= kept
    assert {"f0.jpg", "f10.jpg"} <= kept and "f3.jpg" not in kept and "crop-f3.png" not in kept
    saved = client.get("/api/sessions/" + session["id"] + "/observations").json()
    assert [o["evidence_ref"] for o in saved].count(None) == 10


def test_native_frames_guess_place_then_ask_without_feature_vectors(client, monkeypatch, tmp_path):
    from grounded_memory import api

    async def inference(frame, previous, geometry=None):
        crop = api.WORK / ("crop-" + api.Path(frame).stem + ".png")
        crop.write_bytes(b"crop")
        region = dict(
            id=crop.stem, bbox=[0, 0, 1, 1], asset=crop.name, salience=1, features=[1.0], encoder="t"
        )
        return dict(regions=[region], scene_features=[1.0, 0.0], encoder="t", segmentation="t")

    monkeypatch.setattr(api, "inference", inference)
    client.post("/api/pair", json={"code": "123456"})
    first = client.post("/api/sessions", json={"mode": "live"}).json()["id"]
    client.post(f"/api/sessions/{first}/frames?media_time=0&segment=a.mov", content=b"jpg")
    client.post(f"/api/sessions/{first}/context", json={"label": "Office"})

    second = client.post("/api/sessions", json={"mode": "live"}).json()["id"]
    state = client.post(f"/api/sessions/{second}/frames?media_time=0&segment=b.mov", content=b"jpg").json()
    assert state["status"] == "live" and state["question"] is None
    assert state["place_guess"]["label"] == "Office" and state["context"] is None
    guess = state["place_guess"]
    confirmed = client.post(
        f"/api/sessions/{second}/context",
        json={"label": "Office", "resolution": "same", "previous_id": guess["id"]},
    ).json()
    assert confirmed["status"] == "live" and confirmed["context"]["place_id"] == guess["id"]
    state = client.post(f"/api/sessions/{second}/frames?media_time=0.5&segment=b.mov", content=b"jpg").json()
    assert state["question"] and "features" not in state["question"]["region"]
    assert all("features" not in r for r in state["latest"]["regions"])
    assert (
        client.post(f"/api/sessions/{second}/frames?media_time=1&segment=../x", content=b"jpg").status_code
        == 422
    )


def test_certificate_authority_is_public_but_only_when_present(client, monkeypatch, tmp_path):
    monkeypatch.setenv("GM_CA_CERT", str(tmp_path / "missing.crt"))
    assert client.get("/api/ca").status_code == 404
    (tmp_path / "root.crt").write_text("-----BEGIN CERTIFICATE-----\n")
    monkeypatch.setenv("GM_CA_CERT", str(tmp_path / "root.crt"))
    assert client.get("/api/ca").text.startswith("-----BEGIN CERTIFICATE")


def test_tapping_an_object_makes_it_the_question(client, monkeypatch):
    from grounded_memory import api

    async def inference(frame, previous, geometry=None):
        return dict(regions=[], scene_features=[1.0], encoder="t", segmentation="t")

    async def worker(function, frame, point, folder, geometry=None):
        assert function is api.segment_point and point == (0.45, 0.62)
        (api.WORK / "tapped.png").write_bytes(b"crop")
        return dict(
            id="tapped",
            bbox=[0.3, 0.5, 0.4, 0.3],
            asset="tapped.png",
            salience=0.8,
            features=[1.0],
            encoder="t",
        )

    monkeypatch.setattr(api, "inference", inference)
    monkeypatch.setattr(api, "worker", worker)
    client.post("/api/pair", json={"code": "123456"})
    sid = client.post("/api/sessions", json={"mode": "live"}).json()["id"]
    assert client.post(f"/api/sessions/{sid}/point", json={"x": 0.45, "y": 0.62}).status_code == 422
    client.post(f"/api/sessions/{sid}/frames?media_time=0&segment=a.mov", content=b"jpg")
    client.post(f"/api/sessions/{sid}/context", json={"label": "Desk"})
    state = client.post(f"/api/sessions/{sid}/point", json={"x": 0.45, "y": 0.62}).json()
    assert state["question"]["region"]["id"] == "tapped" and "features" not in state["question"]["region"]
    assert (api.ROOT / "assets" / "tapped.png").exists()
    assert client.post(f"/api/sessions/{sid}/point", json={"x": 2, "y": 0}).status_code == 422


PHONE = {
    "modality": "video",
    "source": "arkit",
    "projection": "rectilinear",
    "device": "iPhone17,2",
    "lens": "wide",
}


def sample(client, sid, t, depth=True):
    import json

    import numpy as np

    meta = {"media_time": t, "recording": "r.mov", "sensor": PHONE}
    files = {"image": ("f.jpg", b"jpg", "image/jpeg")}
    if depth:
        meta.update(
            camera={
                "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                "intrinsics": [1000, 0, 0, 0, 1000, 0, 960, 720, 1],
                "resolution": [1920, 1440],
            },
            depth_size=[4, 3],
        )
        files["depth"] = ("d", np.full(12, 2000, "<u2").tobytes(), "application/octet-stream")
        files["confidence"] = ("c", np.full(12, 2, np.uint8).tobytes(), "application/octet-stream")
    return client.post(f"/api/sessions/{sid}/samples", files=files, data={"meta": json.dumps(meta)})


def test_samples_record_sensor_locate_objects_and_count_relations(client, monkeypatch):
    from grounded_memory import api

    seen = []

    async def inference(frame, previous, geometry=None):
        seen.append(geometry)
        regions = [
            dict(
                id="b",
                bbox=[0.4, 0.4, 0.1, 0.2],
                asset="b.png",
                salience=1,
                features=[1.0, 0.0],
                encoder="t",
                extent={"min": [0.4, 0.75, 0.2], "max": [0.46, 0.87, 0.26]},
            ),
            dict(
                id="d",
                bbox=[0, 0.6, 1, 0.3],
                asset="d.png",
                salience=1,
                features=[0.0, 1.0],
                encoder="t",
                extent={"min": [0, 0.70, 0], "max": [1.2, 0.75, 0.6]},
            ),
        ]
        return dict(regions=regions, scene_features=[1.0], encoder="t", segmentation="t")

    monkeypatch.setattr(api, "inference", inference)
    client.post("/api/pair", json={"code": "123456"})
    for label, vector in (("bottle", [1.0, 0.0]), ("desk", [0.0, 1.0])):
        api.store.teach(kind="object", text=label, label=label, evidence={"features": vector, "encoder": "t"})
    sid = client.post("/api/sessions", json={"mode": "live"}).json()["id"]
    client.post(f"/api/sessions/{sid}/context", json={"label": "Office"})
    state = sample(client, sid, 0.0).json()
    assert state["relations"] == [
        {"id": state["relations"][0]["id"], "subject": "bottle", "predicate": "on", "object": "desk"}
    ]
    assert seen[0]["camera"]["resolution"] == [1920, 1440] and seen[0]["depth"].endswith(".depth.npz")
    assert all(r["projection"] == "rectilinear" and r["sensor_id"] for r in state["latest"]["regions"])
    sample(client, sid, 0.5)
    sample(client, sid, 1.0, depth=False)
    assert seen[-1] is None
    observed = client.get(f"/api/sessions/{sid}/relations").json()
    assert observed[0]["support"] == 3 and len(observed[0]["sensors"]) == 1
    sensor = api.store.sensors()[observed[0]["sensors"][0]]
    assert sensor["lens"] == "wide" and sensor["projection"] == "rectilinear"
    saved = client.get(f"/api/sessions/{sid}/observations").json()
    assert (
        saved[0]["streams"] == {"video": True, "pose": True, "depth": True}
        and not saved[-1]["streams"]["depth"]
    )
    assert not any(c["predicate"] == "on" for e in api.store.entities() for c in e["claims"])
    client.post(f"/api/relations/{observed[0]['id']}/confirm")
    bottle = next(e for e in api.store.entities() if e["label"] == "bottle")
    assert any(c["predicate"] == "on" for c in bottle["claims"])
    bad = {**PHONE, "projection": "pinhole-ish"}
    import json

    reply = client.post(
        f"/api/sessions/{sid}/samples",
        files={"image": ("f.jpg", b"jpg")},
        data={"meta": json.dumps({"media_time": 2, "recording": "r.mov", "sensor": bad})},
    )
    assert reply.status_code == 422


def test_outline_refinement_uses_person_points_searches_sizes_and_resets(client, monkeypatch):
    from grounded_memory import api

    async def inference(frame, previous, geometry=None):
        (api.WORK / "auto.png").write_bytes(b"crop")
        region = dict(
            id="auto", bbox=[0.4, 0.4, 0.2, 0.2], asset="auto.png", salience=1, features=[1.0], encoder="t"
        )
        return dict(regions=[region], scene_features=[1.0], encoder="t", segmentation="t")

    calls = []

    async def worker(function, frame, *args):
        calls.append((function, args))
        name = f"refined{len(calls)}.png"
        (api.WORK / name).write_bytes(b"crop")
        prompt = args[0] if function is api.segment_prompt else {"points": [], "box": None}
        return dict(
            id=name,
            bbox=[0.1, 0.1, 0.8, 0.8],
            asset=name,
            salience=1,
            features=[1.0],
            encoder="t",
            prompt=prompt,
        )

    monkeypatch.setattr(api, "inference", inference)
    monkeypatch.setattr(api, "worker", worker)
    client.post("/api/pair", json={"code": "123456"})
    sid = client.post("/api/sessions", json={"mode": "live"}).json()["id"]
    client.post(f"/api/sessions/{sid}/context", json={"label": "Office"})
    question = sample(client, sid, 0.0, depth=False).json()["question"]
    path = f"/api/sessions/{sid}/question/refine"
    assert client.post(path, json={"question_id": "old", "points": [[0.1, 0.1, True]]}).status_code == 409
    tapped = client.post(
        path, json={"question_id": question["id"], "points": [[0.1, 0.1, True], [0.5, 0.9, False]]}
    ).json()
    function, (prompt, *_) = calls[0]
    assert function is api.segment_prompt and prompt["box"] is None
    assert prompt["points"] == [[0.1, 0.1, True], [0.5, 0.9, False]]
    assert tapped["question"]["id"] == question["id"] and tapped["question"]["region"]["id"] == "refined1.png"
    client.post(path, json={"question_id": question["id"], "granularity": "smaller"})
    function, (current, direction, *_) = calls[1]
    assert function is api.resize_region and direction == "smaller" and current["id"] == "refined1.png"
    reset = client.post(path, json={"question_id": question["id"], "reset": True}).json()
    assert reset["question"]["region"]["id"] == "auto" and len(calls) == 2
    taught = client.post(
        f"/api/sessions/{sid}/teach", json={"question_id": question["id"], "text": "a mug", "label": "mug"}
    ).json()
    assert taught["needs_clarification"] is False


def test_put_aside_questions_are_kept_not_asked_again_and_answerable_later(client, monkeypatch):
    from grounded_memory import api

    vector = [[1.0, 0.0]]

    async def inference(frame, previous, geometry=None):
        name = api.Path(frame).stem + ".png"
        (api.WORK / name).write_bytes(b"crop")
        region = dict(
            id=name, bbox=[0.4, 0.4, 0.2, 0.2], asset=name, salience=1, features=vector[0], encoder="t"
        )
        return dict(regions=[region], scene_features=[1.0], encoder="t", segmentation="t")

    monkeypatch.setattr(api, "inference", inference)
    client.post("/api/pair", json={"code": "123456"})
    sid = client.post("/api/sessions", json={"mode": "live"}).json()["id"]
    client.post(f"/api/sessions/{sid}/context", json={"label": "Office"})
    first = sample(client, sid, 0.0, depth=False).json()["question"]
    client.post(f"/api/sessions/{sid}/control", json={"action": "later", "reason": "timeout"})
    assert sample(client, sid, 0.5, depth=False).json()["question"] is None  # same thing: not asked again
    vector[0] = [0.0, 1.0]
    second = sample(client, sid, 1.0, depth=False).json()["question"]
    assert second and second["id"] != first["id"]  # something else is asked right away
    client.post(f"/api/sessions/{sid}/control", json={"action": "skip"})
    saved = client.get("/api/questions").json()
    assert [q["id"] for q in saved] == [first["id"]] and saved[0]["reason"] == "timeout"
    assert "features" not in saved[0]["region"] and saved[0]["place"] == "Office"
    assert [q["id"] for q in client.get("/api/questions?status=dismissed").json()] == [second["id"]]
    answer = {"question_id": first["id"], "text": "that's my stapler", "label": "stapler"}
    assert (
        client.post(f"/api/questions/{first['id']}/teach", json=answer).json()["needs_clarification"] is False
    )
    assert client.get("/api/questions").json() == []
    assert client.post(f"/api/questions/{first['id']}/teach", json=answer).status_code == 409
    stapler = next(e for e in api.store.entities("object") if e["label"] == "stapler")
    assert stapler["examples"][0]["features"] == [1.0, 0.0]


def test_resending_a_live_recording_keeps_one_copy(client, tmp_path):
    import subprocess

    from grounded_memory import api

    movie = tmp_path / "clip.mov"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=1", str(movie)],
        check=True,
    )
    client.post("/api/pair", json={"code": "123456"})
    sid = client.post("/api/sessions", json={"mode": "live"}).json()["id"]
    for _ in range(2):
        client.put(f"/api/sessions/{sid}/upload?offset=0", content=movie.read_bytes())
        session = client.post(f"/api/sessions/{sid}/upload/complete", json={}).json()
    assert len(session["recordings"]) == 1 and session["recordings"][0].endswith(".source")
    assert (api.ROOT / "assets" / session["preview"]).exists() and "source" not in session


def test_known_things_are_cut_out_of_new_questions(client, monkeypatch):
    from grounded_memory import api

    regions = {}

    async def inference(frame, previous, geometry=None):
        for name in ("both.png", "postit.png"):
            (api.WORK / name).write_bytes(b"crop")
        return dict(
            regions=[dict(r) for r in regions.values()], scene_features=[1.0], encoder="t", segmentation="t"
        )

    prompts = []
    remaining = {"features": [0.0, 1.0]}

    async def worker(function, frame, box, known, folder, geometry=None):
        assert function is api.segment_excluding
        prompts.append({"box": box, "points": [[u, v, False] for u, v in known]})
        (api.WORK / "bottle.png").write_bytes(b"crop")
        return dict(
            id="bottle",
            bbox=[0.2, 0.2, 0.2, 0.5],
            asset="bottle.png",
            salience=1,
            encoder="t",
            excluded_known=len(known),
            **remaining,
        )

    monkeypatch.setattr(api, "inference", inference)
    monkeypatch.setattr(api, "worker", worker)
    client.post("/api/pair", json={"code": "123456"})
    api.store.teach(
        kind="object", text="post-it", label="post-it", evidence={"features": [1.0, 0.0], "encoder": "t"}
    )
    sid = client.post("/api/sessions", json={"mode": "live"}).json()["id"]
    client.post(f"/api/sessions/{sid}/context", json={"label": "Desk"})
    # The unknown box swallows a recognised post-it: the question excludes it.
    regions["both"] = dict(
        id="both", bbox=[0.2, 0.2, 0.6, 0.5], asset="both.png", salience=2, features=[0.6, 0.8], encoder="t"
    )
    regions["postit"] = dict(
        id="postit",
        bbox=[0.6, 0.3, 0.1, 0.1],
        asset="postit.png",
        salience=1,
        features=[1.0, 0.0],
        encoder="t",
    )
    question = sample(client, sid, 0.0, depth=False).json()["question"]
    assert prompts[0]["box"] == [0.2, 0.2, 0.6, 0.5] and prompts[0]["points"] == [[0.65, 0.35, False]]
    assert question["region"]["id"] == "bottle" and question["region"]["excluded_known"] == 1

    # Placed earlier in 3D but not proposed in this frame: still cut out, via pose and depth.
    client.post(f"/api/sessions/{sid}/control", json={"action": "skip"})
    session = api.store.session(sid)
    session["located"] = {"p": {"label": "post-it", "extent": {"center": [-0.4, 0.0, -2.0]}, "media_time": 0}}
    api.store.save_session(session)
    del regions["postit"]
    regions["both"] = dict(regions["both"], features=[0.7, -0.7])  # unlike anything set aside
    sample(client, sid, 1.0)
    ((u, v, include),) = prompts[1]["points"]
    assert not include and abs(u - 0.5) < 0.05 and 0.25 < v < 0.65

    # If only something known is left, nothing is asked.
    client.post(f"/api/sessions/{sid}/control", json={"action": "skip"})
    remaining["features"] = [1.0, 0.0]
    regions["both"] = dict(regions["both"], features=[0.5, -0.85])
    assert sample(client, sid, 2.0).json()["question"] is None and len(prompts) == 3
