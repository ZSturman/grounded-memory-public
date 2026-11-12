import asyncio
import hashlib
import hmac
import multiprocessing
import os
import secrets
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .dialogue import DialogueAdapter
from .lifecycle import exit_with_parent
from .media import frames, multimodal_samples, probe
from .perception import analyze, preview, rank_matches, segment_excluding, segment_point
from .perception import resize as resize_region
from .perception import segment as segment_prompt
from .pipeline.routes import router as pipeline_router
from .pipeline.storage import Ledger
from .spatial import camera_from, project, relations
from .store import MemoryRepository, uid
from .world.contracts import ObservationEnvelope
from .world.engine import WorldEngine
from .world.routes import router as world_router
from .world.sensing import audio_features

ROOT = Path(os.getenv("GM_DATA_DIR") or Path.home() / "Library/Application Support/Grounded Memory")
store = MemoryRepository(ROOT)
world = WorldEngine(store)
# Analysis frames and crops are transient; only question/teaching evidence and periodic
# timeline frames are copied into assets. The recording remains the complete source.
WORK = ROOT / "work"
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir()
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
KEEP_FRAME_EVERY = 5.0
# Recognised at least this well counts as known: labelled, used for relations, not asked about again.
KNOWN = 0.8
# Laplacian variance at 640 px; phone frames taken while moving measured 0–60, steady ones 110+.
QUESTION_MIN_SHARPNESS = 100
# Seconds within which two placed things are compared; large furniture is seldom re-proposed.
RELATION_WINDOW = 20
dialogue = DialogueAdapter()


@asynccontextmanager
async def lifespan(_):
    yield
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)  # don't hold uvicorn's exit for a model load


app = FastAPI(title="Grounded Memory", version="0.2.0", lifespan=lifespan)
app.include_router(world_router(world, store, dialogue))
app.include_router(pipeline_router(Ledger(ROOT / "pipeline"), ROOT))
PAIR_CODE = os.getenv("GM_PAIR_CODE") or str(secrets.randbelow(900000) + 100000)


def pairing_secret():
    # Persisted so paired browsers survive a backend restart; cookies still expire after eight hours.
    path = ROOT / "pairing.key"
    if not path.exists():
        path.touch(mode=0o600)
        path.write_bytes(secrets.token_bytes(32))
    return path.read_bytes()


SECRET = pairing_secret()
for interrupted in store.sessions():
    if interrupted["status"] in ("live", "processing"):
        interrupted["status"] = "interrupted"
        store.save_session(interrupted)
        store.event(interrupted["id"], "interrupted_by_restart")
PAIR_ISSUED = time.time()
print("\nGrounded Memory pairing code: " + PAIR_CODE + "\n", flush=True)
pool = None
tasks = {}
peers = {}
last_question = {}
last_kept = {}
attempts = []
transcription = asyncio.Lock()
last_sweep = 0.0


def retain(name):
    """Copy transient analysis output into permanent assets."""
    if name and (WORK / name).exists() and not (ROOT / "assets" / name).exists():
        shutil.copy2(WORK / name, ROOT / "assets" / name)
    return name


def sweep_work():
    global last_sweep
    if time.time() - last_sweep < 30:
        return
    last_sweep = time.time()
    for path in WORK.iterdir():
        try:
            if time.time() - path.stat().st_mtime > 120:
                path.unlink()
        except FileNotFoundError:
            pass


def valid_cookie(raw):
    try:
        stamp, sig = raw.rsplit(".", 1)
        return 0 <= time.time() - float(stamp) < 28800 and hmac.compare_digest(
            sig, hmac.new(SECRET, stamp.encode(), "sha256").hexdigest()
        )
    except (ValueError, AttributeError):
        return False


@app.middleware("http")
async def authentication(request: Request, call_next):
    if request.url.path.startswith("/api/") and request.url.path not in (
        "/api/pair",
        "/api/status",
        "/api/ca",
    ):
        if not valid_cookie(request.cookies.get("gm_session", "")):
            return Response("Pair this browser first", status_code=401)
    try:
        return await call_next(request)
    except KeyError:
        return Response("Not found", status_code=404)
    except ValueError as e:
        return Response(str(e), status_code=422)


@app.get("/api/status")
def status(request: Request):
    return {"paired": valid_cookie(request.cookies.get("gm_session", ""))}


class Pair(BaseModel):
    code: str


@app.post("/api/pair")
def pair(body: Pair, request: Request):
    attempts[:] = [t for t in attempts if time.time() - t < 60]
    if len(attempts) >= 20:
        raise HTTPException(429, "Wait one minute before retrying.")
    attempts.append(time.time())
    if time.time() - PAIR_ISSUED > 600:
        raise HTTPException(401, "Pairing code expired; restart the application to generate a new code")
    if not hmac.compare_digest(body.code, PAIR_CODE):
        raise HTTPException(401, "Incorrect pairing code")
    stamp = str(time.time())
    cookie = stamp + "." + hmac.new(SECRET, stamp.encode(), "sha256").hexdigest()
    result = Response("{}", media_type="application/json")
    result.set_cookie(
        "gm_session",
        cookie,
        httponly=True,
        samesite="strict",
        max_age=28800,
        secure=request.headers.get("x-forwarded-proto", request.url.scheme) == "https",
    )
    return result


class SessionInput(BaseModel):
    mode: Literal["live", "recall"] = "recall"


@app.post("/api/sessions")
async def create_session(body: SessionInput):
    episode = world.create_episode(mode=body.mode)
    return store.save_session(
        dict(
            id=episode["id"],
            mode=body.mode,
            status="awaiting_context",
            created_at=time.time(),
            context=None,
            question=None,
            latest=None,
            recordings=[],
            paused=False,
        )
    )


@app.get("/api/sessions")
def sessions():
    return store.sessions()


@app.get("/api/sessions/{sid}")
def session(sid: str):
    return store.session(sid)


class Context(BaseModel):
    label: str = "Unknown place"
    unknown: bool = False
    capture_time: str | None = None
    relative_time: bool = True
    resolution: Literal["same", "same_kind", "subtype", "unrelated"] | None = None
    previous_id: str | None = None


@app.post("/api/sessions/{sid}/context")
async def context(sid: str, body: Context):
    s = store.session(sid)
    if not body.relative_time and not body.capture_time:
        raise ValueError("Provide capture time or choose relative time")
    if body.capture_time:
        from datetime import datetime

        datetime.fromisoformat(body.capture_time)
    if body.unknown:
        s["context"] = None
        s["status"] = "ready" if s["status"] == "awaiting_context" else s["status"]
        world.create_episode(sid, s["mode"])
        world.set_place(sid, None, "Unknown place")
        return store.save_session(s)
    evidence = {
        "asset": retain((s.get("latest") or {}).get("frame")),
        "features": (s.get("latest") or {}).get("scene_features", []),
        "encoder": (s.get("latest") or {}).get("encoder", "pending"),
    }
    result = store.teach(
        kind="place",
        text=body.label,
        label=body.label,
        evidence=evidence,
        resolution=body.resolution,
        previous_id=body.previous_id,
    )
    if result["needs_clarification"]:
        return result
    s["context"] = dict(
        place_id=result["entity_id"],
        label=body.label,
        unknown=body.unknown,
        capture_time=body.capture_time,
        relative_time=body.relative_time,
    )
    if s["status"] == "awaiting_context":
        s["status"] = "ready"
    s["place_guess"] = None
    store.save_session(s)
    world.create_episode(sid, s["mode"])
    world.set_place(sid, result["entity_id"], body.label)
    store.event(sid, "context_established", context=s["context"])
    return s


@app.put("/api/sessions/{sid}/upload")
async def upload(sid: str, request: Request, offset: int = 0):
    store.session(sid)
    path = ROOT / "assets" / (sid + ".upload")
    current = path.stat().st_size if path.exists() else 0
    if offset != current:
        raise HTTPException(409, detail={"next_offset": current})
    # Serialize uploads by session, preventing parallel append corruption.
    lock = upload_locks.setdefault(sid, asyncio.Lock())
    async with lock:
        current = path.stat().st_size if path.exists() else 0
        if offset != current:
            raise HTTPException(409, detail={"next_offset": current})
        with path.open("ab") as f:
            async for chunk in request.stream():
                f.write(chunk)
    return {"next_offset": path.stat().st_size}


upload_locks = {}


@app.post("/api/sessions/{sid}/upload/complete")
async def complete(sid: str):
    s = store.session(sid)
    path = ROOT / "assets" / (sid + ".upload")
    metadata = await asyncio.to_thread(probe, path)

    def digest_file():
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    digest = await asyncio.to_thread(digest_file)
    final = ROOT / "assets" / (digest + ".source")
    if not final.exists():
        path.replace(final)
    else:
        path.unlink(missing_ok=True)
    if s["mode"] == "live":
        # A phone recording of a live session: keep it as a source and playback copy only.
        # Re-sent after an interrupted upload: content-addressed, so keep one entry.
        if final.name not in s["recordings"]:
            s["recordings"].append(final.name)
    else:
        s.update(source_id=digest, source=final.name, metadata=metadata)
    # Native source stays untouched. Normalized preview is a derived browser asset.
    preview = ROOT / "assets" / (digest + ".mp4")
    if not preview.exists():
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(final),
            "-vf",
            "scale=960:-2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(preview),
            stderr=asyncio.subprocess.PIPE,
        )
        _, error = await proc.communicate()
        if proc.returncode:
            raise ValueError("Preview conversion failed: " + error.decode()[-200:])
    s["preview"] = preview.name
    store.save_session(s)
    first = s["mode"] != "live" and await asyncio.to_thread(lambda: next(frames(final, WORK), None))
    if first:
        await accept_frame(sid, first[0], first[1], None, digest, file_sensor(metadata))
    return store.session(sid)


async def worker(function, *args):
    """Run model code in the separate perception process."""
    global pool
    if pool is None:
        pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=exit_with_parent,
            initargs=(os.getpid(),),
        )
    return await asyncio.get_running_loop().run_in_executor(pool, function, *args)


async def inference(frame, previous, geometry=None):
    return await worker(analyze, str(frame), previous, str(WORK), geometry)


def geometry_paths(geometry):
    """Worker-side view of a frame's optional depth stream (file names live in WORK or assets)."""
    if not geometry:
        return None
    depth = next(
        (d / geometry["depth"] for d in (WORK, ROOT / "assets") if (d / geometry["depth"]).is_file()), None
    )
    return depth and dict(depth=str(depth), camera=geometry["camera"])


def frame_path(name):
    return next((d / name for d in (WORK, ROOT / "assets") if name and (d / name).is_file()), None)


async def outline(question):
    """Attach a preview of the question's outline on its frozen frame."""
    frame = frame_path(question["frame"])
    try:
        if frame is not None:
            question["preview"] = retain(
                await asyncio.to_thread(preview, str(frame), question["region"], str(WORK))
            )
    except (ValueError, OSError, cv2.error):
        question["preview"] = None  # a convenience; the crop and frame remain the evidence
    return question


def update_located(sid, s, result, offset):
    """Place recognised things in this capture's 3D world and count the relations between them."""
    located = s.setdefault("located", {})
    for region in result["regions"]:
        best = (region.get("matches") or [None])[0]
        if region.get("extent") and best and best["appearance"] >= KNOWN:
            located[best["id"]] = dict(label=best["label"], extent=region["extent"], media_time=offset)
    recent = {k: v for k, v in located.items() if offset - v["media_time"] <= RELATION_WINDOW}
    found = []
    for subject, predicate, obj in relations(recent):
        rid = store.observe_relation(
            sid,
            subject,
            predicate,
            obj,
            dict(
                media_time=offset,
                frame=result["frame"],
                sensor_id=result.get("sensor_id"),
                tracking_frame_id=s.get("tracking_frame_id"),
            ),
        )
        found.append(
            dict(id=rid, subject=recent[subject]["label"], predicate=predicate, object=recent[obj]["label"])
        )
    if found:
        s["relations"] = found


async def accept_frame(
    sid, offset, path, previous, source_id, sensor_id=None, geometry=None, sample_meta=None
):
    result = await inference(path, previous, geometry_paths(geometry))
    candidate = None
    s = store.session(sid)
    sensor = store.sensors().get(sensor_id, {}) if sensor_id else {}
    projection = sensor.get("projection", "unknown")
    for region in result["regions"]:
        # Every piece of evidence names its capture device so different lenses are never conflated.
        region.update(sensor_id=sensor_id, projection=projection)
    result.update(frame=Path(path).name, media_time=offset, sensor_id=sensor_id, geometry=geometry)
    s["latest"] = result
    if s.get("world_version") == 2:
        world.create_episode(sid, s["mode"])
        estimate = world.state(sid).location
        for region in result["regions"]:
            region["matches"] = rank_matches(region, world.recognizable_entities("object"), estimate.place_id)
        sample_meta = sample_meta or {}
        camera = sample_meta.get("camera") or (geometry or {}).get("camera") or {}
        tracking = sample_meta.get("tracking_frame_id", sid)
        if s.get("tracking_frame_id") != tracking:
            s["located"] = {}
            s["relations"] = []
            s["tracking_frame_id"] = tracking
        envelope = ObservationEnvelope(
            id=sample_meta.get("observation_id")
            or hashlib.sha256(f"{source_id}:{offset}:vision".encode()).hexdigest(),
            episode_id=sid,
            source_id=source_id,
            sensor_id=sensor_id or "unknown-camera",
            device_id=sample_meta.get("device_id", sensor.get("device", "unknown-device")),
            clock_id=sample_meta.get("clock_id", source_id),
            tracking_frame_id=tracking,
            timestamp=offset,
            duration=0.5,
            modality="vision",
            evidence_refs=[result["frame"]],
            features=result["scene_features"],
            encoder=result["encoder"],
            quality=min(1, result.get("sharpness", 100) / 100),
            measurements=dict(
                regions=result["regions"],
                projection=projection,
                pose=camera.get("transform"),
                geometry=geometry,
                content_digest=sample_meta.get("content_digest"),
                brightness=result.get("brightness"),
                global_motion=result.get("global_motion", 0),
                local_motion=result.get("local_motion", 0),
                camera_motion=result.get("camera_motion"),
                world_motion=result.get("world_motion"),
                world_motion_box=result.get("world_motion_box"),
            ),
        )
        world.observe(envelope, retain_evidence=lambda names: [retain(name) for name in names])
        snapshot = world.state(sid)
        estimate = snapshot.location
        active = snapshot.attention
        # Preserve the existing known-object subtraction before showing a new visual question.
        if active and active.region and active.observation_ids == [envelope.id]:
            raw_region = active.region.model_dump()
            exclusions = known_inside(s, result, raw_region, geometry)
            if exclusions:
                try:
                    refined = await worker(
                        segment_excluding,
                        str(path),
                        raw_region["bbox"],
                        exclusions,
                        str(WORK),
                        geometry_paths(geometry),
                    )
                    refined.update(sensor_id=sensor_id, projection=projection)
                    refined["matches"] = rank_matches(
                        refined, world.recognizable_entities("object"), estimate.place_id
                    )
                    if refined["matches"] and refined["matches"][0]["appearance"] >= KNOWN:
                        world.withdraw_question(active.id, "The remaining region is already recognized")
                    else:
                        retain(refined["asset"])
                        world.refine_question(dict(id=active.id, region=refined))
                except ValueError:
                    world.withdraw_question(active.id, "No distinct unfamiliar region remains")
                active = world.state(sid).attention
        if active and active.region:
            frozen = world.observation(active.observation_ids[0])
            s["question"] = dict(
                id=active.id,
                text=active.text,
                region=active.region.model_dump(),
                frame=next((r for r in active.evidence_refs if r.endswith(".jpg")), result["frame"]),
                media_time=frozen["timestamp"],
                context=dict(place_id=active.proposition.place_id, label=estimate.label),
                geometry=frozen["measurements"].get("geometry"),
            )
        else:
            s["question"] = None
        s["context"] = (
            dict(place_id=estimate.place_id, label=estimate.label, certainty=estimate.certainty)
            if estimate.place_id
            else None
        )
        update_located(sid, s, result, offset)
        store.observe(
            dict(
                id=envelope.id,
                session_id=sid,
                source_id=source_id,
                media_time=offset,
                modality="vision",
                sensor_id=sensor_id,
                evidence_ref=result["frame"] if (ROOT / "assets" / result["frame"]).exists() else None,
                context=s["context"],
                perception=result,
            )
        )
        store.save_session(s)
        return
    if not s["context"]:
        # A hypothesis for the person to confirm ("Is this the office?"); never applied automatically.
        scene = {"features": result["scene_features"], "encoder": result["encoder"]}
        s["place_guess"] = next(iter(rank_matches(scene, store.entities("place"))), None)
    if s["context"]:
        for region in result["regions"]:
            region["matches"] = rank_matches(
                region, store.entities("object"), (s["context"] or {}).get("place_id")
            )
        update_located(sid, s, result, offset)
        steady = result.get("sharpness", QUESTION_MIN_SHARPNESS) >= QUESTION_MIN_SHARPNESS
        if (
            steady
            and not s.get("question")
            and not s.get("paused")
            and time.time() - last_question.get(sid, 0) >= 15
        ):
            # Not things already put aside here (saved for later or not important).
            set_aside = [
                {"id": q["id"], "label": "", "examples": [q["region"]]}
                for q in store.pending_questions(("open", "dismissed"), s["context"]["place_id"])
            ]
            candidates = [
                r
                for r in result["regions"]
                if (not r["matches"] or r["matches"][0]["appearance"] < KNOWN)
                and not any(m["appearance"] >= 0.85 for m in rank_matches(r, set_aside))
            ]
            if candidates:
                candidate = max(candidates, key=lambda r: r["salience"])
                last_question[sid] = time.time()
    kept = False
    if candidate or offset - last_kept.get(sid, float("-inf")) >= KEEP_FRAME_EVERY:
        kept = True
        last_kept[sid] = offset
        retain(result["frame"])
    sweep_work()
    envelope = dict(
        id=hashlib.sha256(f"{sid}:{source_id}:{offset:.6f}".encode()).hexdigest(),
        session_id=sid,
        source_id=source_id,
        media_time=offset,
        received_at=time.time(),
        capture_time=(s.get("context") or {}).get("capture_time"),
        modality="vision",
        sensor_id=sensor_id,
        streams=dict(video=True, pose=bool(geometry), depth=bool(geometry)),
        evidence_ref=result["frame"] if kept else None,
        context=s["context"],
        perception=dict(result, frame=result["frame"] if kept else None),
    )
    store.observe(envelope)
    store.save_session(s)
    if candidate:
        await ask_about(sid, candidate, result, offset, path, geometry)


def known_inside(s, result, region, geometry):
    """Points (upright normalized) of things already known that lie inside a candidate's box:
    recognised regions in this frame, and things placed earlier in this capture's 3D world."""
    x, y, w, h = region["bbox"]
    inside = lambda u, v: x < u < x + w and y < v < y + h  # noqa: E731
    points = []
    for other in result["regions"]:
        best = (other.get("matches") or [None])[0]
        if other is not region and best and best["appearance"] >= KNOWN:
            u, v = other["bbox"][0] + other["bbox"][2] / 2, other["bbox"][1] + other["bbox"][3] / 2
            if inside(u, v):
                points.append((u, v))
    depth = geometry_paths(geometry)
    if depth and s.get("located"):
        camera = camera_from(geometry["camera"])
        seen = np.load(depth["depth"])["depth"]
        for placed in s["located"].values():
            uv = project(placed["extent"]["center"], camera, seen)
            if uv and inside(*uv):
                points.append(uv)
    return points


async def ask_about(sid, region, result, offset, frame, geometry):
    """Ask about a candidate, first cutting out anything already known that its outline swallowed
    (a taught post-it next to an unknown bottle). Skips it if what remains is known."""
    s = store.session(sid)
    exclude = known_inside(s, result, region, geometry)
    if exclude:
        try:
            refined = await worker(
                segment_excluding, str(frame), region["bbox"], exclude, str(WORK), geometry_paths(geometry)
            )
        except ValueError:
            return  # nothing distinct left once known things are removed
        refined.update(sensor_id=region.get("sensor_id"), projection=region.get("projection", "unknown"))
        refined["matches"] = rank_matches(refined, store.entities("object"), s["context"]["place_id"])
        if refined["matches"] and refined["matches"][0]["appearance"] >= KNOWN:
            return
        region = refined
    s = store.session(sid)
    if s.get("question") or s.get("paused") or not s["context"]:
        return  # the person started something else meanwhile
    retain(region["asset"])
    if geometry:
        retain(geometry["depth"])
    s["question"] = {
        "id": uid(),
        "text": "What is this? What would you call it?",
        "region": region,
        "media_time": offset,
        "context": s["context"],
        "frame": retain(Path(frame).name),
        "geometry": geometry,
    }
    store.save_session(s)
    await attach_outline(sid, s["question"])


async def attach_outline(sid, question):
    """Render the outline preview after saving, then apply it only if the question is still current."""
    rendered = await outline(dict(question))
    s = store.session(sid)
    if (s.get("question") or {}).get("id") == question["id"]:
        s["question"]["preview"] = rendered.get("preview")
        store.save_session(s)


def file_sensor(metadata):
    """What a recording says about its camera; lens projection is unknown unless the file states it."""
    tags = {k.lower(): v for k, v in (metadata.get("format", {}).get("tags") or {}).items()}
    video = next((x for x in metadata.get("streams", []) if x.get("codec_type") == "video"), {})
    descriptor = dict(
        modality="video",
        source="file",
        projection="unknown",
        make=tags.get("com.apple.quicktime.make"),
        model=tags.get("com.apple.quicktime.model"),
        lens=tags.get("com.apple.quicktime.camera.lens_model"),
        resolution=[video.get("width"), video.get("height")],
        codec=video.get("codec_name"),
    )
    return store.register_sensor({k: v for k, v in descriptor.items() if v is not None})


def accept_audio(sid, offset, pcm, rate, source_id, clock_id, playback=False):
    import io
    import wave

    features, measurements, encoder = audio_features(pcm, rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
    data = buffer.getvalue()
    asset = hashlib.sha256(data).hexdigest() + ".wav"
    (ROOT / "assets" / asset).write_bytes(data)
    sensor_id = store.register_sensor(dict(modality="audio", source="local-pcm", sample_rate=rate))
    world.create_episode(sid, store.session(sid)["mode"])
    return world.observe(
        ObservationEnvelope(
            id=hashlib.sha256(f"{source_id}:{offset}:audio".encode()).hexdigest(),
            episode_id=sid,
            source_id=source_id,
            sensor_id=sensor_id,
            clock_id=clock_id,
            timestamp=max(0, offset),
            duration=measurements["duration"],
            modality="audio",
            evidence_refs=[asset],
            features=features,
            encoder=encoder,
            measurements=measurements,
            origin="system_playback" if playback else "sensor",
        )
    )


async def run_file(sid):
    s = store.session(sid)
    sensor_id = file_sensor(s.get("metadata") or {})
    previous = None
    try:
        modern = s.get("world_version") == 2
        iterator = (
            multimodal_samples(ROOT / "assets" / s["source"], WORK)
            if modern
            else frames(ROOT / "assets" / s["source"], WORK)
        )
        while True:
            item = await asyncio.to_thread(lambda: next(iterator, None))
            if item is None:
                break
            if modern:
                offset, modality, content = item
                if modality == "audio":
                    pcm, rate = content
                    await asyncio.to_thread(
                        accept_audio, sid, offset, pcm, rate, s["source_id"] + ":audio", s["source_id"]
                    )
                    continue
                path = content
            else:
                offset, path = item
            await accept_frame(sid, offset, path, previous, s["source_id"], sensor_id)
            previous = str(path)
        s = store.session(sid)
        retain((s.get("latest") or {}).get("frame"))
        s["status"] = "complete"
        store.save_session(s)
    except asyncio.CancelledError:
        s = store.session(sid)
        retain((s.get("latest") or {}).get("frame"))
        s["status"] = "stopped"
        store.save_session(s)
        raise
    except Exception as e:
        s = store.session(sid)
        s["status"] = "error"
        s["error"] = str(e)
        store.save_session(s)


@app.post("/api/sessions/{sid}/start")
async def start(sid: str):
    s = store.session(sid)
    if sid in tasks and not tasks[sid].done():
        raise HTTPException(409, "Already processing")
    if not s.get("source"):
        raise ValueError("Import a video first")
    s["status"] = "processing"
    store.save_session(s)
    tasks[sid] = asyncio.create_task(run_file(sid))
    return s


@app.get("/api/ca")
def certificate_authority():
    # Public root certificate of the local HTTPS proxy, for the native app to pin after the person
    # compares its fingerprint with the one printed on the Mac.
    path = Path(
        os.getenv(
            "GM_CA_CERT", Path.home() / "Library/Application Support/Caddy/pki/authorities/local/root.crt"
        )
    )
    if not path.is_file():
        raise HTTPException(404, "No local certificate authority; run make phone")
    return Response(path.read_bytes(), media_type="application/x-pem-file")


def lean(session):
    """Session state for the native client, without feature vectors."""

    def region(r):
        return {k: v for k, v in r.items() if k != "features"}

    latest = session.get("latest") or {}
    question = session.get("question")
    return dict(
        id=session["id"],
        mode=session["mode"],
        status=session["status"],
        paused=session.get("paused", False),
        world=world.state(session["id"]).model_dump() if session.get("world_version") == 2 else None,
        context=session["context"],
        place_guess=session.get("place_guess"),
        relations=session.get("relations", []),
        question=question
        and {
            k: v
            for k, v in dict(question, region=region(question["region"])).items()
            if k not in ("original_region", "geometry")
        },
        latest=dict(
            media_time=latest.get("media_time"),
            frame=latest.get("frame"),
            regions=[region(r) for r in latest.get("regions", [])],
        ),
    )


@app.post("/api/sessions/{sid}/frames")
async def native_frame(sid: str, request: Request, media_time: float, segment: str):
    """One JPEG from the native camera; returns its analysis. The client sends the next after this returns."""
    s = store.session(sid)
    if s["mode"] != "live":
        raise ValueError("Create a live session")
    if Path(segment).name != segment or not segment:
        raise ValueError("Invalid recording name")
    data = await request.body()
    if not data or len(data) > 8_000_000:
        raise ValueError("Send one JPEG frame")
    if s["status"] in ("awaiting_context", "ready", "interrupted"):
        s["status"] = "live"
        store.save_session(s)
    path = WORK / (uid() + ".jpg")
    path.write_bytes(data)
    legacy = store.register_sensor(dict(modality="video", source="legacy-frames", projection="unknown"))
    await accept_frame(sid, media_time, path, native_previous.get(sid), segment, legacy)
    native_previous[sid] = str(path)
    return lean(store.session(sid))


class Point(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


@app.post("/api/sessions/{sid}/point")
async def point(sid: str, body: Point):
    """The person tapped something in the latest frame: ask about exactly that object."""
    s = store.session(sid)
    latest = s.get("latest") or {}
    if not s["context"] and s.get("world_version") != 2:
        raise ValueError("Establish place first")
    frame = next(
        (
            d / latest.get("frame", "")
            for d in (WORK, ROOT / "assets")
            if (d / latest.get("frame", "")).is_file()
        ),
        None,
    )
    if not latest.get("frame") or frame is None:
        raise ValueError("No recent frame; point the camera and try again")
    geometry = latest.get("geometry")
    region = await worker(segment_point, str(frame), (body.x, body.y), str(WORK), geometry_paths(geometry))
    region.update(sensor_id=latest.get("sensor_id"), projection=sensor_projection(latest.get("sensor_id")))
    region["matches"] = rank_matches(region, store.entities("object"), (s["context"] or {}).get("place_id"))
    if geometry:
        retain(geometry["depth"])
    question = await outline(
        {
            "id": uid(),
            "text": "What is this?",
            "region": region,
            "context": s["context"],
            "media_time": latest["media_time"],
            "frame": retain(latest["frame"]),
            "geometry": geometry,
            "chosen_by": "person",
        }
    )
    retain(region["asset"])
    s = store.session(sid)
    s["question"] = question
    store.save_session(s)
    if s.get("world_version") == 2:
        world.explicit_question(sid, question)
    store.event(sid, "person_pointed", region=region["id"])
    return lean(s)


class Refine(BaseModel):
    question_id: str
    points: list[tuple[float, float, bool]] = []
    granularity: Literal["smaller", "larger"] | None = None
    reset: bool = False


@app.post("/api/sessions/{sid}/question/refine")
async def refine(sid: str, body: Refine):
    """Correct the outline of the current question on its frozen frame: add include/exclude points,
    step SAM's options smaller or larger, or start over from the original proposal."""
    s = store.session(sid)
    question = s.get("question")
    if not question or question["id"] != body.question_id:
        raise HTTPException(409, "Question changed; review current evidence")
    frame = frame_path(question["frame"])
    if frame is None:
        raise ValueError("The frozen frame for this question is no longer available")
    original = question.get("original_region") or question["region"]
    geometry = geometry_paths(question.get("geometry"))
    if body.reset:
        region = original
    else:
        if body.granularity:
            # Search outward or inward for the next distinctly different outline around the current one.
            crop = frame_path(question["region"]["asset"])
            region = await worker(
                resize_region,
                str(frame),
                question["region"],
                body.granularity,
                str(WORK),
                geometry,
                str(crop),
            )
        else:
            prompt = dict(question["region"].get("prompt") or {})
            points = list(prompt.get("points") or []) + [list(p) for p in body.points]
            # The person's points may reach beyond an automatic box, so points replace it.
            prompt["box"] = None if points else question["region"]["bbox"]
            if points and not any(include for *_, include in points):
                x, y, w, h = question["region"]["bbox"]
                points.insert(0, [x + w / 2, y + h / 2, True])
            prompt["points"] = points
            region = await worker(segment_prompt, str(frame), prompt, str(WORK), geometry)
        region.update(sensor_id=original.get("sensor_id"), projection=original.get("projection", "unknown"))
        region["matches"] = rank_matches(region, store.entities("object"), question["context"]["place_id"])
        retain(region["asset"])
    question = await outline(dict(question, region=region, original_region=original))
    s = store.session(sid)
    if (s.get("question") or {}).get("id") != body.question_id:
        raise HTTPException(409, "Question changed; review current evidence")
    s["question"] = question
    store.save_session(s)
    if s.get("world_version") == 2:
        world.refine_question(s["question"])
    store.event(sid, "outline_corrected", question=body.question_id, reset=body.reset)
    return lean(s)


@app.get("/api/sessions/{sid}/relations")
def session_relations(sid: str):
    labels = {e["id"]: e["label"] for e in store.entities()}
    return [
        dict(
            id=r["id"],
            subject=labels.get(r["subject"], r["subject"]),
            predicate=r["predicate"],
            object=labels.get(r["object"], r["object"]),
            support=r["support"],
            sensors=sorted({e.get("sensor_id") for e in r["payload"]["evidence"] if e.get("sensor_id")}),
        )
        for r in sorted(store.observed_relations(sid), key=lambda r: -r["support"])
    ]


@app.post("/api/relations/{rid}/confirm")
def confirm_relation(rid: str):
    relation = next((r for r in store.observed_relations() if r["id"] == rid), None)
    if relation is None:
        raise KeyError(rid)
    labels = {e["id"]: e["label"] for e in store.entities()}
    text = f"Confirmed: {labels.get(relation['subject'])} {relation['predicate']} {labels.get(relation['object'])}"
    return {"claim_id": store.confirm_relation(rid, text)}


class Sensor(BaseModel, extra="allow"):
    """A capture device. Required identity fields; anything else the client knows is kept."""

    modality: Literal["video", "audio", "depth"]
    source: str
    projection: Literal["rectilinear", "fisheye", "equirectangular", "unknown"] = "unknown"


def sensor_projection(sensor_id):
    return store.sensors().get(sensor_id, {}).get("projection", "unknown") if sensor_id else "unknown"


@app.post("/api/v2/episodes/{sid}/samples")
@app.post("/api/sessions/{sid}/samples")
async def native_sample(
    sid: str,
    image: UploadFile = File(),
    meta: str = Form(),
    depth: UploadFile | None = File(None),
    confidence: UploadFile | None = File(None),
):
    """One time-aligned sample from a native client: a video frame plus optional pose and depth streams.

    meta JSON: {media_time, recording, sensor: {modality, source, projection, ...},
    camera?: {transform[16], intrinsics[9] column-major, resolution[w, h]}, depth_size?: [w, h]}
    Depth is little-endian uint16 millimetres in sensor orientation; confidence is uint8 (0-2).
    """
    import json

    s = store.session(sid)
    if s["mode"] != "live":
        raise ValueError("Create a live session")
    meta = json.loads(meta)
    recording = meta.get("recording", "")
    if Path(recording).name != recording or not recording:
        raise ValueError("Invalid recording name")
    sensor_id = store.register_sensor(Sensor(**meta["sensor"]).model_dump())
    data = await image.read()
    if not data or len(data) > 8_000_000:
        raise ValueError("Send one JPEG frame")
    meta["content_digest"] = hashlib.sha256(data).hexdigest()
    if meta.get("observation_id"):
        existing = world.observation(meta["observation_id"])
        if existing:
            if existing["measurements"].get("content_digest") != meta["content_digest"]:
                raise HTTPException(409, "Sample ID reused for a different image")
            return lean(s)
    name = uid()
    path = WORK / (name + ".jpg")
    path.write_bytes(data)
    geometry = None
    if depth is not None and meta.get("camera") and meta.get("depth_size"):
        w, h = meta["depth_size"]
        values = np.frombuffer(await depth.read(), "<u2")
        if values.size != w * h:
            raise ValueError("Depth size does not match depth_size")
        arrays = dict(depth=values.reshape(h, w).astype(np.float32) / 1000)
        if confidence is not None:
            levels = np.frombuffer(await confidence.read(), np.uint8)
            if levels.size == w * h:
                arrays["confidence"] = levels.reshape(h, w)
        np.savez_compressed(WORK / (name + ".depth.npz"), **arrays)
        geometry = dict(depth=name + ".depth.npz", camera=meta["camera"])
    if s["status"] in ("awaiting_context", "ready", "interrupted"):
        s["status"] = "live"
        store.save_session(s)
    await accept_frame(
        sid, float(meta["media_time"]), path, native_previous.get(sid), recording, sensor_id, geometry, meta
    )
    native_previous[sid] = str(path)
    return lean(store.session(sid))


@app.get("/api/sessions/{sid}/state")
def native_state(sid: str):
    return lean(store.session(sid))


native_previous = {}


class Control(BaseModel):
    action: Literal["stop", "skip", "later", "pause", "resume", "select"]
    region_id: str | None = None
    reason: str | None = None


@app.post("/api/sessions/{sid}/control")
async def control(sid: str, body: Control):
    s = store.session(sid)
    if s.get("world_version") == 2 and body.action in ("stop", "pause", "resume"):
        world.control(sid, body.action)
    if body.action == "stop":
        if sid in tasks:
            tasks[sid].cancel()
        if sid in peers:
            await peers.pop(sid).close()
        s["status"] = "stopped"
    elif body.action in ("skip", "later"):
        # Nothing is thrown away: "later" can be answered when the camera is off, "skip" is kept as
        # not important. Either way the same thing is not asked about again; move on now.
        if s.get("question"):
            status = "open" if body.action == "later" else "dismissed"
            reason = body.reason or ("later" if body.action == "later" else "not important")
            store.defer_question(sid, s["question"], status, reason)
            store.event(sid, "question_deferred", question=s["question"]["id"], status=status, reason=reason)
        s["question"] = None
        last_question[sid] = 0
    elif body.action in ("pause", "resume"):
        s["paused"] = body.action == "pause"
    elif body.action == "select":
        if not s["context"] and s.get("world_version") != 2:
            raise ValueError("Establish context first")
        region = next(
            (r for r in (s.get("latest") or {}).get("regions", []) if r["id"] == body.region_id), None
        )
        if not region or not (ROOT / "assets" / retain(region["asset"])).exists():
            raise ValueError("Region no longer available")
        retain(s["latest"]["frame"])
        s["question"] = {
            "id": uid(),
            "text": "What is this?",
            "region": region,
            "context": s["context"],
            "media_time": s["latest"]["media_time"],
            "frame": s["latest"]["frame"],
        }
    if body.action == "select" and s.get("world_version") == 2:
        world.explicit_question(sid, s["question"])
    return store.save_session(s)


class Teaching(BaseModel):
    question_id: str
    text: str = Field(min_length=1, max_length=5000)
    label: str = Field(min_length=1, max_length=200)
    categories: list[str] = []
    aliases: list[str] = []
    resolution: Literal["same", "same_kind", "subtype", "unrelated"] | None = None
    previous_id: str | None = None
    audio_ref: str | None = None


@app.post("/api/sessions/{sid}/teach")
async def teach(sid: str, body: Teaching):
    s = store.session(sid)
    q = s.get("question")
    if not q or q["id"] != body.question_id:
        raise HTTPException(409, "Question changed; review current evidence")
    result = teach_from(q, body)
    if not result["needs_clarification"]:
        if q["region"].get("extent"):
            s.setdefault("located", {})[result["entity_id"]] = dict(
                label=body.label, extent=q["region"]["extent"], media_time=q["media_time"]
            )
        s["question"] = None
        store.save_session(s)
        store.event(sid, "human_teaching", **result)
    return result


def teach_from(q, body):
    """Teach from a question's frozen evidence, whether answered live or later."""
    evidence = {
        **q["region"],
        "place_id": q["context"]["place_id"],
        "frame": q["frame"],
        "media_time": q["media_time"],
    }
    if body.audio_ref:
        if Path(body.audio_ref).name != body.audio_ref or not (ROOT / "assets" / body.audio_ref).is_file():
            raise ValueError("Voice evidence is missing")
        evidence["human_audio_ref"] = body.audio_ref
    result = store.teach(
        kind="object",
        text=body.text,
        label=body.label,
        evidence=evidence,
        categories=body.categories,
        aliases=body.aliases,
        resolution=body.resolution,
        previous_id=body.previous_id,
    )
    return result


def lean_question(q):
    region = {k: v for k, v in q["region"].items() if k != "features"}
    keep = ("id", "text", "frame", "preview", "media_time", "status", "reason", "session_id")
    return dict({k: q.get(k) for k in keep}, region=region, place=q["context"]["label"])


@app.get("/api/questions")
def saved_questions(status: Literal["open", "dismissed", "answered"] = "open"):
    """Things the person has not answered yet, to answer when the camera is off."""
    return [lean_question(q) for q in store.pending_questions((status,))]


@app.post("/api/questions/{qid}/teach")
async def teach_saved(qid: str, body: Teaching):
    q = store.pending_question(qid)
    if q["status"] == "answered":
        raise HTTPException(409, "Already answered")
    result = teach_from(q, body)
    if not result["needs_clarification"]:
        store.settle_question(qid, "answered", entity_id=result["entity_id"])
        store.event(q["session_id"], "human_teaching", answered_later=qid, **result)
    return result


@app.post("/api/questions/{qid}/dismiss")
def dismiss_saved(qid: str):
    store.settle_question(qid, "dismissed", reason="not important")
    return {"dismissed": True}


class Interpret(BaseModel):
    text: str
    model: str | None = None


@app.post("/api/interpret")
async def interpret(body: Interpret):
    try:
        return await dialogue.interpret(body.text, body.model)
    except Exception as e:
        raise HTTPException(503, "Assistant unavailable or unclear: " + str(e)[:180])


@app.get("/api/models")
async def models():
    try:
        return await dialogue.models()
    except Exception as e:
        raise HTTPException(503, "LM Studio unavailable: " + str(e)[:100])


@app.get("/api/memory")
def memory():
    return store.entities()


@app.post("/api/claims/{claim_id}/revoke")
async def revoke(claim_id: str):
    store.revoke(claim_id)
    return {"revoked": True}


@app.get("/api/sessions/{sid}/observations")
def observations(sid: str):
    return store.observations(sid)


@app.get("/api/assets/{name}")
def asset(name: str):
    if Path(name).name != name:
        raise HTTPException(404)
    path = ROOT / "assets" / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File()):
    data = await file.read()
    path = ROOT / "assets" / (uid() + ".voice")
    if len(data) > 20_000_000:
        raise ValueError("Voice reply is too long")
    path.write_bytes(data)

    def work():
        import mlx_whisper
        from huggingface_hub import snapshot_download

        local_model = snapshot_download("mlx-community/whisper-small-mlx", local_files_only=True)
        return mlx_whisper.transcribe(
            str(path),
            path_or_hf_repo=local_model,
            condition_on_previous_text=False,
            verbose=None,
        )

    try:
        # Whisper's Numba code aborts the whole process when entered from two threads at once.
        async with transcription:
            result = await asyncio.to_thread(work)
        return {"text": result["text"], "audio_ref": path.name, "provisional": True}
    except Exception as e:
        raise HTTPException(503, "Local transcription unavailable: " + str(e)[:150])


class Speech(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


@app.post("/api/speech")
async def speech(body: Speech):
    import subprocess
    import tempfile

    from starlette.background import BackgroundTask

    directory = tempfile.TemporaryDirectory(prefix="grounded-speech-")
    path = Path(directory.name) / "question.aiff"
    result = await asyncio.to_thread(subprocess.run, ["say", "-o", str(path), body.text], capture_output=True)
    if result.returncode:
        directory.cleanup()
        raise HTTPException(503, "Install a local macOS speech voice to hear questions.")
    return FileResponse(path, media_type="audio/aiff", background=BackgroundTask(directory.cleanup))


class Offer(BaseModel):
    sdp: str
    type: str


@app.post("/api/sessions/{sid}/offer")
async def offer(sid: str, body: Offer):
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    from aiortc.contrib.media import MediaRecorder, MediaRelay

    s = store.session(sid)
    if s["mode"] != "live":
        raise ValueError("Create a live session")
    if sid in peers:
        await peers.pop(sid).close()
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    peers[sid] = pc
    # Browsers do not reveal the camera's lens; record that explicitly rather than guess.
    browser = store.register_sensor(dict(modality="video", source="webrtc-browser", projection="unknown"))
    relay = MediaRelay()
    segment = uid() + ".mp4"
    sound = segment.replace(".mp4", ".audio.m4a")
    segment_start = max((o["media_time"] for o in store.observations(sid)), default=-0.5) + 0.5
    # Separate files: an MP4 muxer holds every video packet in memory until each track has produced
    # one, so a late or silent microphone would otherwise leave nothing on disk until it spoke.
    fragmented = {"movflags": "frag_keyframe+empty_moov"}
    recorders = {
        "video": MediaRecorder(str(ROOT / "assets" / segment), options=fragmented),
        "audio": MediaRecorder(str(ROOT / "assets" / sound), format="mp4", options=fragmented),
    }
    kinds = set()
    first_seen = {}
    finished = False
    video_task = None
    busy = False
    last = -1.0
    origin = None
    previous = None

    async def receive(track):
        nonlocal busy, last, origin, previous
        try:
            while True:
                frame = await track.recv()
                now = float(frame.time or 0)
                if origin is None:
                    origin = now
                    first_seen["video"] = time.time()
                offset = now - origin + segment_start
                if offset - last < 0.5:
                    continue
                if busy:
                    store.event(sid, "analysis_frame_skipped", media_time=offset)
                    continue
                last = offset
                busy = True
                path = WORK / (uid() + ".jpg")
                cv2.imwrite(str(path), frame.to_ndarray(format="bgr24"))

                async def process(path=path, offset=offset, prior=previous):
                    nonlocal busy
                    try:
                        await accept_frame(sid, offset, path, prior, segment, browser)
                    except Exception as e:
                        store.event(sid, "analysis_error", error=str(e))
                    finally:
                        busy = False

                asyncio.create_task(process())
                previous = str(path)
        except Exception:
            pass

    @pc.on("track")
    def on_track(track):
        nonlocal video_task
        if track.kind in recorders:
            kinds.add(track.kind)
            recorders[track.kind].addTrack(relay.subscribe(track))
        if track.kind == "video":
            video_task = asyncio.create_task(receive(relay.subscribe(track)))
        elif track.kind == "audio":

            async def first_audio(probe=relay.subscribe(track)):
                # Each recording starts at its own first packet; arrival times realign them.
                try:
                    import av

                    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
                    chunks, count, began = [], 0, None
                    audio_origin = None
                    while True:
                        frame = await probe.recv()
                        if audio_origin is None:
                            audio_origin = float(frame.time or 0)
                            first_seen["audio"] = time.time()
                        if s.get("world_version") != 2:
                            break
                        for pcm_frame in resampler.resample(frame):
                            if began is None:
                                # Track clocks differ; establish their common local arrival origin.
                                began = (
                                    segment_start
                                    + max(
                                        0, first_seen["audio"] - first_seen.get("video", first_seen["audio"])
                                    )
                                    + float(frame.time or 0)
                                    - audio_origin
                                )
                            pcm = pcm_frame.to_ndarray().reshape(-1)
                            chunks.append(pcm)
                            count += len(pcm)
                            if count >= 32000:
                                await asyncio.to_thread(
                                    accept_audio,
                                    sid,
                                    began,
                                    np.concatenate(chunks).astype(float) / 32768,
                                    16000,
                                    sound,
                                    segment,
                                )
                                chunks, count, began = [], 0, None
                except Exception as exc:
                    store.event(sid, "audio_stream_ended", detail=type(exc).__name__)
                finally:
                    probe.stop()

            asyncio.create_task(first_audio())

    @pc.on("connectionstatechange")
    async def changed():
        nonlocal finished
        if pc.connectionState not in ("failed", "closed", "disconnected") or finished:
            return
        finished = True
        if video_task:
            video_task.cancel()
        for kind in kinds:
            await recorders[kind].stop()
        store.event(sid, "stream_interrupted", state=pc.connectionState)
        audio = ROOT / "assets" / sound
        if kinds == {"audio", "video"} and audio.exists() and audio.stat().st_size > 1000:
            # Derived playback copy; the separate source tracks remain untouched.
            combined = uid() + ".mp4"
            delay = first_seen.get("audio", 0) - first_seen.get("video", 0) if len(first_seen) == 2 else 0
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-v",
                "error",
                "-itsoffset",
                f"{min(0, delay):.3f}".lstrip("-"),
                "-i",
                str(ROOT / "assets" / segment),
                "-itsoffset",
                f"{max(0, delay):.3f}",
                "-i",
                str(audio),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(ROOT / "assets" / combined),
                stderr=asyncio.subprocess.PIPE,
            )
            _, error = await proc.communicate()
            if proc.returncode:
                store.event(sid, "preview_mux_failed", error=error.decode()[-200:])
            else:
                s = store.session(sid)
                s["preview"] = combined
                store.save_session(s)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
    for kind in kinds:
        await recorders[kind].start()
    await pc.setLocalDescription(await pc.createAnswer())
    s = store.session(sid)
    s["recordings"].extend([segment, *([sound] if "audio" in kinds else [])])
    s["preview"] = segment
    s["status"] = "live"
    store.save_session(s)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.websocket("/api/sessions/{sid}/events")
async def events(ws: WebSocket, sid: str):
    if not valid_cookie(ws.cookies.get("gm_session", "")):
        await ws.close(code=1008)
        return
    await ws.accept()
    try:
        while True:
            await ws.send_json(store.session(sid))
            await asyncio.sleep(1)
    except Exception:
        pass
