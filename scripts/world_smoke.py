"""Disposable real-HTTP world-learning acceptance; no models, Internet, or personal data needed."""

import io
import os
import socket
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import httpx
import numpy as np


def main():
    with tempfile.TemporaryDirectory(prefix="grounded-world-smoke-") as directory:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        root = Path(directory)
        env = {
            **os.environ,
            "GM_DATA_DIR": str(root / "data"),
            "GM_PAIR_CODE": "123456",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        process = None
        log = (root / "api.log").open("w+")

        def boot():
            child = subprocess.Popen(
                [
                    os.sys.executable,
                    "-m",
                    "uvicorn",
                    "grounded_memory.api:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                env=env,
                stdout=log,
                stderr=log,
            )
            for _ in range(100):
                try:
                    if (
                        httpx.get(
                            f"http://127.0.0.1:{port}/api/status", timeout=1, trust_env=False
                        ).status_code
                        == 200
                    ):
                        return child
                except httpx.TransportError:
                    pass
                if child.poll() is not None:
                    break
                time.sleep(0.1)
            child.terminate()
            child.wait(timeout=10)
            log.seek(0)
            raise RuntimeError("World API failed to start: " + log.read()[-2000:])

        def stop(child):
            child.terminate()
            child.wait(timeout=10)

        try:
            process = boot()
            with httpx.Client(base_url=f"http://127.0.0.1:{port}/api", timeout=15, trust_env=False) as client:

                def post(path, body):
                    response = client.post(path, json=body)
                    response.raise_for_status()
                    return response.json()

                def state(episode):
                    response = client.get("/v2/world", params={"episode_id": episode})
                    response.raise_for_status()
                    return response.json()

                post("/pair", {"code": "123456"})
                episode = post("/v2/episodes", {})["id"]
                counter = 0

                def observe(t, *, modality="vision", features=None, event=None):
                    nonlocal counter
                    counter += 1
                    body = dict(
                        id=f"o{counter}",
                        episode_id=episode,
                        source_id=f"original-{counter}",
                        sensor_id=modality,
                        clock_id="smoke-boot",
                        timestamp=t,
                        duration=1,
                        modality=modality,
                        encoder="fixture",
                        features=features if features is not None else [1, 0],
                        measurements={"event": event} if event else {},
                    )
                    post(f"/v2/episodes/{episode}/observations", body)
                    return body

                first = observe(0)
                question = state(episode)["attention"]
                assert question["kind"] == "locate"
                post(f"/v2/questions/{question['id']}/answers", dict(id="place", text="Kitchen"))
                assert state(episode)["location"]["label"] == "Kitchen"
                assert not post(f"/v2/episodes/{episode}/observations", first)["accepted"]
                print("PASS: first awakening, teaching, and retry deduplication")

                buffer = io.BytesIO()
                rng = np.random.default_rng(7)
                # A reproducible test sound, explicitly synthetic; the human supplies its name.
                pcm = (rng.normal(0, 3000, 32000)).clip(-32768, 32767).astype("<i2")
                with wave.open(buffer, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16000)
                    wav.writeframes(pcm.tobytes())

                def sound(t, ident):
                    response = client.post(
                        f"/v2/episodes/{episode}/audio",
                        params=dict(
                            observation_id=ident,
                            source_id="recorded-sound-" + ident,
                            clock_id="smoke-boot",
                            timestamp=t,
                        ),
                        content=buffer.getvalue(),
                    )
                    response.raise_for_status()

                sound(1, "new-sound")
                question = next(q for q in state(episode)["questions"] if q["kind"] == "identify")
                post(f"/v2/questions/{question['id']}/answers", dict(id="sound-name", text="Running water"))
                for t in (10, 20, 30, 40):
                    sound(t, f"kitchen-{t}")
                # Establish a different learned place through a new episode, without hardcoded room rules.
                episode = post("/v2/episodes", {})["id"]
                observe(0, features=[0, 1])
                question = next(q for q in state(episode)["questions"] if q["kind"] == "locate")
                post(f"/v2/questions/{question['id']}/answers", dict(id="living", text="Living room"))
                for t in (1, 11, 21, 31):
                    observe(t, modality="audio", features=[0, 1])
                sound(41, "living-water")
                assert state(episode)["recent_events"][-1]["surprising"]
                question = next(q for q in state(episode)["questions"] if q["kind"] == "explain")
                answer = post(
                    f"/v2/questions/{question['id']}/answers",
                    dict(
                        id="why",
                        text="A bell should follow the water",
                        expected_event="bell",
                        horizon_seconds=3,
                    ),
                )
                sound(51, "water-again")
                for t in (51, 52, 53, 54):
                    observe(t, modality="event", features=[])
                belief = next(b for b in state(episode)["beliefs"] if b["id"] == answer["belief_id"])
                assert belief["status"] == "contested" and belief["value"] == "A bell should follow the water"
                assert any(q["kind"] == "clarify" for q in state(episode)["questions"])
                print("PASS: learned context, sound surprise, causal testimony, prediction, and revision")
                learner_id = state(episode)["learner_id"]
                stop(process)
                process = None
                process = boot()
                restarted = state(episode)
                assert restarted["learner_id"] == learner_id and restarted["location"]["place_id"] is None
                assert any(
                    b["id"] == belief["id"] and b["status"] == "contested" for b in restarted["beliefs"]
                )
                for t in (60, 61, 62):
                    observe(t, features=[0, 1])
                assert state(episode)["location"]["label"] == "Living room"
                print("PASS: cold restart preserves beliefs and requires fresh localization evidence")
        finally:
            if process and process.poll() is None:
                stop(process)
            log.close()


if __name__ == "__main__":
    main()
