"""Disposable real-HTTP import -> context -> perception -> teaching -> restart check."""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

with tempfile.TemporaryDirectory(prefix="grounded-smoke-") as d:
    root = Path(d)
    video = root / "fixture.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 4, (160, 120))
    for i in range(8):
        image = np.full((120, 160, 3), 40, np.uint8)
        cv2.rectangle(image, (30 + i, 30), (80 + i, 90), (60, 190, 240), -1)
        writer.write(image)
    writer.release()
    env = {**os.environ, "GM_DATA_DIR": str(root / "data"), "GM_PAIR_CODE": "123456"}

    def boot():
        p = subprocess.Popen(
            [os.sys.executable, "-m", "uvicorn", "grounded_memory.api:app", "--port", "8193"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            try:
                if httpx.get("http://127.0.0.1:8193/api/status").status_code == 200:
                    return p
            except httpx.ConnectError:
                pass
            time.sleep(0.1)
        p.terminate()
        raise RuntimeError("API failed to start")

    p = boot()
    try:
        with httpx.Client(base_url="http://127.0.0.1:8193/api", timeout=180) as c:
            c.post("/pair", json={"code": "123456"}).raise_for_status()
            s = c.post("/sessions", json={"mode": "recall"}).json()
            sid = s["id"]
            base = "/sessions/" + sid
            c.put(base + "/upload", content=video.read_bytes()).raise_for_status()
            c.post(base + "/upload/complete", json={}).raise_for_status()
            c.post(base + "/context", json={"label": "Test room"}).raise_for_status()
            c.post(base + "/start", json={}).raise_for_status()
            for _ in range(180):
                s = c.get(base).json()
                if s["status"] in ("complete", "error"):
                    break
                time.sleep(0.5)
            assert s["status"] == "complete", s
            assert s["question"] and len(c.get(base + "/observations").json()) == 4
            c.post(
                base + "/teach",
                json={"question_id": s["question"]["id"], "text": "yellow object", "label": "yellow object"},
            ).raise_for_status()
            print(
                "PASS: real HTTP upload, timestamped observation, context gate, automatic question and teaching"
            )
        p.terminate()
        p.wait(timeout=15)
        p = boot()
        with httpx.Client(base_url="http://127.0.0.1:8193/api") as c:
            c.post("/pair", json={"code": "123456"})
            assert len(c.get("/memory").json()) == 2
            assert c.get("/sessions/" + sid).json()["status"] == "complete"
            print("PASS: cold restart preserves learned entities and session")
    finally:
        p.terminate()
        p.wait(timeout=15)
