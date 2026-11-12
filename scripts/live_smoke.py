"""Synthetic WebRTC camera/audio acceptance against the disposable UI backend."""

import asyncio

import httpx
import numpy as np
from aiortc import (
    AudioStreamTrack,
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from av import VideoFrame


class Camera(VideoStreamTrack):
    async def recv(self):
        pts, base = await self.next_timestamp()
        a = np.full((120, 160, 3), 35, dtype=np.uint8)
        a[25:95, 55:105] = (40, 190, 220)
        frame = VideoFrame.from_ndarray(a, format="bgr24")
        frame.pts = pts
        frame.time_base = base
        return frame


async def main():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8100/api", timeout=45) as c:
        (await c.post("/pair", json={"code": "654321"})).raise_for_status()
        sid = (await c.post("/sessions", json={"mode": "live"})).json()["id"]
        base = "/sessions/" + sid
        (await c.post(base + "/context", json={"label": "Synthetic camera test"})).raise_for_status()
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        pc.addTrack(Camera())
        pc.addTrack(AudioStreamTrack())
        try:
            await pc.setLocalDescription(await pc.createOffer())
            answer = await c.post(base + "/offer", json={"sdp": pc.localDescription.sdp, "type": "offer"})
            answer.raise_for_status()
            await pc.setRemoteDescription(RTCSessionDescription(**answer.json()))
            for _ in range(40):
                await asyncio.sleep(0.5)
                s = (await c.get(base)).json()
                if s.get("latest"):
                    break
            assert s.get("latest"), s
            assert s["recordings"] and s["question"]
            await pc.close()
            (await c.post(base + "/control", json={"action": "stop"})).raise_for_status()
            print("PASS: WebRTC video+audio connection, recorded media, analysis and automatic question", sid)
        finally:
            await pc.close()


asyncio.run(main())
