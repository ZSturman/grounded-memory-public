"""Stream a Mac camera (including an iPhone Continuity Camera) into a live session.

Exercises the same WebRTC endpoint as the browser without HTTPS or certificate setup:

    uv run --no-sync python scripts/camera_stream.py --code 123456 --place "Office desk"

List devices with: ffmpeg -f avfoundation -list_devices true -i ""
"""

import argparse
import asyncio
import time

import httpx
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

parser = argparse.ArgumentParser()
parser.add_argument("--code", required=True, help="pairing code printed at startup")
parser.add_argument("--place", default="Unknown place")
parser.add_argument("--video", default="0", help="AVFoundation video device index")
parser.add_argument("--audio", default="0", help="AVFoundation audio device index, or none")
parser.add_argument("--seconds", type=float, default=300)
parser.add_argument("--api", default="http://127.0.0.1:8100/api")
args = parser.parse_args()


async def main():
    async with httpx.AsyncClient(base_url=args.api, timeout=60) as c:
        (await c.post("/pair", json={"code": args.code})).raise_for_status()
        sid = (await c.post("/sessions", json={"mode": "live"})).json()["id"]
        result = (await c.post(f"/sessions/{sid}/context", json={"label": args.place})).json()
        if result.get("needs_clarification"):
            # An existing place with this name: treat it as the same place.
            body = {"label": args.place, "resolution": "same", "previous_id": result["matches"][0]["id"]}
            (await c.post(f"/sessions/{sid}/context", json=body)).raise_for_status()
        player = MediaPlayer(
            f"{args.video}:{args.audio}",
            format="avfoundation",
            options={"framerate": "30", "video_size": "1280x720", "pixel_format": "nv12"},
        )
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        pc.addTrack(player.video)
        if player.audio:
            pc.addTrack(player.audio)
        await pc.setLocalDescription(await pc.createOffer())
        answer = await c.post(
            f"/sessions/{sid}/offer", json={"sdp": pc.localDescription.sdp, "type": "offer"}
        )
        answer.raise_for_status()
        await pc.setRemoteDescription(RTCSessionDescription(**answer.json()))
        print("Streaming session", sid, "— Ctrl-C to stop", flush=True)
        try:
            end = time.time() + args.seconds
            while time.time() < end and pc.connectionState not in ("failed", "closed"):
                await asyncio.sleep(1)
        finally:
            await pc.close()
            player.video.stop()
            await c.post(f"/sessions/{sid}/control", json={"action": "stop"})
            print("Stopped; recording saved in session", sid, flush=True)


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
