"""File adapters retain source presentation timestamps, not decode wall time."""

import hashlib
import json
import subprocess
from pathlib import Path

import av
import cv2


def probe(path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if p.returncode:
        raise ValueError("Unsupported or damaged media: " + p.stderr[-300:])
    result = json.loads(p.stdout)
    if not any(s["codec_type"] == "video" for s in result["streams"]):
        raise ValueError("No video track found")
    return result


def frames(path, asset_dir, start=0):
    with av.open(str(path)) as container:
        origin = None
        last = -1.0
        for frame in container.decode(video=0):
            if frame.time is None:
                continue
            if origin is None:
                origin = frame.time
            offset = float(frame.time - origin)
            if offset < start or offset - last < 0.5:
                continue
            last = offset
            image = frame.to_ndarray(format="bgr24")
            key = hashlib.sha256(image.tobytes()).hexdigest() + ".jpg"
            dest = Path(asset_dir) / key
            if not dest.exists():
                cv2.imwrite(str(dest), image)
            yield offset, dest


def audio_chunks(path, seconds=2):
    """Mono PCM aligned to the video's source origin, decoded without network services."""
    import numpy as np

    with av.open(str(path)) as container:
        if not container.streams.audio:
            return
        video = container.streams.video[0] if container.streams.video else None
        origin = float(video.start_time * video.time_base) if video and video.start_time is not None else 0
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks, count, began = [], 0, None
        for source in container.decode(audio=0):
            for frame in resampler.resample(source):
                if began is None:
                    began = max(0, float(frame.time or 0) - origin)
                pcm = frame.to_ndarray().reshape(-1)
                chunks.append(pcm)
                count += len(pcm)
                if count >= seconds * 16000:
                    yield began, np.concatenate(chunks).astype(float) / 32768, 16000
                    chunks, count, began = [], 0, None
        if chunks:
            yield began, np.concatenate(chunks).astype(float) / 32768, 16000


def multimodal_samples(path, asset_dir):
    """Interleave independent decoders in timestamp order for deterministic replay."""
    import heapq

    video = ((offset, "vision", frame) for offset, frame in frames(path, asset_dir))
    sound = ((offset, "audio", (pcm, rate)) for offset, pcm, rate in audio_chunks(path))
    yield from heapq.merge(video, sound, key=lambda item: item[0])
