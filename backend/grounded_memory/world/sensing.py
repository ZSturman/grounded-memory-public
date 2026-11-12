"""Local acoustic descriptors and conservative visual change detection; no semantic labels."""

import numpy as np

from .localization import similarity


def audio_features(pcm, sample_rate):
    samples = np.asarray(pcm, dtype=float).reshape(-1)
    if not 8000 <= sample_rate <= 96000 or not 0 < samples.size <= sample_rate * 10:
        raise ValueError("Send up to ten seconds of mono PCM at 8–96 kHz")
    rms = float(np.sqrt(np.mean(samples * samples)))
    # 32 log-frequency bands plus temporal variation. Relative power makes volume less dominant.
    size = 1024
    padded = np.pad(samples, (0, max(0, size - len(samples))))
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)[::512]
    power = abs(np.fft.rfft(windows * np.hanning(size), axis=1)) ** 2
    frequencies = np.fft.rfftfreq(size, 1 / sample_rate)
    bounds = np.geomspace(40, min(12000, sample_rate / 2), 33)
    bands = np.stack(
        [
            power[:, (frequencies >= lo) & (frequencies < hi)].mean(axis=1)
            if np.any((frequencies >= lo) & (frequencies < hi))
            else np.zeros(len(power))
            for lo, hi in zip(bounds[:-1], bounds[1:])
        ],
        axis=1,
    )
    bands = np.log1p(bands / (bands.mean() + 1e-12))
    vector = np.concatenate([bands.mean(axis=0), bands.std(axis=0)])
    vector /= np.linalg.norm(vector) + 1e-9
    return vector.tolist(), dict(rms=rms, duration=len(samples) / sample_rate), "spectral-temporal-v1"


def sensory_events(observation, episode, entities):
    m = observation.measurements
    events = []
    if observation.modality == "audio" and m.get("rms", 0) >= 0.008:
        matches = []
        for entity in entities:
            if entity["kind"] != "sound":
                continue
            score = max(
                (
                    similarity(observation.features, e.get("features", []))
                    for e in entity.get("examples", [])
                    if e.get("encoder") == observation.encoder
                ),
                default=0,
            )
            if score >= 0.93:
                matches.append((score, entity))
        matches.sort(key=lambda pair: pair[0], reverse=True)
        match = (
            matches[0][1]
            if matches and (len(matches) == 1 or matches[0][0] - matches[1][0] >= 0.03)
            else None
        )
        if match:
            events.append(
                dict(key="sound:" + match["id"], label=match["label"], kind="sound", entity_id=match["id"])
            )
        else:
            # Reuse unnamed exemplars within an episode; a changing sound remains eligible for attention.
            sounds = episode.setdefault("unfamiliar_sounds", [])
            item = next((s for s in sounds if similarity(s["features"], observation.features) > 0.93), None)
            if not item:
                item = dict(key="sound:" + observation.id, features=observation.features)
                sounds.append(item)
            events.append(dict(key=item["key"], label="Unfamiliar sound", kind="sound", unnamed=True))
    if observation.modality == "vision":
        previous = episode.get("visual", {})
        brightness = m.get("brightness")
        moving_camera = False
        pose = m.get("pose")
        if pose and previous.get("pose") and len(pose) == len(previous["pose"]) == 16:
            moving_camera = np.linalg.norm(np.asarray(pose) - np.asarray(previous["pose"])) > 0.05
        elif not pose:
            moving_camera = m.get("global_motion", 0) > 0.12
        if (
            brightness is not None
            and previous.get("brightness") is not None
            and abs(brightness - previous["brightness"]) > 0.22
        ):
            events.append(dict(key="change:brightness", label="Sudden brightness change", kind="change"))
        world = m.get("world_motion")
        if world is not None:
            # Camera motion is already cancelled. Walking past nearby surfaces still causes parallax,
            # a blurred frame is not evidence, and a frame-sized residual means the model failed.
            translating = (
                pose
                and previous.get("pose")
                and len(pose) == len(previous["pose"]) == 16
                and np.linalg.norm(np.subtract(pose[12:15], previous["pose"][12:15])) > 0.15
            )
            if observation.quality >= 0.5 and 0.02 <= world <= 0.25 and not translating:
                events.append(
                    dict(key="change:motion", label="Movement", kind="change", box=m.get("world_motion_box"))
                )
        elif m.get("local_motion", 0) > 0.08 and not moving_camera:
            events.append(dict(key="change:motion", label="Movement", kind="change"))
        current = {}
        for region in m.get("regions", []):
            match = next(iter(region.get("matches") or []), None)
            if match and match["appearance"] >= 0.8:
                key = "object:" + match["id"]
                current[key] = dict(label=match["label"], timestamp=observation.timestamp)
                events.append(
                    dict(key=key, label=match["label"], kind="object", entity_id=match["id"], region=region)
                )
                if key not in previous.get("objects", {}) and previous and not moving_camera:
                    events.append(
                        dict(key="appeared:" + match["id"], label=match["label"] + " appeared", kind="change")
                    )
            elif observation.quality >= 0.5:
                tracks = episode.setdefault("region_tracks", [])
                tracks[:] = [t for t in tracks if observation.timestamp - t["last"] <= 10]
                track = next(
                    (t for t in tracks if similarity(t["features"], region.get("features", [])) >= 0.9), None
                )
                if not track:
                    track = dict(
                        id=region["id"], features=region.get("features", []), last=observation.timestamp
                    )
                    tracks.append(track)
                track["last"] = observation.timestamp
                events.append(
                    dict(
                        key="region:" + track["id"],
                        label="Unfamiliar object",
                        kind="object",
                        unnamed=True,
                        region=region,
                    )
                )
        missing = episode.setdefault("missing_objects", {})
        for key, old in previous.get("objects", {}).items():
            if key not in current and not moving_camera:
                missing.setdefault(key, dict(old, since=observation.timestamp))
        for key in list(missing):
            if key in current or moving_camera:
                missing.pop(key)
            elif observation.timestamp - missing[key]["since"] >= 5:
                events.append(
                    dict(
                        key="disappeared:" + key,
                        label=missing.pop(key)["label"] + " is no longer visible",
                        kind="change",
                    )
                )
        episode["visual"] = dict(brightness=brightness, pose=pose, objects=current)
    # Typed event adapters report measured events, not model-generated claims.
    if observation.modality == "event" and isinstance(m.get("event"), str):
        events.append(dict(key="event:" + m["event"], label=m["event"], kind="event"))
    return events
