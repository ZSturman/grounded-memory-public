# Sensors, streams and spatial relations

Every observation names the device that produced it. Evidence from different capture devices is never silently pooled.

## Sensor descriptors

A client describes each capture device once per sample; the backend stores it in the `sensors` table under a stable id (a hash of the identity fields) and stamps that `sensor_id` on every observation, region and taught example.

| Field | Required | Meaning |
|---|---|---|
| `modality` | yes | `video`, `audio` or `depth` |
| `source` | yes | capture path, e.g. `arkit-world-tracking`, `webrtc-browser`, `file`, `legacy-frames` |
| `projection` | yes (default `unknown`) | `rectilinear`, `fisheye`, `equirectangular` or `unknown` |
| anything else | no | e.g. `device`, `platform`, `camera`, `lens`, `resolution`, `fps`, `depth` |

Per-frame values (intrinsics, pose) are not part of the identity. Sources that cannot know the lens say so: browser cameras and imported files are `projection: unknown`; files keep any make/model/lens they embed.

Matching (`perception.rank_matches`) never compares examples with different known projections; a fisheye view and a rectilinear view of the same item are not interchangeable. Each match reports `sensors`: `same` (same device), `compatible` (same projection), or `unverified` (either side unknown).

## Streams of one sample

`POST /api/v2/episodes/{id}/samples` (multipart; the legacy `/api/sessions/{id}/samples` route is retained) carries one time-aligned sample. Only `image` and `meta` are required; every other stream is optional, so a device without depth or pose still works.

- `image`: upright JPEG.
- `meta`: JSON `{observation_id, clock_id, tracking_frame_id, media_time, recording, sensor, camera?, depth_size?}`. IDs persist through retries. `camera` is ARKit pose `transform` (16, column-major), `intrinsics` (9, column-major) and sensor `resolution` [w, h].
- `depth`: little-endian uint16 millimetres, sensor orientation, `depth_size` [w, h].
- `confidence`: uint8 per depth pixel (0 low, 1 medium, 2 high).

The backend adds derived motion measurements to each vision observation: `camera_motion` (median tracked-corner shift between the previous and current analysed frame, as a fraction of the image diagonal), and `world_motion` with `world_motion_box` (the largest residual region after cancelling that camera motion with a homography). They are absent when the frame has too little texture to track. See [WORLD_MODEL](WORLD_MODEL.md) for how movement events use them. Pose is retained even when depth is absent. Each AR tracking reset has a new `tracking_frame_id`; metric positions are never compared across unaligned frames. Source-relative timestamps and clock IDs prevent unrelated clocks from being silently pooled.

Native audio is sent independently to `POST /api/v2/episodes/{id}/audio` as mono 16-bit PCM WAV with `observation_id`, `source_id`, `clock_id`, `timestamp`, `device_id` and optional `playback=true` query fields. The supplied app converts to 16 kHz locally. The route validates up to ten seconds per chunk and computes spectral/temporal descriptors; learned sound names come from teachers. Imported recordings and WebRTC feed the same audio learner. Single-microphone observations have a heard-at context, not a measured source direction.

Other adapters can submit a typed `ObservationEnvelope` to `/api/v2/episodes/{id}/observations`, with modality `vision`, `audio`, `pose`, `depth` or measured `event`. Identity, time, quality, encoder/version, measurements and evidence references are explicit. Standalone pose/depth envelopes are preserved; they do not invent scene/object labels. Raw observation IDs are content-bound, and source/sensor/time receipts make replay idempotent.

System-generated speech and playback are excluded from learning through capture-origin flags and source-clock intervals. `/api/v2/output` projects review playback onto recent active live audio clocks. See [WORLD_MODEL](WORLD_MODEL.md) for the exact separation between evidence, beliefs and attention.

## World-model transition

New client sessions use v2 questions and ongoing location estimates. The legacy known-region, context and queue rules below describe the preserved v1 path; v2 uses the attention/learning services, permits unknown location, and can ask about a known thing when its context or behavior is surprising. Spatial relation confirmation and outline refinement remain shared adapters.

## Spatial relations

With pose and depth, each segmented region gets a world-space `extent` (trimmed 10–90th percentile box in ARKit's gravity-aligned metres, `spatial.world_extent`). Recognised (appearance ≥ 0.8) and newly taught things are placed per capture session; extents from different sessions are never compared, because each ARKit session has its own world origin.

`spatial.relate` derives `on`, `above`, `inside` and `next_to` between things placed within 20 seconds of each other (`RELATION_WINDOW`). Each derivation adds support to an observed relation (`observed_relations` table, with recent evidence frames and sensor ids). Observed relations are perception, not teaching: they become claims only when a person confirms them (`POST /api/relations/{id}/confirm`), which records a human utterance as the claim's source.

## Correcting outlines

`POST /api/sessions/{id}/question/refine` re-segments the current question on its frozen frame: add include/exclude `points`, step `granularity` `smaller`/`larger` through SAM's options, or `reset` to the original proposal. The question keeps its id, and a `preview` image shows the outline being taught.

## Not asking about known things

A thing recognised at appearance ≥ 0.8 (`KNOWN`) is labelled, placed and not asked about. Before asking about an unrecognised region, the backend looks for known things inside its box: recognised regions in the same frame, and things placed earlier in the capture's 3D world, projected into the current view with the frame's pose and checked against LiDAR depth for occlusion (`spatial.project`). Each is segmented on its own and its whole outline subtracted (`perception.segment_excluding`); if little remains, or the remainder is itself known, nothing is asked.
