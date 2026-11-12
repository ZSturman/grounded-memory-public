# World model implementation

## Purpose and epistemic boundaries

The learner builds an inspectable model of entities, places, events, relationships and expectations from local experience and fallible teachers. A language model can extract words from an answer or help phrase an explanation; it cannot create independent evidence. Statements from a human can be wrong without being rejected for disagreeing with pretrained common knowledge.

This release predicts structured events and state changes. It does not train a generative neural world model or simulate arbitrary physics. The supported first body is a Mac plus iPhone. Robot control, intervention planning, Bluetooth and teaching between robots are later milestones.

## Data ownership and flow

`world/engine.py` is the ordered owner. It takes a lock and a SQLite `BEGIN IMMEDIATE` transaction for a state transition. Perception and transport do not decide what is true. An update preserves the observation, estimates location, evaluates outstanding forecasts, assesses the event against prior contextual frequencies, learns from it, and schedules attention. The assessed event is not first inserted into its own prior.

Schema v2 adds episodes, observations, beliefs, revisions, teachings, questions, events, exposures, predictions, playback exclusions, receipts and metadata. Legacy tables stay intact. Migration backs up the v1 database with SQLite's backup API and imports claims under stable `legacy:` IDs. Startup and subsequent imports are idempotent. Revoked claims remain retracted; utterance text and provenance are copied, not replaced by generated text.

- **Observation:** source/device/sensor identity, clock, source-relative timestamp and duration, optional tracking frame, modality, quality, feature encoder/version, measurements and retained evidence references. Each learner has a persistent ID.
- **Teaching:** the exact answer, teacher identity, question/version, frozen proposition, evidence and optional reply audio. Short yes/no answers derive meaning from that frozen proposition.
- **Belief:** subject/predicate/value, context, source, explicit certainty/status, support/conflict links, version and revision history.
- **Event:** a time-bounded observed occurrence with context and its pre-observation surprise assessment. Sound source location remains unset unless separately established.
- **Prediction:** expected event, trigger evidence, applicable belief version, source clock, start/deadline and outcome.

Inputs cannot reuse an ID for different content. Original source/sensor/timestamp/modality receipts also reject replay under a new observation ID. Exposure units group a source, modality, place and ten-second interval; adjacent frames are not independent votes. Re-encoded or independently renamed media cannot always be recognized as the same physical experience: preserve original source IDs when adapting a device.

Late observations are retained without moving the online state backward. Recovery drains the phone's disk queue in capture order. Changing tracking frame clears local geometry history; changing clock clears temporal working state. Stopped episodes remain stopped when queued capture arrives.

## Independent review and experimentation

The additive [pipeline](PIPELINE.md) copies retained input records read-only into a separate ledger for annotation revisions, immutable datasets, deterministic retrieval experiments, independent evaluation and explicit recall deployment. It does not alter this engine’s ordered online learning or production beliefs. Source snapshots containing perception features remain derived measurements, not independent teaching. The existing replay scripts remain the evaluation path for localization and whole-engine behavior.

## Localization and events

Localization compares encoder-compatible scene examples and sensor projection compatibility. It retains up to five spaced visual votes and requires three consistent recent votes to recognize a place. Current defaults require appearance similarity 0.88 with a 0.08 lead; multiple learned landmarks permit a lower appearance threshold. Place-transition support is reported separately and can order otherwise tied candidates. These are engineering thresholds, not calibrated probabilities. Histogram fallback cannot automatically certify a place.

A place is more than the view that was named. Teaching or recognition anchors a continuity chain (`world/localization.py`). While the chain holds, the place is kept even when the view no longer resembles a stored example, and each sufficiently novel view (similarity below 0.92 to every stored one) is stored on the place as an `experience` example with its observation ID, basis (`continuity` or `recognition`), sensor, tracking frame and retained frame. With an ARKit pose in the same tracking frame, the chain holds within 3 m of the anchor; this is measured self-motion, so turning the phone does not end it. Without pose, the view must stay at least 0.85 similar to one of the last three views and the chain lasts 30 s from the anchor. One break ends the chain until the next teaching or recognition, and a confident match to another place interrupts it. At most 24 learned views are kept per place; teachings are never evicted. A teacher's "no" to a place guess marks the learned view that produced the guess `excluded` (with the reply ID); it stays stored and is skipped by matching. `PlaceCandidate` reports which stored view matched and whether it was taught or learned.

Real-session evaluation (`scripts/evaluate_places.py`, read-only replay of the 2 Nov sessions: 3 places, 6 labeled sessions plus one multi-room walk, four teaching moments each):

| | Teaching session: place held | Later sessions recognized | Wrong place | Held-out place called known |
|---|---|---|---|---|
| Taught views only (previous behavior) | 45% | 0% | 0% | 0% |
| Continuity without pose | 52% | 36% | 0% | 0% |
| Continuity with ARKit pose | 85.5% | 86.3% | 0% | 0% |

These are frame counts from one afternoon; adjacent frames are not independent and nothing here is a calibrated accuracy. The multi-room walk (labeled "Office desk" at its start) stays unknown throughout, which is conservative but means a room seen only from new angles still needs a question. Pooled DINOv2 patch tokens (GeM and mean) were also measured on the same recordings and separated places worse than the CLS token (different places scored about 0.9 under GeM), so they were not adopted.

A taught place stays provisional under weak observations and is abandoned after sufficient conflicting views. Losing a known location creates a new question opportunity; `previous_place_id` keeps the last known place while the location is unknown. On restart, the old location is retained only as historical prior; fresh observations must establish current location. A late answer names its frozen view and does not assign it to a newer location without timely compatible evidence.

### Familiar and new regions in a place

`world/stability.py` keeps a per-place memory of unfamiliar regions (`world_place_regions`, created in place for existing learners). The first visit treats everything as part of the place. A region matched (crop similarity 0.85) on separate visits is a fixture and its `stability` is the fraction of the place's visits that saw it. On a later visit, an unmatched region is a `new_candidate` only when the current view matched a stored view taken in an earlier episode (otherwise the earlier visit never looked this way, which is missing coverage, recorded as `unseen`). A candidate becomes `new_here` after three sightings at least one second apart; on real replay one-frame segmentation fragments otherwise looked new. A `new_here` region asks "This wasn't here before. What is it?" at higher priority, upgrading an unshown question about it if one exists. Fixtures do not raise questions.

### Control mode and working set

`world/executive.py` derives a fixed mode from the current state each time rather than storing a separate state machine: `EXPLORE` (nothing to compare), `NOVEL` (asking where we are), `VERIFY` (a candidate resembles this view), `RELOCALIZE` (a previous place is only history), `CONFIRMED`, `RECOVER` (camera evidence stopped for 10 s) and `STOPPED`. Each mode carries a `request` for the body: today the person holding the phone, later gaze or wheels. The working set is bounded to 12 items: the place or candidate, the open place question, pending expectations, events seen in the last 30 s on the current clock, and recognized objects in view. The Mac console and phone show the mode and request.

Audio uses 32 log-frequency bands plus temporal variation, normalized for rough gain tolerance. Onsets and changes create event candidates; names come from teaching. Sound matches currently require similarity 0.93 and separation from the next candidate. Native audio is converted locally to mono 16-bit PCM at 16 kHz. Imported audio is aligned with source video timing; WebRTC uses track arrival alignment, which requires device calibration.

Visual events include tracked appearances, disappearance after brief occlusion, brightness changes and motion. Movement is measured after cancelling the camera's own motion: tracked corners (OpenCV pyramidal Lucas–Kanade) give a RANSAC homography, the earlier frame is warped onto the current one, and the largest compact residual blob is the world motion (`perception.world_motion`). A movement event needs a blob covering 2–25% of the frame, a sharp frame, and less than 15 cm of camera translation between frames (walking past nearby surfaces causes parallax). A larger residual means the camera model failed and is not reported. Frames without trackable texture fall back to the earlier rule, where camera movement suppresses motion claims. On five real recordings sampled at the phone's 0.5 s interval, the old rule fired 62 times on a static desk during handheld drift and the compensated rule 0 times; the compensated rule found hands moving objects and a dark object entering the printer-shelf view while the camera moved. Fast blurred swings are rejected by the sharpness gate. The event keeps the motion box. These descriptors do not yet identify arbitrary actions or separate overlapping sources. Geometric relationships retain the existing depth/pose derivation and human confirmation paths.

## Learning, causality and revision

Contextual event counts and a smoothed occurrence estimate produce novelty/surprise. A context needs three prior exposure units, and an event three experiences elsewhere, before a rare known event is treated as contextually surprising. An unfamiliar event can still attract attention before those conditions are met.

Observed sequences within 30 seconds produce transition beliefs. Three evidence units permit forecasts from an observed transition. These are correlations. When an observed transition reaches three evidence units without an intervention test, the learner asks an `intervene` question: could the teacher make the cause happen so it can see whether the effect follows? Agreeing records an `intervention_request` on that belief for 120 s; only a cause the learner itself observes after the reply turns its forecast into an intervention test. The outcome is appended to the belief's `interventions` separately from passive support, and a followed effect sets certainty `tested`. Declining is not evidence against the expectation, and a covered failure contradicts the belief as before. A teacher can separately provide an explanation, a cause/trigger, an expected event and a horizon. Event names resolve to previously observed events when possible; unfamiliar event names remain teacher testimony until a sensor adapter can detect them.

Forecast success needs the expected event within its interval. Failure needs at least 70% coverage from the relevant modality on the same clock. Otherwise the outcome is `unobserved`, not evidence that a statement is false. A failed covered forecast marks its supporting belief contested, preserves the statement, invalidates outstanding forecasts of that belief version and queues clarification. Revised expectations apply to subsequent triggers. Competing explanations stay separate.

Corrections, contradictions, retractions and restorations append a before/after revision. Naming corrections update the current entity label. Identity repair adds a reversible `same_as` belief: recognition uses combined exemplars, while both original identities and evidence remain stored. Retracting that identity belief separates the recognition views again. Coordinate-frame consistency remains an engineering invariant even when a teacher disputes a physical explanation.

## Questions and clients

Question kinds are `locate`, `identify`, `explain`, `clarify`, `predict` and `intervene`. A region is optional. The backend saves evidence before presenting a question. There is one proactive active question across the learner, a 60-second interruption budget, and a 20-second untouched timeout. Editing touches a question to prevent automatic deferral. A tap can bypass the proactive budget. Dismissed/deferred questions are suppressed; materially changed context or a new conflicting belief version can create another opportunity.

Answers reference a frozen question version. Refining an outline increments that version, so an offline answer to the old outline cannot silently teach the new one. Answer and revision IDs make retries idempotent. “Yes” requires a displayed proposition; “I don't know” defers instead of becoming a name. Same-named entities require explicit identity selection.

The Mac reviews predictions, surprises, belief sources and revision history. The phone presents generic question cards, frozen image/audio evidence, saved questions, optional expectations, and local speech. Typed replies and manual fields remain available if LM Studio is absent. Interpretation proposals must be exact substrings of the teacher's message and require the normal save action.

Native observations and answers are queued before transmission. A single delivery gate drains the disk queue; requests are scoped to the paired server. Errors needing human review are archived under `NeedsAttention`, preserving the payload while allowing newer capture to continue. Browser replies retain stable IDs in local storage and retry after reconnecting.

System speech marks capture chunks as playback. Review playback additionally marks the active capture clocks through `/api/v2/output`. Playback and model-origin observations may be retained for audit but cannot create ordinary sensory evidence or learning support. Timing margins are conservative and need physical acceptance.

## API

Generated OpenAPI at `/openapi.json` is the field-level contract. All `/api/v2` routes require the existing pairing cookie.

| Route | Purpose |
|---|---|
| `POST /api/v2/episodes` | Begin an unknown-location experience |
| `GET /api/v2/world?episode_id=…` | Current location, attention, beliefs, events, predictions and revisions |
| `POST /api/v2/episodes/{id}/observations` | Typed device/robot observation envelope |
| `POST /api/v2/episodes/{id}/samples` | Native multipart image, independent pose, optional depth/confidence |
| `POST /api/v2/episodes/{id}/audio` | Mono PCM WAV with stable ID, source clock/time and optional playback flag |
| `POST /api/v2/questions/{id}/interpret` | Optional local extraction from a teacher's words |
| `POST /api/v2/questions/{id}/answers` | Version-bound reply, correction or deferral |
| `POST /api/v2/questions/{id}/touch` | Keep an actively edited question open |
| `POST /api/v2/beliefs/{id}/revisions` | Version-checked correction, contradiction, retraction, restoration or identity repair |
| `POST /api/v2/episodes/{id}/control` | Pause/resume questions or stop the episode |
| `POST /api/v2/episodes/{id}/playback` | Explicit source-clock playback interval |
| `POST /api/v2/output` | Speaker output exclusion for active live capture clocks |

The old API remains a transition adapter for capture, pairing, media, relation confirmation, and outline editing. Existing legacy sessions retain their earlier question UI. New sessions created by either updated client use v2 learning.

`make contracts` generates `web/src/api.d.ts` using a disposable database. GroundedKit mirrors the client-facing world and capture contracts. Tests cover an audio-only question and stable reply versions.

## Validation and scale limits

Run focused world tests with `uv run --no-sync pytest -q tests/test_world.py tests/test_world_api.py`. `make world-smoke` is a real-HTTP, disposable synthetic replay through teaching, context learning, surprise, prediction, revision, restart and localization.

The review snapshot currently caps questions at 100, beliefs at 200, and events/predictions/revisions at 30. Older records remain in SQLite; paginated long-term history and belief retrieval are follow-up work. Acoustic evidence chunks and originals are retained, so disk use must be monitored during long sessions. Indexed episode/context queries reduce replay overhead, but no long-duration memory-growth benchmark has been accepted yet. Browser queues, physical AVAudio/ARKit timestamp alignment, speech exclusion, room/sound recognition, and reconnect behavior still need device acceptance.
