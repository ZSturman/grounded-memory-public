# Grounded Memory

A local experience-driven world model that learns sensory patterns, asks questions and revises beliefs while preserving evidence.

**Status:** Research implementation. Supplied perception adapters target Apple Silicon/macOS; the project is not a general-purpose trained robot mind. Physical iPhone acceptance and recognition calibration remain outstanding. Model provisioning downloads optional weights.

[Project portfolio](https://zacharysturman.com/projects/grounded-memory)

## Quick start

```sh
make setup
make doctor
```

## Usage

```sh
make dev
```

## How it works

A Python/FastAPI runtime orders observations and persists evidence, teaching and revisable beliefs. A Next.js console reviews the learning process; a paired iPhone supplies sensory input. Collection and experimental evaluation use a separate ledger.

## Testing

```sh
.venv/bin/python -m pytest tests/test_world.py tests/test_world_api.py -q
```

See [validation and next steps](docs/NEXT_STEPS.md) for the checks performed on this snapshot. Automated fixture tests do not establish physical-device compatibility.

## Configuration and privacy

This is a curated source snapshot, published November 12, 2026. Configure your own endpoints, credentials and signing identities. Private development history, local environment files, personal recordings, runtime databases, signing materials and deployment automation are excluded. Deployment and device installation are explicit operator actions.

## Further reading

- [World Model](docs/WORLD_MODEL.md)
- [Sensors](docs/SENSORS.md)
- [Pipeline](docs/PIPELINE.md)
- [Applications](docs/APPLICATIONS.md)
- [Phone](docs/PHONE.md)

## License and attribution

Existing licenses and third-party notices apply to the files they cover. Publication of this snapshot does not grant a new license to original material; no additional license is granted where none is supplied.
