.PHONY: setup doctor dev phone check migration-status
setup:
	SAM2_BUILD_CUDA=0 uv sync --extra models
	cd web && npm ci --no-audit --no-fund
	uv run --no-sync python scripts/models.py
doctor:
	uv run --no-sync python scripts/manage.py doctor
dev:
	uv run --no-sync python scripts/manage.py dev
phone:
	uv run --no-sync python scripts/manage.py phone
check:
	uv run --no-sync pytest -q
	cd web && npm run types
migration-status:
	python3 scripts/manage.py migration-status

.PHONY: world-smoke contracts
world-smoke:
	uv run --no-sync python scripts/world_smoke.py
contracts:
	uv run --no-sync python scripts/generate_types.py

.PHONY: pipeline pipeline-smoke
pipeline:
	uv run --no-sync python -m grounded_memory.pipeline $(ARGS)
pipeline-smoke:
	uv run --no-sync python scripts/pipeline_smoke.py

.PHONY: apps-check
apps-check:
	uv run --no-sync pytest -q tests/test_pipeline.py tests/test_pipeline_workspaces.py tests/test_api.py tests/test_world_api.py
	cd web && npm run test:guidance
	cd web && npm run types
