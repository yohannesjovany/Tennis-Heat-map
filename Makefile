.PHONY: help install dev test lint fmt api cli docker-build docker-up

PYTHON := uv run python
UV     := uv

help:
	@echo ""
	@echo "  🎾 Tennis Heatmap — Monorepo Commands"
	@echo ""
	@echo "  make install      Install all workspace packages (dev mode)"
	@echo "  make test         Run all unit tests"
	@echo "  make lint         Run ruff linter"
	@echo "  make fmt          Auto-format with ruff"
	@echo "  make api          Start the FastAPI dev server"
	@echo "  make cli          Show CLI help"
	@echo "  make list-models  List all registered models"
	@echo "  make docker-build Build the Docker image"
	@echo "  make docker-up    Start via docker compose"
	@echo ""

install:
	$(UV) sync --all-packages

test:
	$(UV) run pytest tests/ -v --tb=short

test-unit:
	$(UV) run pytest tests/unit/ -v --tb=short

lint:
	$(UV) run ruff check .

fmt:
	$(UV) run ruff format .

api:
	PYTHONPATH=apps/api/src:libs/pipeline/src:libs/heatmap/src:libs/court/src:libs/trackers/src:libs/detectors/src:libs/core/src \
	$(UV) run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

cli:
	PYTHONPATH=apps/cli/src:libs/pipeline/src:libs/heatmap/src:libs/court/src:libs/trackers/src:libs/detectors/src:libs/core/src \
	$(UV) run tennis-heatmap --help

list-models:
	PYTHONPATH=apps/cli/src:libs/pipeline/src:libs/heatmap/src:libs/court/src:libs/trackers/src:libs/detectors/src:libs/core/src \
	$(UV) run tennis-heatmap list-models

docker-build:
	docker build -f docker/Dockerfile -t tennis-heatmap:latest .

docker-up:
	docker compose -f docker/docker-compose.yml up
