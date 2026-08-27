# Multi-stage Dockerfile for the tennis heatmap system.
# Stage 1: Builder — install uv and sync all workspace dependencies.
# Stage 2: Runtime — lean image with only runtime artifacts.

# ── Stage 1: Builder ───────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for OpenCV + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    ffmpeg libgl1 curl git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy workspace definition
COPY pyproject.toml uv.lock* ./

# Copy all workspace members
COPY libs/ ./libs/
COPY apps/ ./apps/

# Sync all workspace packages
RUN uv sync --no-dev --all-packages

# ── Stage 2: Runtime ───────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Copy application sources and configs
COPY --from=builder /app/libs ./libs
COPY --from=builder /app/apps ./apps
COPY configs/ ./configs/

# Create upload/output directories
RUN mkdir -p /tmp/tennis_heatmap_uploads /tmp/tennis_heatmap_output

EXPOSE 8000

# Default: run the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
