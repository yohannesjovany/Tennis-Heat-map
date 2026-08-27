"""
Tennis Heatmap — FastAPI REST API + Web UI
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tennis_heatmap.core.config import default_config, load_config
from tennis_heatmap.pipeline.video_pipeline import TennisHeatmapPipeline, PipelineProgress

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Tennis Heatmap API",
    description="AI-based heatmap generation for tennis match videos.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static web UI
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

UPLOAD_DIR = Path("/tmp/tennis_heatmap_uploads")
OUTPUT_DIR = Path("/tmp/tennis_heatmap_output")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Job state (in-memory for v1; replace with Redis/DB for production)
# ---------------------------------------------------------------------------

_jobs: Dict[str, Dict[str, Any]] = {}


class JobStatus(BaseModel):
    job_id: str
    status: str  # "queued" | "processing" | "done" | "error"
    progress_pct: float = 0.0
    message: str = ""
    player_count: int = 0
    ball_positions: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def _process_video(
    job_id: str,
    video_path: Path,
    config_path: Optional[str],
) -> None:
    _jobs[job_id]["status"] = "processing"

    cfg = load_config(config_path) if config_path else default_config()
    cfg.output_dir = str(OUTPUT_DIR / job_id)

    pipeline = TennisHeatmapPipeline(config=cfg)

    def on_progress(p: PipelineProgress) -> None:
        _jobs[job_id]["progress_pct"] = round(p.fraction * 100, 1)
        _jobs[job_id]["ball_positions"] = p.ball_positions_accumulated
        _jobs[job_id]["player_positions"] = p.player_positions_accumulated

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: pipeline.run(video_path, run_id=job_id, progress_callback=on_progress),
        )
        _jobs[job_id].update(
            {
                "status": "done",
                "progress_pct": 100.0,
                "player_count": len(result.player_heatmaps),
                "ball_positions": result.metadata.get("ball_positions", 0),
                "result": result,
            }
        )
    except Exception as exc:
        logger.exception("Pipeline error for job %s", job_id)
        _jobs[job_id].update({"status": "error", "error": str(exc)})
    finally:
        # Clean up uploaded video
        video_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web UI."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>Tennis Heatmap API</h1><p><a href='/docs'>Swagger Docs →</a></p>")


@app.post("/analyze", response_model=JobStatus, status_code=202)
async def analyze(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Tennis match video file"),
    config_path: Optional[str] = None,
) -> JobStatus:
    """Upload a video and start async heatmap generation.

    Returns a ``job_id`` to poll for results via ``GET /jobs/{job_id}``.
    """
    job_id = str(uuid.uuid4())
    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{job_id}{suffix}"

    # Save upload
    async with aiofiles.open(video_path, "wb") as f:
        content = await video.read()
        await f.write(content)

    _jobs[job_id] = {"status": "queued", "progress_pct": 0.0}
    background_tasks.add_task(_process_video, job_id, video_path, config_path)

    return JobStatus(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    """Poll the status of a processing job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return JobStatus(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress_pct=job.get("progress_pct", 0.0),
        player_count=job.get("player_count", 0),
        ball_positions=job.get("ball_positions", 0),
        error=job.get("error"),
    )


@app.get("/results/{job_id}/images")
async def get_result_images(job_id: str):
    """Return base64-encoded PNG heatmap images for a completed job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail="Job not yet complete.")

    result = job.get("result")
    if result is None:
        raise HTTPException(status_code=500, detail="Result object missing.")

    images = {}

    for track_id, img in result.player_heatmaps.items():
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images[f"player_{track_id}"] = base64.b64encode(buf.getvalue()).decode()

    if result.ball_heatmap:
        buf = io.BytesIO()
        result.ball_heatmap.save(buf, format="PNG")
        images["ball"] = base64.b64encode(buf.getvalue()).decode()

    return {"job_id": job_id, "images": images, "metadata": result.metadata}


@app.get("/results/{job_id}/download/{filename}")
async def download_file(job_id: str, filename: str):
    """Download a specific output file (PNG or PDF)."""
    safe = Path(filename).name  # strip any directory traversal
    path = OUTPUT_DIR / job_id / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(path), filename=safe)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
