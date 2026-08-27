"""
tennis_heatmap.core.config

Pydantic-based configuration system. Load a YAML config file with
:func:`load_config` and get back a fully-validated :class:`PipelineConfig`.

Adding a new model only requires updating the YAML — no Python code changes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class DetectorConfig(BaseModel):
    """Configuration for a single detector instance."""
    name: str = Field(..., description="Registry key, e.g. 'yolov8_player'")
    model_path: Optional[str] = None
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    device: str = "cpu"
    classes: Optional[List[int]] = None  # restrict to specific COCO class IDs
    extra: Dict[str, Any] = Field(default_factory=dict)


class TrackerConfig(BaseModel):
    """Configuration for a single tracker instance."""
    name: str = Field(..., description="Registry key, e.g. 'bytetrack'")
    track_thresh: float = Field(0.5, ge=0.0, le=1.0)
    track_buffer: int = Field(30, ge=1)
    match_thresh: float = Field(0.8, ge=0.0, le=1.0)
    extra: Dict[str, Any] = Field(default_factory=dict)


class CourtDetectorConfig(BaseModel):
    """Configuration for the court keypoint detector."""
    name: str = Field("hough_lines", description="Registry key")
    model_path: Optional[str] = None
    num_keypoints: int = 12
    extra: Dict[str, Any] = Field(default_factory=dict)


class HeatmapConfig(BaseModel):
    """Configuration for heatmap generation and rendering."""
    colormap: str = "hot"
    kde_bandwidth: float = Field(0.5, description="KDE bandwidth in court metres")
    alpha: float = Field(0.7, ge=0.0, le=1.0, description="Heatmap overlay opacity")
    min_positions: int = Field(10, description="Minimum positions required to render")
    output_formats: List[Literal["png", "pdf", "json"]] = ["png"]
    dpi: int = 150


class VideoConfig(BaseModel):
    """Video reading parameters."""
    max_frames: Optional[int] = None  # None = process all frames
    skip_frames: int = Field(0, ge=0, description="Process every N+1-th frame")
    resize_width: Optional[int] = None  # Resize frames before inference


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    """Root configuration for the full tennis heatmap pipeline."""

    player_detector: DetectorConfig = Field(
        default_factory=lambda: DetectorConfig(name="yolov8_player")
    )
    ball_detector: DetectorConfig = Field(
        default_factory=lambda: DetectorConfig(name="yolov8_ball", confidence_threshold=0.3)
    )
    player_tracker: TrackerConfig = Field(
        default_factory=lambda: TrackerConfig(name="bytetrack")
    )
    ball_tracker: TrackerConfig = Field(
        default_factory=lambda: TrackerConfig(name="bytetrack", track_buffer=5)
    )
    court_detector: CourtDetectorConfig = Field(default_factory=CourtDetectorConfig)
    heatmap: HeatmapConfig = Field(default_factory=HeatmapConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)

    output_dir: str = "./output"

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to a ``.yaml`` config file.

    Returns:
        Validated :class:`PipelineConfig` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        pydantic.ValidationError: If the YAML structure is invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = yaml.safe_load(p.read_text())
    return PipelineConfig.model_validate(raw or {})


def default_config() -> PipelineConfig:
    """Return a :class:`PipelineConfig` with production-sensible defaults.

    Key choices:
    - Player detector confidence 0.30 (catches small broadcast players)
    - Ball detector confidence 0.20 (generic model; low precision)
    - resize_width 1280 (consistent speed on any input resolution)
    """
    cfg = PipelineConfig()
    cfg.player_detector.confidence_threshold = 0.30
    cfg.ball_detector.confidence_threshold = 0.20
    cfg.video.resize_width = 1280
    return cfg
