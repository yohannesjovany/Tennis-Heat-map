"""
tennis_heatmap.core

Core package providing interfaces, data models, and the plugin registry
for the tennis heatmap system. All other packages depend on this.
"""
from tennis_heatmap.core.registry import DetectorRegistry, TrackerRegistry, CourtDetectorRegistry
from tennis_heatmap.core.config import PipelineConfig, load_config

__all__ = [
    "DetectorRegistry",
    "TrackerRegistry",
    "CourtDetectorRegistry",
    "PipelineConfig",
    "load_config",
]
