"""
tennis_heatmap.detectors

Importing this package auto-registers all available detector implementations
into the global DetectorRegistry. Import this package once at application
startup to make all detectors available.
"""
# Auto-register all implementations by importing their modules.
from tennis_heatmap.detectors import yolov8, detectron2  # noqa: F401

__all__ = ["yolov8", "detectron2"]
