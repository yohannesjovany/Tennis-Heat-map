"""
tennis_heatmap.core.models.detection

Shared data model for a single object detection result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ObjectClass(str, Enum):
    """Enumeration of detectable object classes."""
    PLAYER = "player"
    BALL = "ball"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates.

    Attributes:
        x1: Left edge (pixels).
        y1: Top edge (pixels).
        x2: Right edge (pixels).
        y2: Bottom edge (pixels).
    """
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        """(cx, cy) center point of the box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def foot_point(self) -> tuple[float, float]:
        """(cx, y2) bottom-center, used for player ground-plane projection."""
        cx = (self.x1 + self.x2) / 2
        return (cx, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "BoundingBox":
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "BoundingBox":
        return cls(x1=x, y1=y, x2=x + w, y2=y + h)


@dataclass
class Detection:
    """A single object detection from one frame.

    Attributes:
        bbox:         Bounding box in pixel coordinates.
        confidence:   Detection confidence in [0, 1].
        object_class: Detected class label.
        frame_index:  Source frame index (0-based).
        extra:        Optional dict for model-specific metadata (e.g. embeddings).
    """
    bbox: BoundingBox
    confidence: float
    object_class: ObjectClass
    frame_index: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def center_point(self) -> tuple[float, float]:
        return self.bbox.center

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-center — the ground contact point for players."""
        return self.bbox.foot_point
