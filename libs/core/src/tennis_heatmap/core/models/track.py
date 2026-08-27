"""
tennis_heatmap.core.models.track

Track model — a detection that has been assigned a persistent identity
across multiple frames by a tracker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tennis_heatmap.core.models.detection import BoundingBox, ObjectClass


@dataclass
class Track:
    """A tracked object with a stable identity across frames.

    Attributes:
        track_id:     Unique integer ID assigned by the tracker.
        bbox:         Current bounding box in pixel coordinates.
        object_class: Object class (PLAYER or BALL).
        confidence:   Detection confidence of the underlying detection.
        frame_index:  Frame where this track state was last updated.
        is_confirmed: True once the track has been confirmed by the tracker
                      (i.e., survived its minimum hit-streak threshold).
        age:          Number of frames since the track was first created.
        extra:        Optional model-specific metadata.
    """
    track_id: int
    bbox: BoundingBox
    object_class: ObjectClass
    confidence: float
    frame_index: int
    is_confirmed: bool = True
    age: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def center_point(self) -> tuple[float, float]:
        return self.bbox.center

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-center of the bbox — ground contact point for players."""
        return self.bbox.foot_point
