"""
tennis_heatmap.core.models.court

Data models for court detection results and 2D court coordinate space.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Standard ITF tennis court dimensions in metres.
COURT_LENGTH_M = 23.77   # baseline to baseline
COURT_WIDTH_M = 10.97    # doubles sideline to sideline
SINGLES_WIDTH_M = 8.23   # singles sideline to sideline
SERVICE_BOX_LENGTH_M = 6.40


@dataclass
class CourtKeypoints:
    """Detected keypoints for a tennis court in pixel coordinates.

    The standard keypoint ordering follows the ITF court diagram
    (top-left origin, going clockwise):

      0 ─────────────── 1          ← far baseline (doubles)
      |   2 ─────── 3   |
      |   |  4   5  |   |          ← far service line
      |   |         |   |
      |   |  6   7  |   |          ← near service line
      |   8 ─────── 9   |
      10 ─────────────── 11        ← near baseline (doubles)
    """
    keypoints: np.ndarray  # shape (N, 2), pixel coords (x, y)
    confidence_scores: Optional[np.ndarray] = None  # shape (N,) per-keypoint confidence
    frame_index: int = 0

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)

    def get_keypoint(self, idx: int) -> tuple[float, float]:
        kp = self.keypoints[idx]
        return (float(kp[0]), float(kp[1]))


@dataclass
class CourtCoordinate:
    """A projected 2D point in the canonical top-down court coordinate system.

    The origin (0, 0) is the top-left corner of the doubles court.
    Units are metres. The x-axis runs along the baseline width,
    the y-axis runs along the court length (baseline to baseline).

    Attributes:
        x:           Horizontal position in metres.
        y:           Vertical position in metres (depth from top baseline).
        frame_index: Source frame.
        track_id:    Optional track ID the point belongs to.
    """
    x: float
    y: float
    frame_index: int = 0
    track_id: Optional[int] = None

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def is_in_court(self) -> bool:
        """Returns True if the coordinate falls within the doubles court boundary."""
        return 0.0 <= self.x <= COURT_WIDTH_M and 0.0 <= self.y <= COURT_LENGTH_M

    def is_near_court(self, margin_m: float = 2.0) -> bool:
        """Returns True if the coordinate is within *margin_m* metres of the court.

        Use this instead of :meth:`is_in_court` when homography accuracy is
        uncertain (e.g. only partial court lines detected). A 2 m margin is
        generous enough to catch players near the baseline/sideline who are
        technically just outside the doubles line.
        """
        return (
            -margin_m <= self.x <= COURT_WIDTH_M + margin_m
            and -margin_m <= self.y <= COURT_LENGTH_M + margin_m
        )
