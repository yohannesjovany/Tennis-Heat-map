"""
tennis_heatmap.core.interfaces.court_detector

Abstract base class for court keypoint detection implementations.
"""
from __future__ import annotations

import abc

import numpy as np

from tennis_heatmap.core.models.court import CourtKeypoints


class BaseCourtDetector(abc.ABC):
    """Strategy interface for tennis court detectors.

    A court detector takes a video frame and returns the pixel coordinates
    of the standard court keypoints used for homography computation.

    All court detector implementations should be registered via::

        @CourtDetectorRegistry.register("my_court_detector")
        class MyCourtDetector(BaseCourtDetector):
            ...
    """

    @abc.abstractmethod
    def detect_court(self, frame: np.ndarray, frame_index: int = 0) -> CourtKeypoints:
        """Detect tennis court keypoints in a single frame.

        Args:
            frame:       H×W×3 uint8 BGR numpy array (OpenCV format).
            frame_index: 0-based frame index.

        Returns:
            :class:`~tennis_heatmap.core.models.court.CourtKeypoints` containing
            the detected pixel-space keypoints and optional confidence scores.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
