"""
tennis_heatmap.court.homography

Homography computation and coordinate projection.

The ``CourtHomography`` class uses ``cv2.findHomography`` to compute a
perspective transform from camera pixel space → canonical top-down court
coordinate space (metres). Once calibrated on the first frame, it projects
track foot-points and ball detections into the 2D court map used for
heatmap generation.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tennis_heatmap.core.models.court import CourtCoordinate, CourtKeypoints
from tennis_heatmap.court.court_template import get_court_reference_points, COURT_WIDTH_M, COURT_LENGTH_M

logger = logging.getLogger(__name__)


class CourtHomography:
    """Encapsulates the homography matrix H that maps pixel → court coords.

    Workflow::

        h = CourtHomography()
        h.calibrate(keypoints_px)          # call once (or average over N frames)
        court_pt = h.project_point(px, py) # call per-frame per-track

    Attributes:
        H:          3×3 homography matrix (pixel → court metres).
        H_inv:      Inverse matrix (court metres → pixel).
        is_calibrated: True once :meth:`calibrate` has succeeded.
    """

    def __init__(self) -> None:
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.is_calibrated: bool = False
        self._calibration_error: Optional[float] = None

    def calibrate(
        self,
        keypoints: CourtKeypoints,
        method: int = cv2.RANSAC,
        ransac_threshold: float = 5.0,
    ) -> bool:
        """Compute homography from detected court keypoints.

        Args:
            keypoints:        Pixel-space court keypoints (from a court detector).
            method:           OpenCV homography method (default: RANSAC).
            ransac_threshold: Maximum reprojection error for RANSAC inliers.

        Returns:
            True if calibration succeeded, False otherwise.
        """
        reference = get_court_reference_points()

        n_kp = keypoints.num_keypoints
        n_ref = len(reference)

        if n_kp < 4:
            logger.error(
                "Need at least 4 keypoints for homography. Got %d.", n_kp
            )
            return False

        # Use the first min(n_kp, n_ref) matched keypoints.
        n = min(n_kp, n_ref)
        src = keypoints.keypoints[:n].astype(np.float32)   # pixel coords
        dst = reference[:n].astype(np.float32)             # court coords (metres)

        H, mask = cv2.findHomography(src, dst, method, ransac_threshold)

        if H is None:
            logger.error("cv2.findHomography returned None — calibration failed.")
            return False

        inliers = int(mask.sum()) if mask is not None else 0
        logger.info(
            "Homography calibrated. Inliers: %d/%d",
            inliers,
            n,
        )

        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.is_calibrated = True

        # Compute reprojection error for diagnostics
        projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H)
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - dst, axis=1)
        self._calibration_error = float(errors.mean())
        logger.debug("Mean reprojection error: %.4f m", self._calibration_error)

        return True

    def project_point(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """Project a single pixel point to court coordinates.

        Args:
            px: Pixel column (x).
            py: Pixel row (y).

        Returns:
            ``(court_x, court_y)`` in metres, or ``None`` if not calibrated.
        """
        if not self.is_calibrated or self.H is None:
            return None

        pt = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self.H)
        court_x, court_y = result[0, 0]
        return (float(court_x), float(court_y))

    def project_points(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Project multiple pixel points to court coordinates.

        Args:
            points: Shape (N, 2) float array of (px, py) pixel coordinates.

        Returns:
            Shape (N, 2) float array of (court_x, court_y) in metres,
            or ``None`` if not calibrated.
        """
        if not self.is_calibrated or self.H is None:
            return None

        pts = points.reshape(-1, 1, 2).astype(np.float32)
        result = cv2.perspectiveTransform(pts, self.H)
        return result.reshape(-1, 2)

    def unproject_point(self, court_x: float, court_y: float) -> Optional[Tuple[float, float]]:
        """Unproject a single court coordinate back to pixel space.

        Returns:
            ``(px, py)``, or ``None`` if not calibrated.
        """
        if not self.is_calibrated or self.H_inv is None:
            return None

        pt = np.array([[[court_x, court_y]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self.H_inv)
        px, py = result[0, 0]
        return (float(px), float(py))

    def unproject_points(self, points: np.ndarray) -> Optional[np.ndarray]:
        """Unproject multiple court coordinates back to pixel space.

        Args:
            points: Shape (N, 2) float array of (court_x, court_y) metres.

        Returns:
            Shape (N, 2) float array of (px, py), or ``None`` if not calibrated.
        """
        if not self.is_calibrated or self.H_inv is None:
            return None

        pts = points.reshape(-1, 1, 2).astype(np.float32)
        result = cv2.perspectiveTransform(pts, self.H_inv)
        return result.reshape(-1, 2)

    def project_to_court_coord(
        self,
        px: float,
        py: float,
        frame_index: int = 0,
        track_id: Optional[int] = None,
    ) -> Optional[CourtCoordinate]:
        """Project a pixel point to a :class:`CourtCoordinate`.

        Returns:
            :class:`CourtCoordinate` or ``None`` if not calibrated or
            the point is outside the court bounds.
        """
        result = self.project_point(px, py)
        if result is None:
            return None

        court_x, court_y = result
        coord = CourtCoordinate(
            x=court_x,
            y=court_y,
            frame_index=frame_index,
            track_id=track_id,
        )
        return coord

    def calibration_error_m(self) -> Optional[float]:
        """Return mean reprojection error in metres, or None if not calibrated."""
        return self._calibration_error

    def __repr__(self) -> str:
        status = f"error={self._calibration_error:.4f}m" if self.is_calibrated else "not calibrated"
        return f"CourtHomography({status})"
