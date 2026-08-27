"""
tennis_heatmap.court.line_detector

Hough-line-based court keypoint detector registered as ``"hough_lines"``.

This is a classical computer vision approach that works well on clean
broadcast footage with visible court lines. For difficult angles or heavy
shadows, consider a deep-learning court detector (register under a
different key and swap in the config).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tennis_heatmap.core.interfaces.court_detector import BaseCourtDetector
from tennis_heatmap.core.models.court import CourtKeypoints
from tennis_heatmap.core.registry import CourtDetectorRegistry

logger = logging.getLogger(__name__)


@CourtDetectorRegistry.register("hough_lines")
class HoughLineCourtDetector(BaseCourtDetector):
    """Tennis court detector using Hough line transforms.

    Detects court lines via edge detection + probabilistic Hough transform,
    then derives the 12 standard court keypoints from line intersections.

    ⚠ Limitations:
    - Works best on clean broadcast footage with standard hard courts.
    - Can fail on clay/grass or with heavy shadows / spectators near lines.
    - Use the ``manual`` court detector or a deep-learning model for
      difficult footage.

    Args:
        canny_low:    Lower threshold for Canny edge detection.
        canny_high:   Upper threshold for Canny edge detection.
        hough_thresh: Accumulator threshold for Hough transform.
        min_line_len: Minimum line segment length (pixels).
        max_line_gap: Maximum gap in a line segment (pixels).
        num_keypoints: Number of keypoints to extract (up to 12).
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_thresh: int = 80,
        min_line_len: int = 100,
        max_line_gap: int = 20,
        num_keypoints: int = 12,
        **kwargs,
    ) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_thresh = hough_thresh
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.num_keypoints = num_keypoints
        logger.info("HoughLineCourtDetector initialised.")

    def detect_court(self, frame: np.ndarray, frame_index: int = 0) -> CourtKeypoints:
        """Detect court keypoints from a single BGR frame.

        The detection follows these steps:
        1. Convert to HSV and isolate white/light-blue court lines.
        2. Apply Gaussian blur and Canny edge detection.
        3. Run Probabilistic Hough transform to find line segments.
        4. Classify segments as horizontal/vertical.
        5. Compute intersection points as keypoints.

        Returns:
            :class:`CourtKeypoints` with up to ``num_keypoints`` points.
            Fewer points may be returned if not all lines are detected.
        """
        h, w = frame.shape[:2]

        # --- Step 1: Isolate court lines (white in HSV) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # --- Step 2: Edge detection ---
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # --- Step 3: Hough lines ---
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_thresh,
            minLineLength=self.min_line_len,
            maxLineGap=self.max_line_gap,
        )

        if raw_lines is None or len(raw_lines) == 0:
            logger.warning("HoughLineCourtDetector: no lines detected in frame %d.", frame_index)
            return CourtKeypoints(
                keypoints=np.empty((0, 2), dtype=np.float64),
                frame_index=frame_index,
            )

        lines = raw_lines.reshape(-1, 4)  # (N, 4) — x1,y1,x2,y2

        # --- Step 4: Classify H / V lines ---
        h_lines, v_lines = self._classify_lines(lines, angle_tol_deg=20)

        # --- Step 5: Compute intersections ---
        keypoints = self._compute_intersections(h_lines, v_lines)

        if len(keypoints) == 0:
            logger.warning(
                "HoughLineCourtDetector: no intersections found in frame %d.", frame_index
            )
            return CourtKeypoints(
                keypoints=np.empty((0, 2), dtype=np.float64),
                frame_index=frame_index,
            )

        # Sort by y (top→bottom) then x (left→right) to approximate keypoint order
        keypoints = sorted(keypoints, key=lambda p: (round(p[1] / 30) * 30, p[0]))
        keypoints_arr = np.array(keypoints[: self.num_keypoints], dtype=np.float64)

        logger.debug(
            "HoughLineCourtDetector: %d keypoints extracted from frame %d.",
            len(keypoints_arr),
            frame_index,
        )
        return CourtKeypoints(keypoints=keypoints_arr, frame_index=frame_index)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_lines(
        lines: np.ndarray, angle_tol_deg: float = 20
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split lines into approximately horizontal and vertical groups."""
        h_lines, v_lines = [], []
        tol = np.radians(angle_tol_deg)
        for x1, y1, x2, y2 in lines:
            angle = abs(np.arctan2(y2 - y1, x2 - x1))
            if angle < tol or angle > np.pi - tol:
                h_lines.append(np.array([x1, y1, x2, y2], dtype=np.float64))
            elif abs(angle - np.pi / 2) < tol:
                v_lines.append(np.array([x1, y1, x2, y2], dtype=np.float64))
        return h_lines, v_lines

    @staticmethod
    def _line_intersection(
        l1: np.ndarray, l2: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Compute the intersection of two line segments (extended as lines)."""
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None  # parallel
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)

    def _compute_intersections(
        self, h_lines: List, v_lines: List
    ) -> List[Tuple[float, float]]:
        """Compute all intersections between horizontal and vertical lines."""
        pts = []
        seen = set()
        for hl in h_lines:
            for vl in v_lines:
                pt = self._line_intersection(hl, vl)
                if pt is None:
                    continue
                # Deduplicate nearby points (within 15px)
                key = (round(pt[0] / 15), round(pt[1] / 15))
                if key not in seen:
                    seen.add(key)
                    pts.append(pt)
        return pts
