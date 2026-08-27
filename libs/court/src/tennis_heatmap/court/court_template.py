"""
tennis_heatmap.court.court_template

Canonical 2D tennis court coordinate system.

The coordinate origin (0, 0) is the top-left corner of the doubles court
(far baseline, left doubles sideline). Units are **metres**.

 (0,0) ──────────────────────── (W,0)     ← far baseline
   |                              |
   |     (sw,0) ───── (W-sw,0)   |        ← far singles sideline
   |       |               |     |
   |       |   (cx, SL)    |     |        ← far service line
   |       |               |     |
   |       |   (cx, L/2)   |     |        ← net
   |       |               |     |
   |       |   (cx, L-SL)  |     |        ← near service line
   |       |               |     |
   |     (sw,L) ─── (W-sw,L)|   |        ← near singles sideline
   |                              |
 (0,L) ──────────────────────── (W,L)     ← near baseline

Where:
  W  = COURT_WIDTH_M  = 10.97 m (doubles)
  L  = COURT_LENGTH_M = 23.77 m
  sw = (W - SINGLES_WIDTH_M) / 2 = 1.37 m
  SL = SERVICE_BOX_LENGTH_M = 6.40 m
  cx = W / 2 (centre line)
"""
from __future__ import annotations

import numpy as np

# ITF standard court dimensions (metres)
COURT_LENGTH_M: float = 23.77
COURT_WIDTH_M: float = 10.97
SINGLES_WIDTH_M: float = 8.23
SERVICE_BOX_LENGTH_M: float = 6.40
NET_Y_M: float = COURT_LENGTH_M / 2  # 11.885 m from far baseline

SINGLES_OFFSET_M: float = (COURT_WIDTH_M - SINGLES_WIDTH_M) / 2  # 1.37 m


def get_court_reference_points() -> np.ndarray:
    """Return the 12 standard court keypoints in canonical court coordinates (metres).

    The returned array is shape (12, 2) — [x, y] pairs in the order
    described in :class:`~tennis_heatmap.core.models.court.CourtKeypoints`.

    These are the *world-space targets* for the homography computation.
    Match them with the corresponding pixel detections from the court detector.
    """
    W = COURT_WIDTH_M
    L = COURT_LENGTH_M
    sw = SINGLES_OFFSET_M
    SL = SERVICE_BOX_LENGTH_M
    cx = W / 2

    return np.array(
        [
            # Far baseline (doubles)
            [0.0,  0.0],   # 0: far-left
            [W,    0.0],   # 1: far-right
            # Far singles sideline / baseline corners
            [sw,   0.0],   # 2: far singles left
            [W-sw, 0.0],   # 3: far singles right
            # Far service line
            [sw,   SL],    # 4: far service left
            [W-sw, SL],    # 5: far service right
            # Near service line
            [sw,   L-SL],  # 6: near service left
            [W-sw, L-SL],  # 7: near service right
            # Near singles sideline / baseline corners
            [sw,   L],     # 8: near singles left
            [W-sw, L],     # 9: near singles right
            # Near baseline (doubles)
            [0.0,  L],     # 10: near-left
            [W,    L],     # 11: near-right
        ],
        dtype=np.float64,
    )


def court_to_pixel(
    court_coords: np.ndarray,
    canvas_width_px: int,
    canvas_height_px: int,
    margin_frac: float = 0.05,
) -> np.ndarray:
    """Convert court-space coordinates (metres) to canvas pixel coordinates.

    Useful for rendering heatmaps on a blank canvas of given size.

    Args:
        court_coords:    Shape (N, 2) array of (x, y) in metres.
        canvas_width_px: Target canvas width in pixels.
        canvas_height_px: Target canvas height in pixels.
        margin_frac:     Fractional margin around the court (0–0.5).

    Returns:
        Shape (N, 2) float array of (col, row) pixel coordinates.
    """
    margin_x = canvas_width_px * margin_frac
    margin_y = canvas_height_px * margin_frac

    usable_w = canvas_width_px - 2 * margin_x
    usable_h = canvas_height_px - 2 * margin_y

    px_col = margin_x + (court_coords[:, 0] / COURT_WIDTH_M) * usable_w
    px_row = margin_y + (court_coords[:, 1] / COURT_LENGTH_M) * usable_h

    return np.stack([px_col, px_row], axis=1)
