"""
tennis_heatmap.heatmap.kde_generator

Kernel Density Estimation (KDE) for smooth heatmap generation.

Takes a list of (x, y) court-space positions and produces a 2D density
grid normalised to [0, 1] for rendering.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy.stats import gaussian_kde

from tennis_heatmap.court.court_template import COURT_WIDTH_M, COURT_LENGTH_M

logger = logging.getLogger(__name__)


class KDEHeatmapGenerator:
    """Generates a 2D density grid from court-space positions using Gaussian KDE.

    Args:
        bandwidth:   KDE bandwidth (Scott's rule if ``None``).
        grid_cols:   Number of grid columns (court width direction).
        grid_rows:   Number of grid rows (court length direction).
        court_width: Court width in metres (default: doubles width).
        court_length: Court length in metres (baseline to baseline).
    """

    def __init__(
        self,
        bandwidth: Optional[float] = None,
        grid_cols: int = 220,
        grid_rows: int = 475,
        court_width: float = COURT_WIDTH_M,
        court_length: float = COURT_LENGTH_M,
    ) -> None:
        self.bandwidth = bandwidth
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.court_width = court_width
        self.court_length = court_length

        # Pre-compute evaluation grid (reused across calls)
        x_lin = np.linspace(0, court_width, grid_cols)
        y_lin = np.linspace(0, court_length, grid_rows)
        self._xx, self._yy = np.meshgrid(x_lin, y_lin)
        self._grid_pts = np.vstack([self._xx.ravel(), self._yy.ravel()])

    def generate(
        self, positions: np.ndarray, min_positions: int = 10,
        boundary_margin_m: float = 2.0
    ) -> Optional[np.ndarray]:
        """Compute a normalised 2D density grid.

        Args:
            positions:         Shape (N, 2) array of (x, y) court coordinates.
            min_positions:     Minimum number of points required (returns None if not met).
            boundary_margin_m: Accept positions this many metres outside the court
                               boundary (handles slight homography inaccuracy).

        Returns:
            Shape (grid_rows, grid_cols) float array in [0, 1], or ``None``.
        """
        if positions is None or len(positions) < min_positions:
            logger.warning(
                "Insufficient positions (%d < %d) for KDE.",
                len(positions) if positions is not None else 0,
                min_positions,
            )
            return None

        # Use all positions as-is — the accumulator already accepted them.
        # No additional boundary filtering here; positions may be slightly
        # outside the court (players standing behind the baseline etc.) and
        # that is perfectly valid for the heatmap.
        pts = positions

        try:
            bw = self.bandwidth if self.bandwidth else "scott"
            kde = gaussian_kde(pts.T, bw_method=bw)
            density = kde(self._grid_pts).reshape(self.grid_rows, self.grid_cols)
        except Exception as exc:
            logger.error("KDE computation failed: %s", exc)
            return None

        # Normalise to [0, 1]
        d_min, d_max = density.min(), density.max()
        if d_max - d_min < 1e-10:
            return np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        normalised = ((density - d_min) / (d_max - d_min)).astype(np.float32)
        logger.debug(
            "KDE generated: grid=%dx%d, pts=%d",
            self.grid_rows,
            self.grid_cols,
            len(pts),
        )
        return normalised
