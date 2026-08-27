"""
tennis_heatmap.heatmap.renderer

Renders a KDE density grid as a colour-coded heatmap overlaid on a
top-down tennis court diagram using Matplotlib.
"""
from __future__ import annotations

import io
import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in headless / server environments
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap
from PIL import Image

from tennis_heatmap.court.court_template import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    SERVICE_BOX_LENGTH_M,
    SINGLES_OFFSET_M,
)

logger = logging.getLogger(__name__)


class HeatmapRenderer:
    """Renders heatmap density grids on a tennis court diagram.

    Args:
        colormap:    Matplotlib colourmap name (e.g. ``"hot"``, ``"plasma"``).
        alpha:       Heatmap overlay opacity (0–1).
        dpi:         Output image DPI.
        court_color: Court surface fill colour (CSS name or hex).
        line_color:  Court line colour.
        line_width:  Court line width in points.
    """

    def __init__(
        self,
        colormap: str = "hot",
        alpha: float = 0.75,
        dpi: int = 150,
        court_color: str = "#2c7bb6",     # hard court blue
        line_color: str = "white",
        line_width: float = 1.5,
    ) -> None:
        self.colormap = colormap
        self.alpha = alpha
        self.dpi = dpi
        self.court_color = court_color
        self.line_color = line_color
        self.line_width = line_width

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        density_grid: Optional[np.ndarray],
        title: str = "Heatmap",
        subtitle: str = "",
    ) -> Image.Image:
        """Render a density grid as a PIL Image.

        Args:
            density_grid: Shape (rows, cols) float array in [0, 1], or None.
            title:        Title text drawn above the court.
            subtitle:     Secondary label (e.g. "Player 1 | Set 2").

        Returns:
            A PIL Image (RGBA) ready for saving or further compositing.
        """
        fig, ax = self._setup_figure()
        self._draw_court(ax)

        if density_grid is not None:
            self._draw_heatmap(ax, density_grid)
        else:
            ax.text(
                COURT_WIDTH_M / 2,
                COURT_LENGTH_M / 2,
                "Insufficient data",
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
                style="italic",
            )

        self._apply_labels(ax, title, subtitle)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _setup_figure(self):
        aspect = COURT_LENGTH_M / COURT_WIDTH_M
        fig_w = 6.0
        fig_h = fig_w * aspect + 1.2  # extra vertical space for title
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#1a1a2e")
        ax.set_facecolor(self.court_color)
        ax.set_xlim(-0.5, COURT_WIDTH_M + 0.5)
        ax.set_ylim(-0.5, COURT_LENGTH_M + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig, ax

    def _draw_court(self, ax) -> None:
        """Draw the top-down tennis court diagram."""
        W = COURT_WIDTH_M
        L = COURT_LENGTH_M
        sw = SINGLES_OFFSET_M
        SL = SERVICE_BOX_LENGTH_M
        cx = W / 2
        lw = self.line_width
        lc = self.line_color

        def line(x1, y1, x2, y2):
            ax.plot([x1, x2], [y1, y2], color=lc, linewidth=lw, solid_capstyle="round")

        # Outer court (doubles)
        ax.add_patch(mpatches.Rectangle(
            (0, 0), W, L, fill=False, edgecolor=lc, linewidth=lw + 0.5
        ))

        # Singles sidelines
        line(sw, 0, sw, L)
        line(W - sw, 0, W - sw, L)

        # Far service line
        line(sw, SL, W - sw, SL)

        # Near service line
        line(sw, L - SL, W - sw, L - SL)

        # Centre service line (between service boxes)
        line(cx, SL, cx, L - SL)

        # Net
        ax.plot([0, W], [L / 2, L / 2], color=lc, linewidth=lw * 2,
                linestyle="--", alpha=0.9, label="Net")

        # Centre marks on baselines
        mark_len = 0.15
        line(cx - mark_len, 0, cx + mark_len, 0)
        line(cx - mark_len, L, cx + mark_len, L)

    def _draw_heatmap(self, ax, density_grid: np.ndarray) -> None:
        """Overlay the density grid on the court."""
        ax.imshow(
            density_grid,
            extent=[0, COURT_WIDTH_M, COURT_LENGTH_M, 0],
            origin="upper",
            cmap=self.colormap,
            alpha=self.alpha,
            aspect="auto",
            interpolation="bilinear",
        )

    def _apply_labels(self, ax, title: str, subtitle: str) -> None:
        """Add title, subtitle, and direction labels."""
        fig = ax.get_figure()

        # Title
        fig.text(
            0.5, 0.97, title,
            ha="center", va="top",
            fontsize=14, fontweight="bold",
            color="white",
            fontfamily="DejaVu Sans",
        )
        if subtitle:
            fig.text(
                0.5, 0.94, subtitle,
                ha="center", va="top",
                fontsize=9, color="#aaaaaa",
            )

        # Direction labels
        ax.text(
            COURT_WIDTH_M / 2, -0.35, "← Far Baseline →",
            ha="center", va="top", fontsize=7, color="#cccccc",
        )
        ax.text(
            COURT_WIDTH_M / 2, COURT_LENGTH_M + 0.35, "← Near Baseline →",
            ha="center", va="bottom", fontsize=7, color="#cccccc",
        )
