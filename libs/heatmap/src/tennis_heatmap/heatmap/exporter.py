"""
tennis_heatmap.heatmap.exporter

Saves heatmap render results to disk in the configured output formats.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional

from PIL import Image

logger = logging.getLogger(__name__)

OutputFormat = Literal["png", "pdf", "json"]


@dataclass
class HeatmapResult:
    """Container for all heatmap outputs from a single pipeline run.

    Attributes:
        player_heatmaps: Dict mapping track_id → PIL Image.
        ball_heatmap:    PIL Image for the ball density.
        metadata:        JSON-serialisable dict of run metadata.
    """
    player_heatmaps: Dict[int, Image.Image] = field(default_factory=dict)
    ball_heatmap: Optional[Image.Image] = None
    metadata: Dict = field(default_factory=dict)


class HeatmapExporter:
    """Saves :class:`HeatmapResult` images and metadata to an output directory.

    Args:
        output_dir:    Path to the output directory (created if missing).
        formats:       List of output formats. Supported: ``"png"``, ``"pdf"``.
        dpi:           DPI for PDF/PNG output.
    """

    def __init__(
        self,
        output_dir: str | Path = "./output",
        formats: List[OutputFormat] = ("png",),
        dpi: int = 150,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.formats = list(formats)
        self.dpi = dpi

    def export(self, result: HeatmapResult, run_id: str = "run") -> List[Path]:
        """Save all heatmap images and metadata JSON.

        Args:
            result: Completed :class:`HeatmapResult`.
            run_id: Prefix used for all output filenames.

        Returns:
            List of paths to all created files.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Clean stale files from previous runs so only this run's output remains.
        for old_file in self.output_dir.iterdir():
            if old_file.is_file():
                old_file.unlink()
                logger.debug("Removed stale output: %s", old_file.name)

        created: List[Path] = []

        # Player heatmaps — key is "player_1" / "player_2" (or legacy int track IDs)
        for key, img in result.player_heatmaps.items():
            stem = f"{run_id}_{key}"
            created.extend(self._save_image(img, stem))

        # Ball heatmap
        if result.ball_heatmap is not None:
            stem = f"{run_id}_ball"
            created.extend(self._save_image(result.ball_heatmap, stem))

        # Metadata JSON
        if "json" in self.formats:
            meta_path = self.output_dir / f"{run_id}_metadata.json"
            # Convert non-serialisable values to strings
            safe_meta = {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v
                         for k, v in result.metadata.items()}
            meta_path.write_text(json.dumps(safe_meta, indent=2))
            created.append(meta_path)
            logger.info("Saved metadata → %s", meta_path)

        return created

    def _save_image(self, img: Image.Image, stem: str) -> List[Path]:
        """Save a PIL Image in all configured formats."""
        paths = []
        if "png" in self.formats:
            p = self.output_dir / f"{stem}.png"
            img.save(p, format="PNG", dpi=(self.dpi, self.dpi))
            logger.info("Saved PNG → %s", p)
            paths.append(p)

        if "pdf" in self.formats:
            p = self.output_dir / f"{stem}.pdf"
            # PDF requires RGB (no alpha)
            rgb = img.convert("RGB")
            rgb.save(p, format="PDF", resolution=self.dpi)
            logger.info("Saved PDF → %s", p)
            paths.append(p)

        return paths
