"""
tennis_heatmap.pipeline.video_pipeline

Main video processing loop. Reads a video frame by frame, runs the full
detection → tracking → projection → accumulation pipeline, then generates
and exports heatmap images.

Common issues & fixes
---------------------
* Players: 0 / Ball: 0  →  homography is bad (< 8 inliers) OR confidence
  threshold too high. Lower player_detector.confidence_threshold to 0.25
  and let calibration run on more frames.
* Only ball heatmap produced  →  no player tracks — same root cause.
* Court image only, no overlay  →  insufficient positions (< min_positions).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from tennis_heatmap.core.config import PipelineConfig
from tennis_heatmap.core.models.detection import ObjectClass
from tennis_heatmap.heatmap.exporter import HeatmapResult
from tennis_heatmap.pipeline.factory import PipelineComponents, build_pipeline

logger = logging.getLogger(__name__)


@dataclass
class PipelineProgress:
    """Progress snapshot emitted during video processing."""
    current_frame: int
    total_frames: int
    elapsed_seconds: float
    player_positions_accumulated: int
    ball_positions_accumulated: int

    @property
    def fraction(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return self.current_frame / self.total_frames

    @property
    def fps(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.current_frame / self.elapsed_seconds


class TennisHeatmapPipeline:
    """End-to-end tennis heatmap generation pipeline.

    Args:
        config:     Validated :class:`PipelineConfig`.
        components: Pre-built components (if None, they are built from config).
    """

    def __init__(
        self,
        config: PipelineConfig,
        components: Optional[PipelineComponents] = None,
    ) -> None:
        self.config = config
        self.components = components or build_pipeline(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        video_path: str | Path,
        run_id: str = "match",
        progress_callback: Optional[Callable[[PipelineProgress], None]] = None,
    ) -> HeatmapResult:
        """Process a video file and return all heatmap results.

        Args:
            video_path:        Path to the input video file.
            run_id:            Identifier used for output filenames.
            progress_callback: Optional callable receiving :class:`PipelineProgress`
                               snapshots (useful for progress bars / websocket updates).

        Returns:
            :class:`HeatmapResult` containing player and ball heatmaps.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        comp = self.components
        cfg = self.config

        # Reset stateful components for this run
        comp.player_tracker.reset()
        comp.accumulator.reset()

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0

        max_frames = cfg.video.max_frames or total_frames
        skip = max(0, cfg.video.skip_frames)
        resize_w = cfg.video.resize_width

        logger.info(
            "Processing: %s | frames=%d | fps=%.1f | skip=%d",
            video_path.name,
            total_frames,
            fps_src,
            skip,
        )

        # --- Warmup detectors ---
        comp.player_detector.warmup()

        frame_idx = 0
        processed = 0
        calibrated = False
        best_inliers = 0
        calibration_attempts = 0
        # Keep trying to calibrate until we get a good H matrix (>= MIN_INLIERS).
        # Don't stop at the first successful attempt — a weak calibration
        # (e.g. 6/12 inliers on a close-up frame) produces a bad H matrix.
        MIN_GOOD_INLIERS = 8
        MAX_CALIBRATION_FRAMES = 60  # try up to 60 frames (~2.4s at 25fps)
        start_time = time.monotonic()

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame skipping
            if skip > 0 and frame_idx % (skip + 1) != 0:
                frame_idx += 1
                continue

            # Optional resize for speed
            if resize_w:
                h, w = frame.shape[:2]
                scale = resize_w / w
                frame = cv2.resize(frame, (resize_w, int(h * scale)))

            # --- Court calibration — keep improving until we have a good H ---
            # Strategy: try every frame up to MAX_CALIBRATION_FRAMES.
            # Accept a weak calibration only if we cannot do better.
            if not calibrated or (best_inliers < MIN_GOOD_INLIERS and calibration_attempts < MAX_CALIBRATION_FRAMES):
                keypoints = comp.court_detector.detect_court(frame, frame_idx)
                if keypoints.num_keypoints >= 4:
                    ok = comp.homography.calibrate(keypoints)
                    calibration_attempts += 1
                    if ok:
                        inliers = comp.homography._calibration_error  # use error as proxy
                        # Count inliers from keypoints
                        n_inliers = min(keypoints.num_keypoints, 12)
                        if n_inliers > best_inliers:
                            best_inliers = n_inliers
                            calibrated = True
                            logger.info(
                                "Court calibration updated on frame %d (keypoints=%d).",
                                frame_idx, n_inliers
                            )
                        if best_inliers >= MIN_GOOD_INLIERS:
                            logger.info(
                                "Court calibration is good (keypoints=%d >= %d). Locking in.",
                                best_inliers, MIN_GOOD_INLIERS
                            )

            # --- Player detection + tracking ---
            player_detections = comp.player_detector.detect(frame, frame_idx)
            player_tracks = comp.player_tracker.update(player_detections, frame_idx)

            # --- Coordinate projection + accumulation ---
            if calibrated:
                frame_player_added = 0
                for track in player_tracks:
                    # Skip unconfirmed / lost tracks (ByteTrack uses -1 / 0)
                    if track.track_id < 1:
                        continue
                    px, py = track.foot_point
                    coord = comp.homography.project_to_court_coord(
                        px, py, frame_index=frame_idx, track_id=track.track_id
                    )
                    # Court-half accumulation — routes to Player 1 or Player 2
                    # by which side of the net the coord falls on.
                    if coord is not None and coord.is_near_court(margin_m=2.0):
                        comp.accumulator.add_player_coord(coord)
                        frame_player_added += 1

                # Log a diagnostic every 100 frames to help spot issues early.
                if processed % 100 == 1:
                    logger.debug(
                        "Frame %d: player_detections=%d player_tracks=%d added=%d",
                        frame_idx,
                        len(player_detections), len(player_tracks), frame_player_added,
                    )

            processed += 1
            frame_idx += 1

            # Emit progress every 30 processed frames
            if progress_callback and processed % 30 == 0:
                elapsed = time.monotonic() - start_time
                progress_callback(
                    PipelineProgress(
                        current_frame=frame_idx,
                        total_frames=min(total_frames, max_frames),
                        elapsed_seconds=elapsed,
                        player_positions_accumulated=comp.accumulator.total_player_positions(),
                        ball_positions_accumulated=comp.accumulator.total_ball_positions(),
                    )
                )

        cap.release()
        elapsed = time.monotonic() - start_time
        logger.info(
            "Video processed: %d frames in %.1fs (%.1f fps). %s",
            processed,
            elapsed,
            processed / elapsed if elapsed > 0 else 0,
            comp.accumulator.summary(),
        )

        # --- Heatmap generation ---
        result = self._generate_heatmaps(run_id, video_path, total_frames, fps_src)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_heatmaps(
        self,
        run_id: str,
        video_path: Path,
        total_frames: int,
        source_fps: float,
    ) -> HeatmapResult:
        comp = self.components
        hm_cfg = self.config.heatmap
        player_heatmaps = {}

        # Always exactly 2 player heatmaps — near half and far half.
        player_labels = {
            "player_1": ("Player 1 — Near Half", "hot"),
            "player_2": ("Player 2 — Far Half",  "cool"),
        }

        for key, entity_pos in comp.accumulator.get_both_players().items():
            positions = entity_pos.as_array()
            title, cmap = player_labels[key]
            logger.info(
                "%s: %d positions accumulated.", title, entity_pos.count
            )
            # Use a temp renderer with per-player colormap
            from tennis_heatmap.heatmap.renderer import HeatmapRenderer
            renderer = HeatmapRenderer(
                colormap=cmap,
                alpha=hm_cfg.alpha,
                dpi=hm_cfg.dpi,
            )
            density = comp.kde_generator.generate(
                positions, min_positions=hm_cfg.min_positions
            )
            img = renderer.render(
                density,
                title=title,
                subtitle=f"{video_path.name} | {entity_pos.count} samples",
            )
            player_heatmaps[key] = img

        result = HeatmapResult(
            player_heatmaps=player_heatmaps,
            ball_heatmap=None,
            metadata={
                "video": str(video_path),
                "run_id": run_id,
                "total_frames": total_frames,
                "source_fps": source_fps,
                "player_tracks": list(player_heatmaps.keys()),
                "player_positions": comp.accumulator.total_player_positions(),
                "ball_positions": 0,
                "homography_calibrated": comp.homography.is_calibrated,
                "homography_error_m": comp.homography.calibration_error_m(),
                "player_detector": str(comp.player_detector),
                "ball_detector": "Disabled",
                "player_tracker": str(comp.player_tracker),
                "ball_tracker": "Disabled",
            },
        )

        # Export to disk
        saved = comp.exporter.export(result, run_id=run_id)
        logger.info("Exported %d files to %s", len(saved), self.config.output_dir)

        return result
