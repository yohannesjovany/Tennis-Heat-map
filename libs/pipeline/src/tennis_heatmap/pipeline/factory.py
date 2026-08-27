"""
tennis_heatmap.pipeline.factory

Factory for constructing the full pipeline from a :class:`PipelineConfig`.

This is the single place where registry lookups happen. All other modules
depend only on interfaces — the factory wires the concrete implementations.
"""
from __future__ import annotations

import logging

from tennis_heatmap.core.config import PipelineConfig
from tennis_heatmap.core.registry import DetectorRegistry, TrackerRegistry, CourtDetectorRegistry
from tennis_heatmap.core.interfaces.detector import BaseDetector
from tennis_heatmap.core.interfaces.tracker import BaseTracker
from tennis_heatmap.core.interfaces.court_detector import BaseCourtDetector
from tennis_heatmap.court.homography import CourtHomography
from tennis_heatmap.heatmap.accumulator import CourtPositionAccumulator
from tennis_heatmap.heatmap.kde_generator import KDEHeatmapGenerator
from tennis_heatmap.heatmap.renderer import HeatmapRenderer
from tennis_heatmap.heatmap.exporter import HeatmapExporter

# Auto-register all implementations by importing their packages.
import tennis_heatmap.detectors   # noqa: F401
import tennis_heatmap.trackers    # noqa: F401
import tennis_heatmap.court       # noqa: F401

logger = logging.getLogger(__name__)


class PipelineComponents:
    """Bag of fully-constructed pipeline components ready for use."""

    def __init__(
        self,
        player_detector: BaseDetector,
        ball_detector: BaseDetector,
        player_tracker: BaseTracker,
        ball_tracker: BaseTracker,
        court_detector: BaseCourtDetector,
        homography: CourtHomography,
        accumulator: CourtPositionAccumulator,
        kde_generator: KDEHeatmapGenerator,
        renderer: HeatmapRenderer,
        exporter: HeatmapExporter,
        config: PipelineConfig,
    ) -> None:
        self.player_detector = player_detector
        self.ball_detector = ball_detector
        self.player_tracker = player_tracker
        self.ball_tracker = ball_tracker
        self.court_detector = court_detector
        self.homography = homography
        self.accumulator = accumulator
        self.kde_generator = kde_generator
        self.renderer = renderer
        self.exporter = exporter
        self.config = config


def build_pipeline(config: PipelineConfig) -> PipelineComponents:
    """Construct all pipeline components from a validated config.

    This is the canonical factory function. Call it once at startup.

    Args:
        config: Validated :class:`PipelineConfig`.

    Returns:
        Fully wired :class:`PipelineComponents`.

    Raises:
        KeyError: If a configured model name is not registered.
    """
    logger.info("Building pipeline from config...")

    # --- Detectors ---
    pd_cfg = config.player_detector
    player_detector = DetectorRegistry.build(
        pd_cfg.name,
        model_path=pd_cfg.model_path or "yolov8n.pt",
        confidence_threshold=pd_cfg.confidence_threshold,
        iou_threshold=pd_cfg.iou_threshold,
        device=pd_cfg.device,
        classes=pd_cfg.classes,
        **pd_cfg.extra,
    )

    bd_cfg = config.ball_detector
    ball_detector = DetectorRegistry.build(
        bd_cfg.name,
        model_path=bd_cfg.model_path or "yolov8n.pt",
        confidence_threshold=bd_cfg.confidence_threshold,
        iou_threshold=bd_cfg.iou_threshold,
        device=bd_cfg.device,
        classes=bd_cfg.classes,
        **bd_cfg.extra,
    )

    # --- Trackers ---
    pt_cfg = config.player_tracker
    player_tracker = TrackerRegistry.build(
        pt_cfg.name,
        track_thresh=pt_cfg.track_thresh,
        track_buffer=pt_cfg.track_buffer,
        match_thresh=pt_cfg.match_thresh,
        **pt_cfg.extra,
    )

    bt_cfg = config.ball_tracker
    ball_tracker = TrackerRegistry.build(
        bt_cfg.name,
        track_thresh=bt_cfg.track_thresh,
        track_buffer=bt_cfg.track_buffer,
        match_thresh=bt_cfg.match_thresh,
        **bt_cfg.extra,
    )

    # --- Court detector ---
    cd_cfg = config.court_detector
    court_detector = CourtDetectorRegistry.build(
        cd_cfg.name,
        num_keypoints=cd_cfg.num_keypoints,
        **cd_cfg.extra,
    )

    # --- Shared components ---
    homography = CourtHomography()

    hm_cfg = config.heatmap
    accumulator = CourtPositionAccumulator()
    kde_generator = KDEHeatmapGenerator(bandwidth=hm_cfg.kde_bandwidth)
    renderer = HeatmapRenderer(
        colormap=hm_cfg.colormap,
        alpha=hm_cfg.alpha,
        dpi=hm_cfg.dpi,
    )
    exporter = HeatmapExporter(
        output_dir=config.output_dir,
        formats=hm_cfg.output_formats,
        dpi=hm_cfg.dpi,
    )

    logger.info("Pipeline built successfully.")
    logger.info("  Player detector : %s", player_detector)
    logger.info("  Ball detector   : %s", ball_detector)
    logger.info("  Player tracker  : %s", player_tracker)
    logger.info("  Ball tracker    : %s", ball_tracker)
    logger.info("  Court detector  : %s", court_detector)

    return PipelineComponents(
        player_detector=player_detector,
        ball_detector=ball_detector,
        player_tracker=player_tracker,
        ball_tracker=ball_tracker,
        court_detector=court_detector,
        homography=homography,
        accumulator=accumulator,
        kde_generator=kde_generator,
        renderer=renderer,
        exporter=exporter,
        config=config,
    )
