"""
tennis_heatmap.detectors.detectron2

Stub detector implementations backed by Meta's Detectron2 (Apache 2.0).
Registered under ``"detectron2_player"`` in the global DetectorRegistry.

This module is a ready-to-implement stub. The interface is wired up and the
detector is registered; replace the body of :meth:`detect` with real
Detectron2 inference when needed.

License notice
--------------
Detectron2 is licensed under **Apache 2.0** — safe for commercial use
without additional licensing requirements.
See https://github.com/facebookresearch/detectron2/blob/main/LICENSE
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from tennis_heatmap.core.interfaces.detector import BaseDetector
from tennis_heatmap.core.models.detection import BoundingBox, Detection, ObjectClass
from tennis_heatmap.core.registry import DetectorRegistry

logger = logging.getLogger(__name__)


@DetectorRegistry.register("detectron2_player")
class Detectron2PlayerDetector(BaseDetector):
    """Player detector backed by Detectron2 (Apache 2.0).

    **Status: Stub — not yet implemented.**

    To implement:
    1. ``pip install detectron2`` (see https://detectron2.readthedocs.io/tutorials/install.html)
    2. Replace the ``detect()`` body with Detectron2 inference.
    3. Return a list of ``Detection`` objects — the rest of the pipeline
       is unaffected by this swap.

    Args:
        model_config: Path to a Detectron2 model config YAML.
        model_weights: URL or path to Detectron2 model weights.
        confidence_threshold: Detection confidence threshold.
        device: ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model_config: str = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        model_weights: str = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        self.model_config = model_config
        self.model_weights = model_weights
        self.confidence_threshold = confidence_threshold
        self.device = device
        logger.warning(
            "Detectron2PlayerDetector is a STUB. "
            "Implement detect() before using in production."
        )

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> List[Detection]:
        """Not yet implemented — returns empty list."""
        # TODO: implement Detectron2 inference
        # Example skeleton:
        #   from detectron2.engine import DefaultPredictor
        #   from detectron2.config import get_cfg
        #   cfg = get_cfg()
        #   cfg.merge_from_file(self.model_config)
        #   cfg.MODEL.WEIGHTS = self.model_weights
        #   predictor = DefaultPredictor(cfg)
        #   outputs = predictor(frame)
        #   instances = outputs["instances"].to("cpu")
        #   for i in range(len(instances)):
        #       box = instances.pred_boxes[i].tensor.numpy()[0]
        #       conf = float(instances.scores[i].numpy())
        #       if conf >= self.confidence_threshold:
        #           detections.append(Detection(...))
        logger.debug("Detectron2PlayerDetector stub called — returning no detections.")
        return []
