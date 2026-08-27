"""
tennis_heatmap.detectors.yolov8

YOLOv8-based detector implementations for players and the ball.
Registered under the keys ``"yolov8_player"`` and ``"yolov8_ball"``
in the global :data:`DetectorRegistry`.

License notice
--------------
The ``ultralytics`` package is licensed under **GNU AGPL-3.0**.
For closed-source commercial deployment, an Ultralytics Enterprise License
is required. See https://ultralytics.com/license for details.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from tennis_heatmap.core.interfaces.detector import BaseDetector
from tennis_heatmap.core.models.detection import BoundingBox, Detection, ObjectClass
from tennis_heatmap.core.registry import DetectorRegistry

logger = logging.getLogger(__name__)


def _load_ultralytics():
    """Lazy import of ultralytics so the package is optional at import time."""
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError as exc:
        raise ImportError(
            "The 'ultralytics' package is required for YOLOv8 detectors. "
            "Install it with: pip install ultralytics\n"
            "⚠ Note: ultralytics is AGPL-3.0. A commercial license is required "
            "for closed-source / SaaS deployments."
        ) from exc


@DetectorRegistry.register("yolov8_player")
class YOLOv8PlayerDetector(BaseDetector):
    """Detects tennis players in a video frame using a YOLOv8 model.

    By default this filters COCO class 0 (``person``). To use a custom
    fine-tuned model pass the ``model_path`` pointing to your ``.pt`` file.

    Args:
        model_path:           Path to ``.pt`` weights (default: ``"yolov8n.pt"`` — auto-download).
        confidence_threshold: Minimum detection confidence (0–1).
        iou_threshold:        NMS IoU threshold (0–1).
        device:               ``"cuda"``, ``"cpu"``, or ``"mps"``.
        classes:              COCO class IDs to keep. ``None`` → keep all.
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        classes: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        YOLO = _load_ultralytics()
        self._model = YOLO(model_path)
        self._model.to(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        # Default to filtering persons only unless custom classes provided.
        self.classes = classes if classes is not None else [self.PERSON_CLASS_ID]
        logger.info("YOLOv8PlayerDetector loaded: model=%s, device=%s", model_path, device)

    def warmup(self, frame_size: tuple[int, int] = (720, 1280)) -> None:
        h, w = frame_size
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        self._model(dummy, verbose=False)
        logger.debug("YOLOv8PlayerDetector warmup done.")

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> List[Detection]:
        results = self._model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
        )
        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append(
                    Detection(
                        bbox=BoundingBox.from_xyxy(*xyxy),
                        confidence=conf,
                        object_class=ObjectClass.PLAYER,
                        frame_index=frame_index,
                    )
                )
        return detections

    def __repr__(self) -> str:
        return (
            f"YOLOv8PlayerDetector(conf={self.confidence_threshold}, "
            f"device={self.device!r})"
        )


@DetectorRegistry.register("yolov8_ball")
class YOLOv8BallDetector(BaseDetector):
    """Detects the tennis ball using a fine-tuned YOLOv8 model.

    Ball detection is challenging due to the ball's small size and high speed.
    For best results, supply a model fine-tuned on tennis ball data.

    Args:
        model_path:           Path to fine-tuned ``.pt`` weights.
        confidence_threshold: Lower threshold recommended (~0.25–0.35).
        iou_threshold:        NMS IoU threshold.
        device:               Inference device.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.3,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        YOLO = _load_ultralytics()
        self._model = YOLO(model_path)
        self._model.to(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        logger.info("YOLOv8BallDetector loaded: model=%s, device=%s", model_path, device)

    def warmup(self, frame_size: tuple[int, int] = (720, 1280)) -> None:
        h, w = frame_size
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        self._model(dummy, verbose=False)

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> List[Detection]:
        results = self._model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append(
                    Detection(
                        bbox=BoundingBox.from_xyxy(*xyxy),
                        confidence=conf,
                        object_class=ObjectClass.BALL,
                        frame_index=frame_index,
                    )
                )
        return detections

    def __repr__(self) -> str:
        return (
            f"YOLOv8BallDetector(conf={self.confidence_threshold}, "
            f"device={self.device!r})"
        )
