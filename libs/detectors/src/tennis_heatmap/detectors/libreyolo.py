"""
tennis_heatmap.detectors.libreyolo

LibreYOLO-based detector implementation for players.
Registered under the key ``"libreyolo_player"`` in the global :data:`DetectorRegistry`.

License notice
--------------
LibreYOLO is MIT-licensed, making it fully permissive for commercial use.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from tennis_heatmap.core.interfaces.detector import BaseDetector
from tennis_heatmap.core.models.detection import BoundingBox, Detection, ObjectClass
from tennis_heatmap.core.registry import DetectorRegistry

logger = logging.getLogger(__name__)


def _load_libreyolo():
    """Lazy import of libreyolo."""
    try:
        from libreyolo import LibreYOLOX
        return LibreYOLOX
    except ImportError as exc:
        raise ImportError(
            "The 'libreyolo' package is required. "
            "Install it with: pip install libreyolo"
        ) from exc


@DetectorRegistry.register("libreyolo_player")
class LibreYoloPlayerDetector(BaseDetector):
    """Detects tennis players in a video frame using a LibreYOLO model.

    Args:
        model_path:           Path to weights or model name (e.g., ``"yolox_s.pt"``).
        confidence_threshold: Minimum detection confidence (0–1).
        iou_threshold:        NMS IoU threshold (0–1).
        device:               ``"cuda"``, ``"cpu"``, or ``"mps"``.
        classes:              COCO class IDs to keep. ``None`` → keep all. (Person is 0).
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolox_s.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        classes: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        LibreYOLOX = _load_libreyolo()
        self._model = LibreYOLOX(model_path)
        self._model.to(device)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        # Default to filtering persons only unless custom classes provided.
        self.classes = classes if classes is not None else [self.PERSON_CLASS_ID]
        logger.info("LibreYoloPlayerDetector loaded: model=%s, device=%s", model_path, device)

    def warmup(self, frame_size: tuple[int, int] = (720, 1280)) -> None:
        h, w = frame_size
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        self._model.predict(dummy, verbose=False)
        logger.debug("LibreYoloPlayerDetector warmup done.")

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> List[Detection]:
        # Note: LibreYOLO's inference method is `.predict()`
        results = self._model.predict(
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
            f"LibreYoloPlayerDetector(conf={self.confidence_threshold}, "
            f"device={self.device!r})"
        )
