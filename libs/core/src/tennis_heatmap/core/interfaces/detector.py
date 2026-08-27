"""
tennis_heatmap.core.interfaces.detector

Abstract base class that every detector implementation must satisfy.
A detector processes a single video frame and returns a list of Detections.
"""
from __future__ import annotations

import abc
from typing import List

import numpy as np

from tennis_heatmap.core.models.detection import Detection


class BaseDetector(abc.ABC):
    """Strategy interface for object detectors.

    Implementors must override :meth:`detect`. Optionally override
    :meth:`warmup` for model-specific initialisation that benefits from
    being done once before the main loop.

    All detector implementations should be registered via::

        @DetectorRegistry.register("my_detector")
        class MyDetector(BaseDetector):
            ...
    """

    @abc.abstractmethod
    def detect(self, frame: np.ndarray, frame_index: int = 0) -> List[Detection]:
        """Run inference on a single BGR frame.

        Args:
            frame:       H×W×3 uint8 BGR numpy array (OpenCV format).
            frame_index: 0-based index of the frame in the video.

        Returns:
            List of :class:`~tennis_heatmap.core.models.detection.Detection`
            objects, one per detected object. Empty list if nothing is found.
        """
        ...

    def warmup(self, frame_size: tuple[int, int] = (720, 1280)) -> None:
        """Optional warm-up pass to initialise GPU kernels.

        The default implementation does nothing. Override in subclasses
        where the first inference call is significantly slower than subsequent
        ones (e.g. TensorRT, CoreML).

        Args:
            frame_size: (height, width) of the warm-up dummy frame.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
