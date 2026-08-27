"""
tennis_heatmap.trackers.deepsort

DeepSORT tracker stub.

License notice
--------------
The original DeepSORT (nwojke/deep_sort) is **GPL-3.0** — incompatible with
closed-source commercial use. The community port ``deep-sort-realtime``
is also GPL-3.0. For commercial use, re-implement the Kalman+Hungarian
association logic from scratch (MIT/Apache), or use ByteTrack instead.

This stub is registered under ``"deepsort"`` so configs referencing it
produce a clear error rather than a silent failure.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from tennis_heatmap.core.interfaces.tracker import BaseTracker
from tennis_heatmap.core.models.detection import Detection
from tennis_heatmap.core.models.track import Track
from tennis_heatmap.core.registry import TrackerRegistry

logger = logging.getLogger(__name__)


@TrackerRegistry.register("deepsort")
class DeepSORTTracker(BaseTracker):
    """DeepSORT tracker stub (GPL-3.0 — NOT for commercial use without licensing).

    **Status: Stub — not implemented.**

    ⚠ License Warning: DeepSORT is GPL-3.0. For commercial deployment,
    use ``"bytetrack"`` (Apache 2.0) or obtain a commercial re-implementation.

    To implement this stub:
    1. Verify your licensing obligations.
    2. ``pip install deep-sort-realtime``
    3. Replace the ``update()`` body with DeepSORT inference.
    """

    def __init__(self, **kwargs) -> None:
        logger.warning(
            "DeepSORTTracker is a STUB and NOT implemented. "
            "⚠ DeepSORT is GPL-3.0 — verify licensing before commercial use. "
            "Consider using ByteTrack ('bytetrack') instead."
        )

    def update(self, detections: List[Detection], frame_index: int = 0) -> List[Track]:
        raise NotImplementedError(
            "DeepSORTTracker is not implemented. Use 'bytetrack' instead."
        )

    def reset(self) -> None:
        pass
