"""
tennis_heatmap.trackers.bytetrack

ByteTrack multi-object tracker implementation backed by the
``trackers`` library (Roboflow, Apache 2.0).

License
-------
``trackers``:   Apache 2.0 — ✅ safe for commercial use
``supervision``: MIT — ✅ safe for commercial use

Registered under key ``"bytetrack"`` in :data:`TrackerRegistry`.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from tennis_heatmap.core.interfaces.tracker import BaseTracker
from tennis_heatmap.core.models.detection import BoundingBox, Detection, ObjectClass
from tennis_heatmap.core.models.track import Track
from tennis_heatmap.core.registry import TrackerRegistry

logger = logging.getLogger(__name__)


def _import_trackers():
    """Lazy import of the trackers library."""
    try:
        from trackers import ByteTrackTracker
        return ByteTrackTracker
    except ImportError as exc:
        raise ImportError(
            "The 'trackers' package is required for ByteTrack. "
            "Install it with: pip install trackers supervision"
        ) from exc


def _import_supervision():
    """Lazy import of supervision for Detections conversion."""
    try:
        import supervision as sv
        return sv
    except ImportError as exc:
        raise ImportError(
            "The 'supervision' package is required for ByteTrack. "
            "Install it with: pip install supervision"
        ) from exc


@TrackerRegistry.register("bytetrack")
class ByteTrackPlayerTracker(BaseTracker):
    """ByteTrack multi-object tracker (Apache 2.0).

    Wraps the Roboflow ``trackers.ByteTrackTracker``, converting between
    the pipeline's ``Detection``/``Track`` dataclasses and the
    ``supervision.Detections`` format expected by the library.

    Args:
        track_thresh:  High-confidence detection threshold for track initiation.
        track_buffer:  Number of frames to keep a lost track alive before
                       dropping it. Use a small value (≤5) for ball tracking.
        match_thresh:  IoU threshold for detection–track association.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        **kwargs,
    ) -> None:
        ByteTrackTrackerCls = _import_trackers()
        self._tracker = ByteTrackTrackerCls()
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        logger.info(
            "ByteTrackPlayerTracker init: track_thresh=%.2f, buffer=%d, match=%.2f",
            track_thresh,
            track_buffer,
            match_thresh,
        )

    def update(self, detections: List[Detection], frame_index: int = 0) -> List[Track]:
        """Associate detections with existing tracks.

        Args:
            detections:  Per-frame detections from the detector.
            frame_index: Current frame index.

        Returns:
            List of active confirmed :class:`Track` objects.
        """
        sv = _import_supervision()

        if not detections:
            # Feed empty detections to update lost-track counters.
            empty = sv.Detections.empty()
            tracked_sv = self._tracker.update(empty)
            return []

        # Convert pipeline Detections → supervision.Detections
        xyxy = np.array([d.bbox.as_xyxy() for d in detections], dtype=np.float32)
        confs = np.array([d.confidence for d in detections], dtype=np.float32)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
        )

        # Run ByteTrack association
        tracked_sv = self._tracker.update(sv_detections)

        # Convert supervision tracked Detections → pipeline Tracks
        tracks: List[Track] = []
        if tracked_sv.tracker_id is None:
            return tracks

        # Determine object class from the original detections (first match)
        # We assume all detections in one call are the same class.
        obj_class = detections[0].object_class if detections else ObjectClass.UNKNOWN

        for i, tid in enumerate(tracked_sv.tracker_id):
            if tid is None:
                continue
            xyxy_i = tracked_sv.xyxy[i]
            conf_i = float(tracked_sv.confidence[i]) if tracked_sv.confidence is not None else 0.5

            tracks.append(
                Track(
                    track_id=int(tid),
                    bbox=BoundingBox.from_xyxy(*xyxy_i),
                    object_class=obj_class,
                    confidence=conf_i,
                    frame_index=frame_index,
                    is_confirmed=True,
                )
            )

        return tracks

    def reset(self) -> None:
        """Reset ByteTrack state for a new video sequence."""
        ByteTrackTrackerCls = _import_trackers()
        self._tracker = ByteTrackTrackerCls()
        logger.debug("ByteTrackPlayerTracker reset.")

    def __repr__(self) -> str:
        return (
            f"ByteTrackPlayerTracker(thresh={self.track_thresh}, "
            f"buffer={self.track_buffer})"
        )
