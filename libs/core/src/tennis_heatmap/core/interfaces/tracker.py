"""
tennis_heatmap.core.interfaces.tracker

Abstract base class for all multi-object tracker implementations.
A tracker consumes per-frame detections and maintains persistent track identities.
"""
from __future__ import annotations

import abc
from typing import List

from tennis_heatmap.core.models.detection import Detection
from tennis_heatmap.core.models.track import Track


class BaseTracker(abc.ABC):
    """Strategy interface for multi-object trackers.

    Implementors must override :meth:`update`. The tracker is stateful —
    it accumulates state across successive :meth:`update` calls.
    Call :meth:`reset` to start a new sequence.

    All tracker implementations should be registered via::

        @TrackerRegistry.register("my_tracker")
        class MyTracker(BaseTracker):
            ...
    """

    @abc.abstractmethod
    def update(self, detections: List[Detection], frame_index: int = 0) -> List[Track]:
        """Associate new detections with existing tracks.

        Args:
            detections:  List of detections for the current frame.
            frame_index: 0-based index of the current frame.

        Returns:
            List of active :class:`~tennis_heatmap.core.models.track.Track`
            objects for the current frame. Only confirmed tracks are included
            by default (implementation-dependent).
        """
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset all internal tracker state.

        Must be called before processing a new video sequence to clear
        accumulated track histories and ID counters.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
