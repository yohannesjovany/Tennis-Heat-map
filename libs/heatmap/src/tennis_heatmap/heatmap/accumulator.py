"""
tennis_heatmap.heatmap.accumulator

Collects 2D court coordinates per-entity and provides a clean interface
for the KDE generator and renderer.

Court-half strategy (default for tennis singles/doubles)
---------------------------------------------------------
Instead of relying on ByteTrack IDs (which fragment every time a player
leaves frame), positions are bucketed by which court half they fall in:

  y < COURT_LENGTH_M / 2  → Player 1  (near/bottom half)
  y >= COURT_LENGTH_M / 2 → Player 2  (far/top half)

This is robust to camera cuts, replays, and close-up shots.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from tennis_heatmap.core.models.court import CourtCoordinate, COURT_WIDTH_M, COURT_LENGTH_M

# Court net is at y = COURT_LENGTH_M / 2
_COURT_MIDLINE_M = COURT_LENGTH_M / 2.0


@dataclass
class EntityPositions:
    """Accumulated court-space positions for a single entity (player or ball).

    Attributes:
        entity_id:    Human-readable label, e.g. ``"player_1"``, ``"ball"``.
        positions:    List of (x, y) tuples in court metres.
        frame_indices: Corresponding frame indices.
    """
    entity_id: str
    positions: List[tuple[float, float]] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)

    def add(self, x: float, y: float, frame_index: int = 0) -> None:
        self.positions.append((x, y))
        self.frame_indices.append(frame_index)

    def as_array(self) -> np.ndarray:
        """Return positions as shape (N, 2) numpy array."""
        if not self.positions:
            return np.empty((0, 2), dtype=np.float64)
        return np.array(self.positions, dtype=np.float64)

    @property
    def count(self) -> int:
        return len(self.positions)

    def __repr__(self) -> str:
        return f"EntityPositions(id={self.entity_id!r}, n={self.count})"


class CourtPositionAccumulator:
    """Accumulates court-space positions, always producing exactly 2 player buckets.

    **Court-half strategy**: rather than using ByteTrack IDs (which create a
    new ID every time a player leaves and re-enters frame), every detected
    player position is routed to one of two buckets based on which side of
    the net they are on:

    - ``player_1``  — near half  (y < net line)
    - ``player_2``  — far half   (y >= net line)

    This guarantees exactly **2 player heatmaps** regardless of how many track
    IDs ByteTrack assigns, and is robust to broadcast camera cuts.

    The ball is accumulated separately as before.
    """

    PLAYER_1_KEY = "player_1"   # near half  (y < midline)
    PLAYER_2_KEY = "player_2"   # far half   (y >= midline)
    BALL_KEY = "ball"

    def __init__(self) -> None:
        self._player_1 = EntityPositions(entity_id=self.PLAYER_1_KEY)
        self._player_2 = EntityPositions(entity_id=self.PLAYER_2_KEY)
        self._ball = EntityPositions(entity_id=self.BALL_KEY)

    # ------------------------------------------------------------------
    # Add methods
    # ------------------------------------------------------------------

    def add_player_by_half(self, x: float, y: float, frame_index: int = 0) -> None:
        """Route a player position to Player 1 or Player 2 by court half."""
        if y < _COURT_MIDLINE_M:
            self._player_1.add(x, y, frame_index)
        else:
            self._player_2.add(x, y, frame_index)

    def add_player_coord(self, coord: CourtCoordinate) -> None:
        """Convenience wrapper for :class:`CourtCoordinate` — ignores track_id."""
        self.add_player_by_half(coord.x, coord.y, coord.frame_index)

    # Legacy track-ID based add (kept for backward compat, routes by half)
    def add_player(self, track_id: int, x: float, y: float, frame_index: int = 0) -> None:
        """Add player by track_id — routes to half bucket, ignoring track_id."""
        self.add_player_by_half(x, y, frame_index)

    def add_ball(self, x: float, y: float, frame_index: int = 0) -> None:
        """Record a ball position in court-space metres."""
        self._ball.add(x, y, frame_index)

    def add_ball_coord(self, coord: CourtCoordinate) -> None:
        self.add_ball(coord.x, coord.y, coord.frame_index)

    # ------------------------------------------------------------------
    # Accessor methods
    # ------------------------------------------------------------------

    def get_player_1(self) -> EntityPositions:
        """Near-half player (Player 1)."""
        return self._player_1

    def get_player_2(self) -> EntityPositions:
        """Far-half player (Player 2)."""
        return self._player_2

    def get_both_players(self) -> Dict[str, EntityPositions]:
        """Return both player buckets keyed by label."""
        return {
            self.PLAYER_1_KEY: self._player_1,
            self.PLAYER_2_KEY: self._player_2,
        }

    def get_ball(self) -> EntityPositions:
        return self._ball

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def total_player_positions(self) -> int:
        return self._player_1.count + self._player_2.count

    def total_ball_positions(self) -> int:
        return self._ball.count

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._player_1 = EntityPositions(entity_id=self.PLAYER_1_KEY)
        self._player_2 = EntityPositions(entity_id=self.PLAYER_2_KEY)
        self._ball = EntityPositions(entity_id=self.BALL_KEY)

    def summary(self) -> str:
        return (
            f"Players: 2 (court-half strategy)\n"
            f"  Player 1 (near half): {self._player_1.count} positions\n"
            f"  Player 2 (far half) : {self._player_2.count} positions\n"
            f"Ball: {self._ball.count} positions"
        )
