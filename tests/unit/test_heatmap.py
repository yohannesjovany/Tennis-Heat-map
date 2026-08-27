"""
Unit tests for heatmap KDE generation.
No GPU required — pure numpy/scipy.
"""
import numpy as np
import pytest

from tennis_heatmap.heatmap.kde_generator import KDEHeatmapGenerator
from tennis_heatmap.heatmap.accumulator import CourtPositionAccumulator


def test_kde_returns_none_on_insufficient_data():
    gen = KDEHeatmapGenerator()
    # Empty positions
    result = gen.generate(np.empty((0, 2)), min_positions=10)
    assert result is None


def test_kde_returns_grid_with_sufficient_data():
    gen = KDEHeatmapGenerator(grid_cols=50, grid_rows=100)
    rng = np.random.default_rng(42)
    positions = rng.uniform([0, 0], [10.97, 23.77], size=(100, 2))
    result = gen.generate(positions, min_positions=10)
    assert result is not None
    assert result.shape == (100, 50)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_kde_uses_all_positions_without_filtering():
    """KDE should use all positions without filtering — accumulator handles bounds."""
    gen = KDEHeatmapGenerator(grid_cols=50, grid_rows=100)
    # Positions spread across both court halves (some slightly outside)
    rng = np.random.default_rng(42)
    positions = rng.uniform([-1, -1], [12.0, 25.0], size=(100, 2))
    result = gen.generate(positions, min_positions=10)
    assert result is not None
    assert result.shape == (100, 50)


def test_accumulator_player_and_ball():
    acc = CourtPositionAccumulator()
    # y < 11.885 (midline) → Player 1 (near half)
    acc.add_player(1, 3.0, 5.0, frame_index=0)
    acc.add_player(1, 3.1, 5.1, frame_index=1)
    # y >= 11.885 → Player 2 (far half)
    acc.add_player(2, 7.0, 18.0, frame_index=0)
    acc.add_ball(5.5, 11.9, frame_index=0)

    assert acc.total_player_positions() == 3  # 2 near + 1 far
    assert acc.total_ball_positions() == 1

    p1 = acc.get_player_1()
    assert p1 is not None
    assert p1.count == 2   # the two near-half positions

    p2 = acc.get_player_2()
    assert p2 is not None
    assert p2.count == 1   # the one far-half position

    both = acc.get_both_players()
    assert set(both.keys()) == {"player_1", "player_2"}

    ball = acc.get_ball()
    assert ball.count == 1
    arr = ball.as_array()
    assert arr.shape == (1, 2)


def test_accumulator_reset():
    acc = CourtPositionAccumulator()
    acc.add_player(1, 1.0, 1.0)
    acc.add_ball(5.0, 12.0)
    acc.reset()
    assert acc.total_player_positions() == 0
    assert acc.total_ball_positions() == 0
