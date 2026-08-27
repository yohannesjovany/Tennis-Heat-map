"""
Shared pytest fixtures and configuration.
"""
import pytest
import numpy as np


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A blank 720x1280 BGR frame for testing detectors/court code."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_positions() -> np.ndarray:
    """100 random court-space positions within the doubles court."""
    rng = np.random.default_rng(42)
    return rng.uniform([0, 0], [10.97, 23.77], size=(100, 2))
