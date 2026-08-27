"""
Unit tests for court homography computation.
"""
import numpy as np
import pytest

from tennis_heatmap.court.homography import CourtHomography
from tennis_heatmap.court.court_template import get_court_reference_points, COURT_WIDTH_M, COURT_LENGTH_M
from tennis_heatmap.core.models.court import CourtKeypoints


def _make_synthetic_keypoints(noise_px: float = 0.0) -> CourtKeypoints:
    """Create a synthetic CourtKeypoints where the pixel coords are the
    reference world coords scaled to a 1280×720 canvas (with margin)."""
    ref = get_court_reference_points()  # (12, 2) in metres
    margin_x, margin_y = 50, 30
    w_px, h_px = 1280, 720

    usable_w = w_px - 2 * margin_x
    usable_h = h_px - 2 * margin_y

    px = margin_x + (ref[:, 0] / COURT_WIDTH_M) * usable_w
    py = margin_y + (ref[:, 1] / COURT_LENGTH_M) * usable_h

    if noise_px > 0:
        rng = np.random.default_rng(0)
        px += rng.normal(0, noise_px, px.shape)
        py += rng.normal(0, noise_px, py.shape)

    keypoints = np.stack([px, py], axis=1)
    return CourtKeypoints(keypoints=keypoints, frame_index=0)


def test_calibrate_succeeds_with_synthetic_keypoints():
    h = CourtHomography()
    kp = _make_synthetic_keypoints()
    ok = h.calibrate(kp)
    assert ok is True
    assert h.is_calibrated is True


def test_project_baseline_corners():
    """After perfect calibration, pixel corners should project near (0,0) etc."""
    h = CourtHomography()
    kp = _make_synthetic_keypoints()
    h.calibrate(kp)

    # The first keypoint in our synthetic set is the far-left pixel of the far baseline.
    # In court space, that should be ~ (0, 0).
    far_left_px = kp.keypoints[0]
    result = h.project_point(far_left_px[0], far_left_px[1])
    assert result is not None
    court_x, court_y = result
    assert abs(court_x) < 0.5  # within 0.5 m of x=0
    assert abs(court_y) < 0.5  # within 0.5 m of y=0


def test_project_returns_none_before_calibration():
    h = CourtHomography()
    assert h.project_point(640, 360) is None


def test_calibrate_fails_with_too_few_keypoints():
    h = CourtHomography()
    kp = CourtKeypoints(keypoints=np.array([[100, 200], [300, 400]]))
    ok = h.calibrate(kp)
    assert ok is False


def test_project_out_of_court_point():
    """A pixel far outside the court area should project out-of-court."""
    h = CourtHomography()
    kp = _make_synthetic_keypoints()
    h.calibrate(kp)
    coord = h.project_to_court_coord(0, 0, frame_index=0)
    # (0,0) pixel is outside the court — is_in_court should be False
    if coord is not None:
        assert not coord.is_in_court()
