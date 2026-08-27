"""
tennis_heatmap.pipeline.video_viewer

Live video playback with annotated bounding boxes, track IDs, confidence
scores, and court-half player assignment overlaid on each frame.

Usage (via CLI):
    tennis-heatmap watch --video match.mp4

Controls:
    SPACE   — pause / resume
    Q / ESC — quit
    S       — save current frame as PNG
    + / -   — increase / decrease playback speed
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Fix for Wayland crash with opencv-python on Linux
if os.name == 'posix' and 'QT_QPA_PLATFORM' not in os.environ:
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
import numpy as np

from tennis_heatmap.core.config import PipelineConfig
from tennis_heatmap.core.models.detection import Detection
from tennis_heatmap.core.models.track import Track
from tennis_heatmap.court.court_template import COURT_LENGTH_M, COURT_WIDTH_M

logger = logging.getLogger(__name__)

# Court midline for player assignment
_MIDLINE_Y = COURT_LENGTH_M / 2.0

COLOR_P1 = (0, 200, 255)     # orange-yellow — Player 1 (near half)
COLOR_P2 = (255, 160, 0)     # cyan-blue    — Player 2 (far half)
COLOR_COURT = (255, 255, 0)  # cyan         — court keypoints
COLOR_TEXT_BG = (20, 20, 20) # dark overlay for text
COLOR_WHITE = (255, 255, 255)


def _draw_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a bounding box with a label tag."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps: float,
    calibrated: bool,
    p1_count: int,
    p2_count: int,
    paused: bool,
    speed: float,
) -> None:
    """Draw a heads-up display overlay."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), COLOR_TEXT_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Left: frame info
    left_text = f"Frame {frame_idx}/{total_frames}  |  {fps:.1f} FPS  |  Speed: {speed:.1f}x"
    if paused:
        left_text = "[PAUSED]  " + left_text
    cv2.putText(frame, left_text, (10, 20), font, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

    # Middle: player counts
    mid_text = f"P1: {p1_count}  P2: {p2_count}"
    cv2.putText(frame, mid_text, (10, 42), font, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

    # Right: calibration status
    cal_text = "Court: OK" if calibrated else "Court: UNCALIBRATED"
    cal_color = (0, 255, 0) if calibrated else (0, 0, 255)
    cv2.putText(frame, cal_text, (w - 200, 20), font, 0.5, cal_color, 1, cv2.LINE_AA)

    # Controls hint at bottom
    controls = "SPACE=pause  Q=quit  S=screenshot  +/-=speed"
    cv2.putText(frame, controls, (10, h - 10), font, 0.4, (120, 120, 120), 1, cv2.LINE_AA)


def _assign_player_color(
    track: Track,
    homography,
    calibrated: bool,
) -> Tuple[Tuple[int, int, int], str]:
    """Determine player color and label based on court-half assignment."""
    if not calibrated:
        return COLOR_P1, f"ID:{track.track_id}"

    px, py = track.foot_point
    coord = homography.project_point(px, py)
    if coord is None:
        return COLOR_P1, f"ID:{track.track_id}"

    court_y = coord[1]
    if court_y < _MIDLINE_Y:
        return COLOR_P1, f"P1 ({track.confidence:.0%})"
    else:
        return COLOR_P2, f"P2 ({track.confidence:.0%})"


def run_viewer(
    config: PipelineConfig,
    video_path: str | Path,
    window_width: int = 1280,
) -> None:
    """Play a video with live detection overlays.

    Args:
        config:       Pipeline configuration.
        video_path:   Path to the input video.
        window_width: Resize the display window to this width.
    """
    from tennis_heatmap.pipeline.factory import build_pipeline

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    comp = build_pipeline(config)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    window_name = f"Tennis Heatmap — {video_path.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, int(window_width * 9 / 16))

    frame_idx = 0
    paused = False
    speed = 1.0
    p1_total = 0
    p2_total = 0
    calibrated = False
    saved_count = 0
    prev_time = time.monotonic()

    logger.info("Viewer started: %s (%d frames @ %.1f fps)", video_path.name, total_frames, src_fps)
    logger.info("Controls: SPACE=pause, Q/ESC=quit, S=screenshot, +/-=speed")

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize for display and processing
            h, w = frame.shape[:2]
            if w > window_width:
                scale = window_width / w
                frame = cv2.resize(frame, (window_width, int(h * scale)))

            # --- Court calibration ---
            if not calibrated:
                kp = comp.court_detector.detect_court(frame, frame_idx)
                if kp.num_keypoints >= 4:
                    ok = comp.homography.calibrate(kp)
                    if ok:
                        calibrated = True

            # --- Player detection + tracking ---
            player_dets = comp.player_detector.detect(frame, frame_idx)
            player_tracks = comp.player_tracker.update(player_dets, frame_idx)

            # --- Draw court polygon ---
            if calibrated:
                corners = np.array([
                    [0, 0],
                    [COURT_WIDTH_M, 0],
                    [COURT_WIDTH_M, COURT_LENGTH_M],
                    [0, COURT_LENGTH_M]
                ])
                px_corners = comp.homography.unproject_points(corners)
                if px_corners is not None:
                    # Draw a glowing cyan polygon for the court
                    pts = px_corners.astype(np.int32)
                    cv2.polylines(frame, [pts], isClosed=True, color=COLOR_COURT, thickness=2, lineType=cv2.LINE_AA)

            # --- Draw player boxes ---
            for track in player_tracks:
                if track.track_id < 1:
                    continue
                color, label = _assign_player_color(track, comp.homography, calibrated)
                b = track.bbox
                _draw_box(frame, int(b.x1), int(b.y1), int(b.x2), int(b.y2), label, color)

                # Draw foot point
                fx, fy = track.foot_point
                cv2.circle(frame, (int(fx), int(fy)), 4, color, -1)

                # Count by half
                if calibrated:
                    coord = comp.homography.project_point(fx, fy)
                    if coord is not None:
                        if coord[1] < _MIDLINE_Y:
                            p1_total += 1
                        else:
                            p2_total += 1

            # --- HUD overlay ---
            now = time.monotonic()
            display_fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            _draw_hud(
                frame, frame_idx, total_frames, display_fps,
                calibrated, p1_total, p2_total,
                paused, speed,
            )

            frame_idx += 1

        # Show frame
        cv2.imshow(window_name, frame)

        # Key handling — wait scaled by playback speed
        wait_ms = max(1, int((1000 / src_fps) / speed))
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord(' '):             # SPACE — toggle pause
            paused = not paused
        elif key == ord('s'):             # S — save screenshot
            saved_count += 1
            out_name = f"screenshot_{frame_idx:05d}.png"
            cv2.imwrite(out_name, frame)
            logger.info("Saved screenshot: %s", out_name)
        elif key == ord('+') or key == ord('='):
            speed = min(speed + 0.5, 8.0)
        elif key == ord('-') or key == ord('_'):
            speed = max(speed - 0.5, 0.25)

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Viewer closed. Processed %d frames.", frame_idx)
