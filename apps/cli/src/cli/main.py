"""
Tennis Heatmap CLI

Usage examples:

  # Basic: process a match video with default config
  tennis-heatmap run --video match.mp4

  # Custom config (swap detector/tracker without code changes)
  tennis-heatmap run --video match.mp4 --config configs/bytetrack_yolov8.yaml

  # Set output directory
  tennis-heatmap run --video match.mp4 --output ./results/

  # Show available registered models
  tennis-heatmap list-models

  # Override specific config values inline
  tennis-heatmap run --video match.mp4 --device cuda --skip-frames 1
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from rich.table import Table

console = Console()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.version_option("0.1.0", prog_name="tennis-heatmap")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """🎾 Tennis Heatmap — AI-powered player and ball heatmap generator."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@cli.command()
@click.option("--video", "-i", required=True, type=click.Path(exists=True), help="Input video file.")
@click.option("--config", "-c", type=click.Path(exists=True), default=None, help="YAML config file.")
@click.option("--output", "-o", type=click.Path(), default="./output", show_default=True, help="Output directory.")
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps"]), default=None, help="Override inference device.")
@click.option("--skip-frames", type=int, default=None, help="Process every N+1-th frame (0=all).")
@click.option("--run-id", type=str, default=None, help="Identifier prefix for output files.")
@click.pass_context
def run(
    ctx: click.Context,
    video: str,
    config: Optional[str],
    output: str,
    device: Optional[str],
    skip_frames: Optional[int],
    run_id: Optional[str],
) -> None:
    """Process a tennis match video and generate heatmaps."""
    from tennis_heatmap.core.config import load_config, default_config
    from tennis_heatmap.pipeline.video_pipeline import TennisHeatmapPipeline, PipelineProgress

    # Load config
    if config:
        console.print(f"[cyan]Config:[/cyan] {config}")
        cfg = load_config(config)
    else:
        console.print("[cyan]Config:[/cyan] defaults")
        cfg = default_config()

    # Apply CLI overrides
    cfg.output_dir = output
    if device:
        cfg.player_detector.device = device
        cfg.ball_detector.device = device
    if skip_frames is not None:
        cfg.video.skip_frames = skip_frames

    video_path = Path(video)
    _run_id = run_id or video_path.stem

    console.rule("[bold green]Tennis Heatmap Pipeline[/bold green]")
    console.print(f"  Video  : [bold]{video_path.name}[/bold]")
    console.print(f"  Output : [bold]{output}[/bold]")
    console.print(f"  Run ID : [bold]{_run_id}[/bold]")
    console.print()

    pipeline = TennisHeatmapPipeline(config=cfg)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing frames…", total=100)

        def on_progress(p: PipelineProgress) -> None:
            pct = int(p.fraction * 100)
            progress.update(
                task,
                completed=pct,
                description=(
                    f"Frame {p.current_frame}/{p.total_frames} | "
                    f"{p.fps:.0f} fps | "
                    f"Players: {p.player_positions_accumulated} pos | "
                    f"Ball: {p.ball_positions_accumulated} pos"
                ),
            )

        try:
            result = pipeline.run(video_path, run_id=_run_id, progress_callback=on_progress)
            progress.update(task, completed=100)
        except Exception as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            if ctx.obj.get("verbose"):
                console.print_exception()
            sys.exit(1)

    console.rule("[bold green]Done[/bold green]")
    p1 = result.player_heatmaps.get("player_1")
    p2 = result.player_heatmaps.get("player_2")
    console.print(f"\n[bold]Player 1 (near half):[/bold] {'✅ heatmap generated' if p1 else '⚠️  insufficient data'}")
    console.print(f"[bold]Player 2 (far half): [/bold] {'✅ heatmap generated' if p2 else '⚠️  insufficient data'}")
    console.print(f"[bold]Ball heatmap:[/bold]        {'✅' if result.ball_heatmap else '⚠️  insufficient data'}")
    console.print(f"\n[bold green]Results saved to:[/bold green] {output}\n")
    console.print(f"  📄 {output}/{Path(video).stem}_player_1.png  ← Player 1")
    console.print(f"  📄 {output}/{Path(video).stem}_player_2.png  ← Player 2")
    console.print(f"  📄 {output}/{Path(video).stem}_ball.png      ← Ball\n")


@cli.command("watch")
@click.option("--video", "-i", required=True, type=click.Path(exists=True), help="Input video file.")
@click.option("--config", "-c", type=click.Path(exists=True), default=None, help="YAML config file.")
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default=None, help="Override compute device.")
@click.option("--width", type=int, default=1280, show_default=True, help="Display window width in pixels.")
@click.option("--skip-frames", type=int, default=None, help="Skip N frames between processed frames.")
def watch(video: str, config: Optional[str], device: Optional[str], width: int, skip_frames: Optional[int]) -> None:
    """Play a video with live bounding boxes on detected players.

    Shows Player 1 (near half, orange) and Player 2 (far half, blue)
    with real-time detection overlays, court calibration status, and FPS.

    \b
    Controls:
      SPACE     Pause / resume
      Q / ESC   Quit
      S         Save current frame as PNG screenshot
      + / -     Increase / decrease playback speed
    """
    from tennis_heatmap.core.config import PipelineConfig, load_config, default_config
    from tennis_heatmap.pipeline.video_viewer import run_viewer

    # Load config
    if config:
        cfg = load_config(config)
        console.print(f"Config: [cyan]{config}[/cyan]")
    else:
        cfg = default_config()
        console.print("Config: defaults")

    # Apply CLI overrides
    if device:
        cfg.player_detector.device = device
        cfg.ball_detector.device = device
    if skip_frames is not None:
        cfg.video.skip_frames = skip_frames

    console.rule("[bold cyan]Tennis Heatmap — Live View[/bold cyan]")
    console.print(f"  Video  : {video}")
    console.print(f"  Device : {cfg.player_detector.device}")
    console.print(f"  Width  : {width}px")
    console.print()
    console.print("[dim]Controls: SPACE=pause  Q=quit  S=screenshot  +/-=speed[/dim]")
    console.print()

    try:
        run_viewer(config=cfg, video_path=video, window_width=width)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}")
        logger.exception("Viewer error")
        sys.exit(1)

    console.print("[green]Viewer closed.[/green]")

@cli.command("list-models")
def list_models() -> None:
    """List all registered detectors, trackers, and court detectors."""
    from tennis_heatmap.core.registry import DetectorRegistry, TrackerRegistry, CourtDetectorRegistry
    # Trigger auto-registration
    import tennis_heatmap.detectors   # noqa: F401
    import tennis_heatmap.trackers    # noqa: F401
    import tennis_heatmap.court       # noqa: F401

    def _table(title: str, registry) -> Table:
        t = Table(title=title, show_header=True, header_style="bold magenta")
        t.add_column("Key", style="cyan", no_wrap=True)
        for key in registry.list_available():
            cls = registry._registry[key]
            t.add_row(key, cls.__name__, cls.__module__)
        return t

    console.print()
    console.print(_table("🔍 Detectors", DetectorRegistry))
    console.print()
    console.print(_table("🎯 Trackers", TrackerRegistry))
    console.print()
    console.print(_table("🏟️  Court Detectors", CourtDetectorRegistry))
    console.print()


@cli.command("diagnose")
@click.option("--video", "-i", required=True, type=click.Path(exists=True), help="Input video file.")
@click.option("--frames", type=int, default=10, show_default=True, help="Number of evenly-spaced frames to sample.")
@click.option("--player-conf", type=float, default=0.3, show_default=True, help="Player confidence threshold.")
@click.option("--save-frames", is_flag=True, default=False, help="Save annotated sample frames as PNG files.")
def diagnose(video: str, frames: int, player_conf: float, save_frames: bool) -> None:
    """Sample frames from a video and show exactly what the detector sees.

    Use this when you get \"Players: 0 / Ball: 0\" to understand WHY.
    It shows per-frame detection counts, confidence scores, and whether
    the homography projects detections inside or outside the court.
    """
    import cv2
    import numpy as np
    from tennis_heatmap.core.config import default_config
    from tennis_heatmap.pipeline.factory import build_pipeline

    cfg = default_config()
    cfg.player_detector.confidence_threshold = player_conf
    comp = build_pipeline(cfg)

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0

    console.rule("[bold cyan]Diagnose: Detection Check[/bold cyan]")
    console.print(f"  Video       : {video}")
    console.print(f"  Total frames: {total}  |  FPS: {fps_src:.1f}")
    console.print(f"  Sampling    : {frames} evenly-spaced frames")
    console.print(f"  Player conf : {player_conf}")
    console.print()

    sample_indices = [int(total * i / frames) for i in range(frames)]
    calibrated = False
    results = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize for consistency
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h * scale)))

        # Court calibration
        kp = comp.court_detector.detect_court(frame, idx)
        cal_status = f"keypoints={kp.num_keypoints}"
        if kp.num_keypoints >= 4 and not calibrated:
            ok = comp.homography.calibrate(kp)
            if ok:
                calibrated = True
                cal_status += " ✅ calibrated"
            else:
                cal_status += " ❌ calibration failed"

        # Player detection (raw — no tracking)
        player_dets = comp.player_detector.detect(frame, idx)
        in_court = 0
        if calibrated:
            for d in player_dets:
                bx, by = d.foot_point
                coord = comp.homography.project_to_court_coord(bx, by)
                if coord and coord.is_near_court(margin_m=2.0):
                    in_court += 1

        # Ball detection
        ball_dets = comp.ball_detector.detect(frame, idx)

        confs = [round(float(d.confidence), 2) for d in player_dets]
        results.append((idx, len(player_dets), in_court, len(ball_dets), cal_status, confs))

        if save_frames:
            # Draw boxes on frame and save
            annotated = frame.copy()
            for d in player_dets:
                x1, y1, x2, y2 = d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated, f"{d.confidence:.2f}", (int(x1), int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            out_path = f"diagnose_frame_{idx:05d}.png"
            cv2.imwrite(out_path, annotated)
            console.print(f"  Saved: [cyan]{out_path}[/cyan]")

    cap.release()

    # Print results table
    from rich.table import Table as RichTable
    t = RichTable(title="Detection Results per Sampled Frame", show_header=True)
    t.add_column("Frame#", style="dim")
    t.add_column("Players\nDetected", justify="right")
    t.add_column("Players\nIn-Court", justify="right", style="green")
    t.add_column("Ball\nDetected", justify="right")
    t.add_column("Court Status")
    t.add_column("Confidence Scores")

    for frame_n, n_players, n_in_court, n_ball, cal_st, confs in results:
        player_color = "green" if n_in_court > 0 else ("yellow" if n_players > 0 else "red")
        t.add_row(
            str(frame_n),
            f"[{player_color}]{n_players}[/{player_color}]",
            f"[{'green' if n_in_court > 0 else 'red'}]{n_in_court}[/{'green' if n_in_court > 0 else 'red'}]",
            str(n_ball),
            cal_st,
            str(confs) if confs else "—",
        )

    console.print(t)
    console.print()

    # Summary advice
    total_players = sum(r[1] for r in results)
    total_in_court = sum(r[2] for r in results)

    if total_players == 0:
        console.print("[bold red]❌ No players detected at all.[/bold red]")
        console.print("   → Try: [cyan]--player-conf 0.15[/cyan] to lower the threshold")
        console.print("   → The video may be close-up shots with no full-body players visible")
    elif total_in_court == 0:
        console.print("[bold yellow]⚠️  Players detected but NONE projected in-court.[/bold yellow]")
        console.print("   → Homography is bad — court lines not detected correctly")
        console.print("   → Make sure the video has clear overhead/wide court shots")
        console.print("   → Try a clip that starts with a full court view")
    else:
        console.print(f"[bold green]✅ {total_in_court}/{total_players} player detections projected in-court.[/bold green]")
        console.print("   The pipeline should produce heatmaps. Run: [cyan]tennis-heatmap run[/cyan]")
