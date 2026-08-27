# Tennis Heatmap — AI Match Analysis

AI-powered heatmap generation for tennis match videos. Processes raw match footage and produces colour-coded heatmaps showing **player movement patterns** and **ball positioning** on a top-down court diagram.

---

## Monorepo Structure

```
tennis-heatmap/
├── libs/
│   ├── core/        # Interfaces, data models, plugin registry
│   ├── detectors/   # YOLOv8 + stub implementations
│   ├── trackers/    # ByteTrack (default) + stub
│   ├── court/       # Court detection & homography
│   ├── heatmap/     # KDE generation, rendering, export
│   └── pipeline/    # Full orchestration
├── apps/
│   ├── cli/         # `tennis-heatmap` command
│   └── api/         # FastAPI REST API + Web UI
├── configs/         # YAML configuration presets
├── docker/          # Dockerfile + compose
└── tests/
```

---

## Quickstart

### 1. Install (requires [uv](https://docs.astral.sh/uv/))

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone & install all workspace packages
git clone <repo> tennis-heatmap && cd tennis-heatmap
make install
```

### 2. Run on a video (CLI)

```bash
# Basic — uses default config (YOLOv8 + ByteTrack, CPU)
uv run tennis-heatmap run --video match.mp4

# With GPU and custom config
uv run tennis-heatmap run --video match.mp4 \
    --config configs/bytetrack_yolov8.yaml \
    --output ./results/

# See all available models
uv run tennis-heatmap list-models
```

### 3. Start the API + Web UI

```bash
make api
# Then open http://localhost:8000
```

### 4. Docker

```bash
make docker-build
make docker-up
# API at http://localhost:8000
```

---

## Swapping Models

Change **one YAML field** to swap the detector or tracker — no Python changes needed:

```yaml
# configs/default.yaml

pipeline:
  player_detector:
    name: "yolov8_player"      # ← change to "detectron2_player"

  player_tracker:
    name: "bytetrack"          # ← change to "deepsort" (GPL-3.0 — see LICENSE notes)
```

### Registered Models

| Registry | Key | Status |
|---|---|---|
| Detector | `yolov8_player` | ✅ Implemented |
| Detector | `yolov8_ball` | ✅ Implemented |
| Detector | `detectron2_player` | 🔨 Stub (Apache 2.0) |
| Tracker | `bytetrack` | ✅ Implemented (Apache 2.0) |
| Tracker | `deepsort` | ⚠️ Stub (GPL-3.0) |
| Court | `hough_lines` | ✅ Implemented |

---

## Adding a New Model

1. Create a new file in `libs/detectors/src/tennis_heatmap/detectors/` or `libs/trackers/`.
2. Implement the `BaseDetector` or `BaseTracker` interface.
3. Add the `@DetectorRegistry.register("my_model")` decorator.
4. Import the module in the package `__init__.py`.
5. Set `name: "my_model"` in your YAML config.

That's it — no factory or pipeline code needs changing.

---

## License Notes

| Component | License | Commercial Use |
|---|---|---|
| YOLOv8 (`ultralytics`) | AGPL-3.0 | ⚠️ Enterprise license required |
| ByteTrack (`trackers`) | Apache 2.0 | ✅ |
| Supervision | MIT | ✅ |
| OpenCV | Apache 2.0 | ✅ |
| PyTorch | BSD-3-Clause | ✅ |
| FastAPI | MIT | ✅ |
| Detectron2 (stub) | Apache 2.0 | ✅ |

> **Critical**: `ultralytics` (YOLOv8) is AGPL-3.0. For closed-source commercial deployment, purchase an [Ultralytics Enterprise License](https://ultralytics.com/license) or replace with Detectron2.

---

## Development

```bash
make test      # Run all unit tests
make lint      # Ruff linter
make fmt       # Auto-format
```
