# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses [Semantic Versioning](https://semver.org/).

---

## [0.2.0] — 2026-06-14

### Fixed (Issue #1)
- **Added `pip_requirements.txt`** — the README referenced this file but it did not exist.
- **Replaced `conda_requirements.txt` with `environment.yml`** — the old file was a Windows conda
  export dump containing invalid entries like `ace-tools  0.0  pypi_0  pypi`. The new
  `environment.yml` uses valid conda environment format that `conda env create -f` accepts.
- **Added `video_processing.py`** — the README documented
  `python video_processing.py --video path/to/video.mp4 --save` but only
  `video_processing_gui.py` existed. The new CLI script is a full-featured wrapper.

### Added
- **`pyproject.toml`** with proper packaging, CLI entry point (`ai-video-analyzer`), and
  optional extras (`[full]`, `[gui]`, `[visionservex]`, `[dev]`).
- **Package structure** (`ai_powered_video_analyzer/`) — installable Python package.
- **`ai_powered_video_analyzer/cli.py`** — Click-free argparse CLI with subcommands:
  `analyze`, `doctor`, `benchmark`, `gui`.
- **`ai_powered_video_analyzer/config.py`** — `AnalysisConfig` dataclass with all tunable
  parameters and sensible defaults.
- **`ai_powered_video_analyzer/frames.py`** — `FrameSampler` with five strategies:
  `uniform`, `adaptive`, `scene_change`, `motion_aware`, `hybrid`. Supports `target_fps`,
  `min_fps`, `max_fps`, `scene_threshold`, `motion_threshold`, `max_frames`.
- **`ai_powered_video_analyzer/backends/`** — pluggable detection backend system:
  - `YOLOBackend` — wraps ultralytics YOLO with batched inference support.
  - `VisionServeXBackend` — optional backend; gracefully raises `ImportError` with install
    instructions when VisionServeX is not installed.
  - `NullBackend` — returns empty detections for testing and graceful degradation.
  - `load_backend(name)` — auto-selects best available backend (VisionServeX → YOLO → Null).
- **`ai_powered_video_analyzer/diagnostics.py`** — `doctor` command checks Python, ffmpeg,
  OpenCV, torch, ultralytics, Whisper, BLIP, PANNs, Ollama, VisionServeX, Tesseract, moviepy.
- **`ai_powered_video_analyzer/reports.py`** — `AnalysisReport` dataclass with `.to_json()`,
  `.to_markdown()`, `.to_dict()`. Writes JSON sidecar + Markdown + legacy `report.txt`.
- **`ai_powered_video_analyzer/audio.py`** — audio extraction and PANNs event detection.
- **`ai_powered_video_analyzer/transcription.py`** — lazy-loading Whisper transcription with
  model cache and GPU cleanup.
- **`ai_powered_video_analyzer/captioning.py`** — BLIP captioning with lazy model cache.
- **`ai_powered_video_analyzer/summarization.py`** — Ollama LLM summarization with
  graceful fallback when Ollama is not running.
- **`ai_powered_video_analyzer/logging_utils.py`** — structured logging, `timed_stage()`
  context manager for per-stage timing.
- **`ai_powered_video_analyzer/core.py`** — main pipeline orchestrator.
- **`environment.yml`** — valid conda environment file (replaces broken conda export).
- **`pip_requirements.txt`** — clean pip requirements (fixes issue #1).
- **`.gitignore`** — comprehensive gitignore including model weights, temp files, outputs.
- **`tests/`** — test suite runnable without model downloads:
  - `test_dependencies.py` — verifies all referenced files exist, pip_requirements valid,
    environment.yml valid, README consistency.
  - `test_cli.py` — CLI help and error exit smoke tests.
  - `test_backends.py` — backend unit tests with mocks.
  - `test_frames.py` — frame sampler unit tests with synthetic frames.
  - `test_reports.py` — report schema and serialization tests.
  - `test_diagnostics.py` — doctor command tests.
  - `test_config.py` — config dataclass tests.

### Changed
- **`video_processing_gui.py`** — preserved verbatim; GUI functionality unchanged.
- **`README.md`** — fully rewritten: correct install instructions, valid file references,
  CLI and GUI usage, VisionServeX optional backend, troubleshooting, development guide.

### Removed
- **`conda_requirements.txt`** — replaced by `environment.yml` with valid conda format.

---

## [0.1.0] — 2025 (initial release)

- Initial release with `video_processing_gui.py`.
- Offline object detection (YOLO), captioning (BLIP), transcription (Whisper),
  audio event detection (PANNs), and LLM summarization (Ollama).
