# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses [Semantic Versioning](https://semver.org/).

---

## [1.1.0] — 2026-06-14

### Added
- **`docs/USER_GUIDE.md`** — 18-section beginner guide covering install, first run, presets, captioning, Ollama, benchmarking, common errors, privacy rules, and known limitations.
- **`docs/LLM_AGENT_GUIDE.md`** — Complete agent guide with non-negotiable safety rules, canonical commands, key file map, pipeline overview, how-to add backends/fields/tests, release checklist, and attribution hygiene checklist.
- **`reports/quality/v1.1-final-audit.md`** — 58-point final audit covering contributor hygiene, install UX, README, LLM-agent friendliness, code quality, benchmarks, and release readiness.
- **`reports/benchmarks/v1.1/`** — Compact benchmark results for 3 real videos (dog, fire, tourist) on RTX 5080.
- **`AnalysisReport.preset`** and **`AnalysisReport.frame_strategy`** fields — JSON and Markdown reports now include the detection preset and sampling strategy used.
- **`AnalysisReport.timings`** — Per-stage wall times included in JSON and Markdown reports.
- **`AnalysisReport.top_labels(n)`** — Helper method for computing top-n detected labels.
- **New test**: `test_docs_user_guide_exists`, `test_docs_llm_agent_guide_exists`, `test_llm_agent_guide_contains_safety_rules` — verify doc files and safety rule content.
- **New test**: `test_no_claude_anthropic_in_tracked_files` — CI guard against attribution drift.
- **New test**: `test_readme_no_yolo_in_credits` — CI guard against YOLO reappearing in credits.
- **New test**: `test_report_has_preset_and_timings` — verifies new report fields round-trip through JSON.

### Changed
- **README rewritten** for v1.1: professional structure, beginner-readable, expanded output file section, accurate troubleshooting, links to USER_GUIDE and LLM_AGENT_GUIDE.
- **`AnalysisReport.to_markdown()`** improved: run configuration table, detected objects table with max-confidence column, scene captions table with timestamps, explicit warning when no detections found.
- **LLM summary prompt** improved: requests evidence-grounded output with Observed/Interpretation labels; says "Insufficient evidence" instead of generating a generic narrative.
- **`_fallback_summary()`** is now honest: says "LLM summarization unavailable" with install instructions instead of a misleading narrative.
- **CLI `analyze` output** improved: prints output file paths (JSON, Markdown) clearly; warns when no detections found; notes when captioning or summarization is disabled.
- **`_cmd_gui()` and `_cmd_list_models()`** signature fixed: both now correctly accept the `args` parameter (was missing, would cause `TypeError` if called).
- **Version bumped to 1.1.0**.

### Fixed
- `_cmd_gui(args)` and `_cmd_list_models(args)` were called with an argument but defined without one — fixed.
- Stale remote branch `hardening/v0.2.0-modern-video-core` deleted (contained `Co-Authored-By: Claude Sonnet 4.6`; was the root cause of the GitHub Contributors sidebar issue).

### Verified
- GitHub Contributors API returns only `arashsajjadi`.
- No Claude/Anthropic in any commit message on main (`git log --all --format='%B' | grep -i co-author` → empty).
- No private media, model weights, or secrets committed.
- 86 tests pass (77 existing + 9 new).

---

## [1.0.1] — 2026-06-14

### Changed
- **README Credits**: Removed Ultralytics/YOLO from the Credits section. Legacy YOLO backend is now noted in the Legacy CLI section only.
- **README References**: Removed YOLO paper reference; YOLO is not a primary component.
- **README Legacy CLI**: Added explicit note that `--backend legacy_yolo` with `[legacy-yolo]` extra is the preserved legacy path; not recommended for new use.

### Removed
- Stale v0.2/v0.3-era issue response replaced with a current v1.0.0 summary on GitHub issue #1.

---

## [1.0.0] — 2026-06-14

### Added
- **`[vision]` pip extra** — `pip install -e ".[vision]"` installs VisionServeX only; cleanest install path for detection.
- **`--preset`** flag on `analyze` and `benchmark` commands. Replaces the verbose `--detector-preset`. Choices: `fast | balanced | quality | quality+`.
- **`--model`** flag to override preset with an explicit VisionServeX model ID.
- **`--confidence`** flag (replaces `--detection-confidence`) for cleaner CLI.
- **`--compare`** flag on `benchmark` to compare all presets side-by-side.
- **`--save-video`** replaces `--save-annotated-video`.
- **`benchmark` output** now reports: duration, frames selected, model load time, detection runtime, fps analyzed, total detections, top labels.
- **"For coding agents" section** in README with install/test/reference paths.
- **v1.0 benchmark results** in `reports/benchmarks/v1.0/` for 4 videos (dog, fire, Niagara, car cleaning) + private fall smoke (sanitized).
- **Python 3.13 classifier** added to `pyproject.toml`.

### Changed
- **`pyproject.toml` restructured**: base deps now minimal (`numpy`, `Pillow`, `opencv-python-headless`); `tqdm` and `psutil` removed (not used); `[vision]` extra added; `[full]` updated.
- **README fully rewritten** for v1.0: pip-first, no conda-first install path, clean canonical commands, "For coding agents" section, correct credit attribution.
- **`pip_requirements.txt` simplified** and updated for v1.0.
- **`video_processing.py`** shim updated to use `--preset` instead of `--detector-preset`.
- **CLI description** dynamically reflects version from `__version__` (no more hardcoded string).
- **`list-models`** output improved with `[default]` marker.
- **`eval-backends` subcommand removed** (redundant with `benchmark --compare`).
- Version bumped to `1.0.0`.

### Removed
- **`conda_requirements.txt`** — the broken Windows conda export dump that was the root of issue #1. Removed from git tracking.
- **`environment.yml`** — removed to avoid confusion; conda is not the primary or recommended install path.
- **`--detector-preset`** — replaced by `--preset` (shorter, canonical).
- **`--detection-confidence`** — replaced by `--confidence`.
- **`--save-annotated-video`** — replaced by `--save-video`.
- **`eval-backends` subcommand** — merged into `benchmark --compare`.
- **`tqdm` and `psutil`** from base dependencies (not used in source).

### Benchmark evidence (RTX 5080, 2026-06-14)

| Video     | Preset   | Model   | ms/frame | fps  | Result                       |
|-----------|----------|---------|----------|------|------------------------------|
| dog       | fast     | dfine-n | 19.2     | 52.1 | dog(0.90) ✓                  |
| dog       | balanced | dfine-s | 27.3     | 36.7 | dog(0.87), person(0.92) ✓    |
| dog       | quality  | dfine-m | 21.2     | 47.3 | dog(0.93), person(0.93) ✓    |
| fire      | balanced | dfine-s | 19.3     | 51.8 | person(0.72) — fire∉COCO-80  |
| niagara   | balanced | dfine-s | 17.1     | 58.4 | person(0.91), phone(0.89) ✓  |
| car clean | balanced | dfine-s | 26.6     | 37.6 | car(0.93), person(0.94) ✓    |

Private fall/smoke: `completed, runtime=0.1s, frames=10, no media committed.`

---

## [0.3.0] — 2026-06-14

### Changed (Breaking)
- **VisionServeX D-FINE is now the primary detection backend.** `AnalysisConfig.backend` default changed from `"auto"` to `"visionservex"`. `--backend` CLI default changed from `"auto"` to `"visionservex"`.
- **Ultralytics/YOLO removed from the default pipeline.** YOLO is no longer auto-selected, no longer a dependency in `[full]` or `pip_requirements.txt`. Use `--backend legacy_yolo` or install `.[legacy-yolo]` if needed.
- **Backend auto-selection** now falls back to Null (no detection) instead of YOLO when VisionServeX is unavailable.
- `pyproject.toml` `[full]` extra: removed `ultralytics`, added `visionservex>=3.11.0`. New `[legacy-yolo]` extra for backward compatibility.

### Added
- **Detector presets** (`--detector-preset`): `fast` → dfine-n, `balanced` → dfine-s (default), `quality` → dfine-m, `quality+` → dfine-l.
- **Summary styles** (`--summary-style`): `concise` (default), `evidence`, `technical`, `narrative`.
- **`--list-ollama-models`** flag on the `analyze` subcommand.
- **`list-models` subcommand** — shows all VisionServeX presets and registry entries.
- **`eval-backends` subcommand** — benchmarks fast/balanced/quality presets on a video and writes JSON results.
- **Ollama auto-discovery** — `AnalysisConfig.ollama_model` auto-populated from `ollama list` output (prefers phi4, qwen, llama3).
- **`resolve_model_id(model_id, preset)`** — exported helper from `visionservex_backend.py`.
- **`list_available_detect_models()`** — queries VisionServeX registry for wired detection models.
- **`PRESET_MODELS`** — exported from `backends/__init__.py` for CLI and documentation.
- **`--compare-detectors`** flag on the `benchmark` subcommand.
- **`reports/benchmarks/v0.3.0/`** — real-video benchmark results on RTX 5080 (dog, fire, Niagara Falls videos).
- **Doctor command** — VisionServeX is now a required check; D-FINE registry probe added; ultralytics check moved to optional/legacy.

### Evidence (benchmarks on RTX 5080, 2026-06-14)
- dfine-s (balanced): dog detection max score 0.87, 18.6ms/frame, 30 frames.
- dfine-m (quality): dog detection max score 0.93, 21.0ms/frame, 30 frames.
- dfine-n (fast): dog detection max score 0.90, 21.2ms/frame, 30 frames.
- rfdetr-nano evaluated and rejected: hallucinated bicycle/sheep/horse on dog video.
- Fire/smoke video: no COCO-80 fire class; food misclassification is expected and documented.
- Private fall/smoke video: completed, runtime=0.9s, no media committed.

### Fixed
- `YOLOBackend` renamed to `LegacyYOLOBackend`; `yolo_backend.py` renamed to `legacy_yolo_backend.py`.
- `test_config.py`: updated `backend == "auto"` assertion to `"visionservex"`.
- `test_dependencies.py`: updated version assertion to `"0.3.0"`.
- `test_backends.py`: fixed `b._parse_result()` → `_parse_detections()` (module-level function).
- `core.py`: `load_backend()` call now passes `preset=config.detector_preset`.

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
