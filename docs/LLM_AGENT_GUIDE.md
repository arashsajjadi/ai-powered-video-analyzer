# LLM Agent Guide — AI-Powered Video Analyzer

This document is for coding agents, AI assistants, and automated tools working in this repository.
Read this before making any changes.

---

## 1. Repository purpose

This is a local, offline AI video analysis tool. It analyzes video files using VisionServeX D-FINE for
object detection, BLIP for captioning, Whisper for transcription, PANNs for audio events, and Ollama for
LLM summarization. All inference is local — no cloud, no API keys, no telemetry.

Primary use: researchers and developers who need structured video analysis without data leaving their machine.

---

## 2. Non-negotiable rules

These rules apply to every commit, PR, comment, and file change:

### Media and weights
- **Do NOT commit** `.mp4`, `.mov`, `.avi`, or any video file
- **Do NOT commit** `.pt`, `.pth`, `.onnx`, `.bin`, `.safetensors`, or any model weights
- **Do NOT commit** extracted frames, screenshots, or private media
- **Do NOT commit** `.wav` audio files extracted from private recordings
- **Do NOT commit** `.env` files, API keys, or credentials

### Attribution
- **Do NOT credit** Claude, Anthropic, or any AI assistant as a contributor, author, co-author,
  creator, maintainer, or assistant in any public file, commit message, PR description,
  release note, or issue response
- Do NOT add `Co-authored-by: Claude ...` or similar trailers to commits
- The sole project author is Arash Sajjadi. Maintain this in pyproject.toml and `__init__.py`

### Detection backend
- **Do NOT** reintroduce Ultralytics/YOLO as a default or recommended detection path
- VisionServeX D-FINE is the primary and default backend
- The `[legacy-yolo]` extra exists for backward compatibility only
- The legacy backend must not appear in Credits, References, or first-run documentation

### VisionServeX reference project
- **Do NOT modify** `/home/arash/PycharmProjects/VisionServeX` — this is a read-only reference
- You may inspect it to understand the API, but never edit it
- VisionServeX API: `VisionModel(model_id, device=None)`, `model.predict(pil_image)` → `DetectionResult`

---

## 3. Canonical setup commands

```bash
# Clone
git clone https://github.com/arashsajjadi/ai-powered-video-analyzer.git
cd ai-powered-video-analyzer

# Install (detection only)
python -m pip install -U pip
python -m pip install -e ".[vision]"

# Install (full pipeline)
python -m pip install -r pip_requirements.txt

# Install (development)
python -m pip install -e ".[dev]"
```

---

## 4. Canonical test commands

Always run these before committing:

```bash
# Compile check (syntax errors)
python -m compileall ai_powered_video_analyzer/ -q

# Full test suite (77 tests, no model downloads required)
python -m pytest tests/ -q --tb=short

# Verify CLI works
ai-video-analyzer --version
ai-video-analyzer doctor
ai-video-analyzer analyze --help
ai-video-analyzer benchmark --help
ai-video-analyzer list-models
python video_processing.py --help
```

All 77 tests must pass before any commit is made.

---

## 5. Canonical benchmark commands

Use only local test videos. Never commit the video files.

```bash
# Detection benchmark — single preset
ai-video-analyzer benchmark "/path/to/local/video.mp4" --preset balanced --max-frames 40

# Preset comparison
ai-video-analyzer benchmark "/path/to/local/video.mp4" --compare --max-frames 40

# Save results to reports/benchmarks/
# Only commit the summary text, not video files or generated media
```

Benchmark result files go in `reports/benchmarks/<version>/`. Only commit JSON and Markdown summaries.

---

## 6. Key files

| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, extras, entry points |
| `ai_powered_video_analyzer/__init__.py` | Version string — keep in sync with pyproject.toml |
| `ai_powered_video_analyzer/config.py` | `AnalysisConfig` dataclass — all pipeline settings |
| `ai_powered_video_analyzer/core.py` | `analyze_video(config)` — main pipeline orchestrator |
| `ai_powered_video_analyzer/cli.py` | CLI entry point — argparse subcommands |
| `ai_powered_video_analyzer/frames.py` | `FrameSampler` — adaptive frame selection |
| `ai_powered_video_analyzer/reports.py` | `AnalysisReport` dataclass + `save_report()` |
| `ai_powered_video_analyzer/backends/` | Detection backend adapters |
| `ai_powered_video_analyzer/backends/visionservex_backend.py` | Primary VisionServeX backend |
| `ai_powered_video_analyzer/diagnostics.py` | Doctor command — `CheckResult`, `run_doctor()` |
| `ai_powered_video_analyzer/summarization.py` | Ollama LLM summarization |
| `tests/` | Full test suite — 77 tests, all model-free |

---

## 7. Pipeline overview

```
analyze_video(config: AnalysisConfig) → AnalysisReport
    │
    ├─ FrameSampler.sample(video_path) → list[FrameRecord]
    │
    ├─ load_backend(config.backend, preset=config.detector_preset)
    │       → VisionServeXBackend | NullBackend | LegacyYOLOBackend
    │
    ├─ backend.predict_batch(frames) → list[list[Detection]]
    │
    ├─ caption_frame(pil_image) → str    [optional, BLIP]
    ├─ extract_audio(video_path, wav_path)
    ├─ transcribe(wav_path) → (text, language)    [optional, Whisper]
    ├─ detect_audio_events(wav_path) → dict    [optional, PANNs]
    ├─ summarize_report(text, model) → str    [optional, Ollama]
    │
    └─ save_report(report, output_dir, stem) → dict[format: path]
```

The `AnalysisReport` dataclass collects all outputs:
- `detections: list[dict]` — per-frame object detections with label, score, box, timestamp
- `captions: list[dict]` — per-frame BLIP descriptions with timestamp
- `transcript: str` — Whisper output
- `audio_events: dict[str, list[str]]` — PANNs events
- `summary: str` — Ollama LLM summary
- `timings: dict[str, float]` — per-stage wall time
- `preset: str` — which detection preset was used
- `limitations: list[str]` — honest notes about what was skipped or unavailable

---

## 8. How to add a backend safely

1. Create `ai_powered_video_analyzer/backends/my_backend.py`
2. Subclass `BaseBackend` from `backends/base.py`
3. Implement `predict(pil_image, confidence_threshold) → list[Detection]`
4. Register in `backends/__init__.py` under `load_backend()`
5. Add an optional check in `diagnostics.py` (`_check_my_backend()`) with `optional=True`
6. Add tests in `tests/test_backends.py` that mock the external dependency
7. Do NOT add the new backend to base dependencies — use an optional extra

Do NOT load models at import time. Use lazy loading inside `__init__` or `predict()`.

---

## 9. How to add a report field safely

1. Add the field to `AnalysisReport` in `reports.py` with a default value:
   ```python
   new_field: str = ""
   ```
2. Populate it in `core.py` where the report is constructed
3. Update `to_markdown()` to render it in the Markdown report
4. Add a test in `tests/test_reports.py` checking the field survives `to_dict()` and `to_json()`
5. Update `CHANGELOG.md`

`AnalysisReport` uses `dataclasses.asdict()`, so all fields are automatically serialized to JSON.

---

## 10. How to add tests without model downloads

All tests must run without downloading any model weights. Use these patterns:

```python
# Mock VisionServeX
from unittest.mock import patch, MagicMock
with patch("visionservex.VisionModel") as mock_model:
    mock_model.return_value.predict.return_value = fake_result

# Mock Ollama
with patch("subprocess.run") as mock_run:
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "Observed: A dog walks."

# Synthetic frames (no video file needed)
import numpy as np
from PIL import Image
frame = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
```

Run the test suite to verify:

```bash
python -m pytest tests/ -q --tb=short
```

All 77 tests should complete in under 10 seconds.

---

## 11. How to run optional real-video benchmarks

Benchmark videos are local-only. Never commit them.

```bash
# Set your local video path
VIDEO="/path/to/local/test_video.mp4"

# Run benchmark
ai-video-analyzer benchmark "$VIDEO" --compare --max-frames 40

# Save compact results (text summary only, no video)
```

Store sanitized results in `reports/benchmarks/<version>/README.md` and `results.json`.
Do NOT store video files, frames, screenshots, or audio in `reports/`.

If a benchmark video is private, store only a one-line sanitized result:
```
private_video: completed, runtime=0.2s, frames=30, no media committed.
```

---

## 12. Release checklist

Before bumping the version:

- [ ] All 77 tests pass: `pytest -q`
- [ ] `python -m compileall ai_powered_video_analyzer/ -q` exits 0
- [ ] `ai-video-analyzer doctor` exits 0
- [ ] `ai-video-analyzer --version` shows the new version
- [ ] `ai-video-analyzer analyze --help` shows `--preset`
- [ ] Version is consistent: `pyproject.toml` == `__init__.py`
- [ ] CHANGELOG has an entry for the new version
- [ ] No Claude/Anthropic in any file: `grep -rI "claude\|anthropic\|co-authored-by" . --exclude-dir=.git`
- [ ] No model weights committed: `find . -name "*.pt" -o -name "*.pth" -o -name "*.onnx"`
- [ ] No video files committed: `find . -name "*.mp4" -o -name "*.mov" -o -name "*.avi"`
- [ ] README commands are correct and current
- [ ] docs/USER_GUIDE.md and docs/LLM_AGENT_GUIDE.md exist

---

## 13. Public attribution hygiene checklist

This project is authored solely by Arash Sajjadi. These rules are permanent:

- [ ] `pyproject.toml` `authors` field contains only `Arash Sajjadi`
- [ ] `pyproject.toml` `maintainers` field contains only `Arash Sajjadi`
- [ ] `ai_powered_video_analyzer/__init__.py` `__author__` is `"Arash Sajjadi"`
- [ ] README credits list VisionServeX, D-FINE, Whisper, BLIP, PANNs, Ollama, Dr. Mark Eramian — no AI assistants
- [ ] CHANGELOG entries have no `Co-authored-by` trailers
- [ ] No commit message in the repo contains `Co-authored-by: Claude` or `Co-authored-by: Anthropic`
- [ ] GitHub Contributors API returns only `arashsajjadi`

If the GitHub Contributors sidebar shows an AI assistant:
1. Check `git log --all --format='%B' | grep -i "co-authored"` to find the offending commit
2. Check `git ls-remote --heads origin` for stale branches with dirty commits
3. Delete confirmed stale merged branches: `git push origin --delete BRANCH_NAME`
4. If commits on main have the trailer: use `git-filter-repo` with a local backup bundle first
