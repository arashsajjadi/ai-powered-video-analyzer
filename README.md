# AI-Powered Video Analyzer

**Offline, privacy-first AI video understanding.**

Analyze local video files using a fully offline AI pipeline: object detection, scene captioning, speech transcription, audio event detection, and LLM summarization — no cloud, no data upload, no telemetry.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)

---

## What It Does

| Capability | Model | Notes |
|---|---|---|
| Object detection | VisionServeX D-FINE (dfine-s default) | Primary backend since v0.3.0 — real-video benchmarked |
| Scene captioning | BLIP (Salesforce) | Natural-language frame descriptions |
| Speech transcription | Whisper (OpenAI) | Multilingual, local |
| Audio event detection | PANNs (CNN14) | Requires CNN14 checkpoint |
| LLM summarization | Any Ollama model | Runs fully offline |
| Adaptive sampling | Built-in | Scene-change and motion-aware frame selection |

---

## Why VisionServeX D-FINE (since v0.3.0)

The default detection backend is now VisionServeX with D-FINE transformer models rather than Ultralytics YOLO. This change is backed by real-video benchmarks run on RTX 5080:

| Preset   | Model    | ms/frame | Use case |
|----------|----------|----------|----------|
| fast     | dfine-n  | ~18–21   | Speed-critical pipelines |
| balanced | dfine-s  | ~15–19   | **Default** — best accuracy/speed balance |
| quality  | dfine-m  | ~16–21   | Highest COCO accuracy |
| quality+ | dfine-l  | higher   | Large model, maximum accuracy |

Key findings from benchmarks (see `reports/benchmarks/v0.3.0/`):
- All D-FINE presets correctly identify COCO-class subjects (dog detection: max confidence 0.87–0.93).
- rfdetr-nano was evaluated and rejected — it hallucinated bicycle/sheep/horse on a dog video.
- **Limitation**: COCO-80 classes do not include fire or smoke. Detection of abstract/environmental categories requires the BLIP captioning stage.

Ultralytics/YOLO is still accessible via `--backend legacy_yolo` but is not part of the default pipeline.

---

## Privacy Guarantee

All inference runs on your local machine. No frames, audio, or text leave your computer. Internet access is only needed for **initial model downloads**.

---

## Installation

### Option 1 — pip (recommended)

```bash
# Minimal install (no heavy models)
pip install -e .

# Full local AI stack — VisionServeX D-FINE + Whisper + BLIP + PANNs + Ollama
pip install -r pip_requirements.txt

# Or use extras
pip install -e ".[full]"       # Full stack (VisionServeX is included)
pip install -e ".[full,gui]"   # Full stack + GUI dependencies
pip install -e ".[dev]"        # Add test/lint tools

# Legacy YOLO backend (not recommended, not installed by default since v0.3.0)
pip install -e ".[legacy-yolo]"
```

### Option 2 — Conda

```bash
conda env create -f environment.yml
conda activate ai-video-analyzer
```

---

## Required External Tools

### Ollama (LLM summarization)

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Then pull a model (phi4 is a good balance of speed and quality)
ollama pull phi4:latest
# Other good options:
ollama pull qwen:14b
ollama pull llama3
```

### ffmpeg (audio extraction)

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg
```

---

## Model Downloads

| Model | Command | Where |
|---|---|---|
| D-FINE / RF-DETR | Auto-downloaded by VisionServeX on first use | `~/.cache/visionservex/` |
| Whisper | Auto-downloaded on first use | `~/.cache/whisper/` |
| BLIP | Auto-downloaded on first use | `~/.cache/huggingface/` |
| PANNs CNN14 | Manual download required | See below |

### PANNs CNN14 checkpoint

```bash
mkdir -p models
# Download cnn14.pth from the PANNs repository:
# https://github.com/qiuqiangkong/audioset_tagging_cnn
# Place it at:
models/cnn14.pth
```

---

## Usage

### CLI

```bash
# Basic analysis (VisionServeX dfine-s, adaptive sampling)
ai-video-analyzer analyze /path/to/video.mp4

# Choose a detector preset
ai-video-analyzer analyze video.mp4 --detector-preset fast        # dfine-n, ~18ms/frame
ai-video-analyzer analyze video.mp4 --detector-preset balanced    # dfine-s (default)
ai-video-analyzer analyze video.mp4 --detector-preset quality     # dfine-m
ai-video-analyzer analyze video.mp4 --detector-preset quality+    # dfine-l

# Choose a summary style
ai-video-analyzer analyze video.mp4 --summary-style concise      # one paragraph (default)
ai-video-analyzer analyze video.mp4 --summary-style evidence     # grounded in detected objects
ai-video-analyzer analyze video.mp4 --summary-style technical    # model names and confidence scores
ai-video-analyzer analyze video.mp4 --summary-style narrative    # story-form prose

# List available Ollama models
ai-video-analyzer analyze video.mp4 --list-ollama-models

# List available VisionServeX detection models
ai-video-analyzer list-models

# Explicit model ID (overrides preset)
ai-video-analyzer analyze video.mp4 --detector-model dfine-x

# Skip heavy models for a quick structural test
ai-video-analyzer analyze video.mp4 --no-captioning --no-audio-events --no-summarization --backend none

# Compatibility shim (matches the original README example)
python video_processing.py --video /path/to/video.mp4 --save

# Check your environment
ai-video-analyzer doctor

# Benchmark detection speed and frame sampling
ai-video-analyzer benchmark video.mp4 --compare-detectors

# Evaluate all detector presets on a video
ai-video-analyzer eval-backends video.mp4 --output-dir reports/benchmarks/
```

### GUI

```bash
# Launch the Tkinter GUI
python video_processing_gui.py

# Or via CLI subcommand
ai-video-analyzer gui
```

### Python API

```python
from ai_powered_video_analyzer.config import AnalysisConfig
from ai_powered_video_analyzer.core import analyze_video

config = AnalysisConfig(
    video_path="my_video.mp4",
    frame_strategy="adaptive",
    target_fps=1.0,
    backend="visionservex",         # primary backend since v0.3.0
    detector_preset="balanced",     # fast | balanced | quality | quality+
    whisper_model="base",
    ollama_model="",                # auto-discovered from local Ollama
    summary_style="concise",        # concise | evidence | technical | narrative
    output_dir="./results",
)

report = analyze_video(config)
print(report.summary)
print(report.to_json())
```

---

## Adaptive Frame Sampling

Instead of analyzing every frame (slow) or every N-th frame (may miss events), the adaptive sampler selects frames based on what is actually happening:

| Strategy | Description |
|---|---|
| `uniform` | Every N-th frame at `target_fps` |
| `scene_change` | Triggers on large histogram differences |
| `motion_aware` | Triggers on pixel-level motion |
| `adaptive` | Combines scene + motion + periodic fallback |
| `hybrid` | Same as adaptive |

```bash
# Analyze at 2 fps maximum, using scene-change detection
ai-video-analyzer analyze video.mp4 --strategy adaptive --target-fps 2.0

# Force uniform 1fps (like the original behavior)
ai-video-analyzer analyze video.mp4 --strategy uniform --target-fps 1.0
```

---

## Known Limitations

**Fire, smoke, and environmental phenomena** are not COCO-80 classes. D-FINE and all other COCO-trained detectors will misidentify these as the closest COCO category (food items like cake or pizza at similar color/texture). The BLIP captioning stage handles these cases via natural-language descriptions. A future update may add an open-vocabulary detector preset.

---

## Output

The pipeline produces three output files:

| File | Description |
|---|---|
| `<stem>_analysis.json` | Machine-readable JSON with full detections, captions, transcript, audio events |
| `<stem>_analysis.md` | Human-readable Markdown report |
| `report.txt` | Legacy text report (for backward compatibility) |

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.12 |
| RAM | 8 GB | 32 GB |
| GPU | None (CPU works) | NVIDIA RTX 3060+ |
| Disk | 5 GB | 30 GB (for all models) |

---

## Troubleshooting

### `ffmpeg not found`
```bash
sudo apt install ffmpeg     # Ubuntu/Debian
brew install ffmpeg         # macOS
```

### `Ollama not running`
```bash
ollama serve                # Start in a terminal
ollama pull phi4:latest     # Download a model
```

### `PANNs model not found`
Download `cnn14.pth` from [github.com/qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) and place it at `models/cnn14.pth`.

### `VisionServeX not installed`
```bash
pip install "visionservex[hf,rfdetr]"
# Or with the full stack:
pip install -e ".[full]"
```

### GPU not detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU
ai-video-analyzer analyze video.mp4 --device cpu
```

### Run the doctor
```bash
ai-video-analyzer doctor
```

---

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/arashsajjadi/ai-powered-video-analyzer.git
cd ai-powered-video-analyzer
pip install -e ".[dev]"

# Run tests (no model downloads required)
pytest -q

# Lint
ruff check .
```

---

## How It Works

```
Video file
    │
    ├─ Frame sampling (adaptive / scene-change / motion-aware)
    │       └─ FrameRecord list (frame, timestamp, reason)
    │
    ├─ Object detection  → VisionServeX D-FINE detections per frame
    ├─ Image captioning  → BLIP captions per frame
    │
    ├─ Audio extraction
    │       ├─ Speech transcription  (Whisper)
    │       └─ Audio event detection (PANNs)
    │
    └─ LLM summarization (Ollama) → narrative summary
            │
            └─ AnalysisReport
                    ├─ report.json
                    ├─ report.md
                    └─ report.txt
```

---

## Contributing

This project is open-source. Fork, improve, and open a pull request. Please:
- Run `pytest` before submitting.
- Keep new dependencies optional where possible.
- Document any new model requirements.

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgments

- [VisionServeX](https://github.com/arashsajjadi/VisionServeX) — local-first CV model gateway (primary detection backend since v0.3.0)
- [D-FINE](https://github.com/Peterande/D-FINE) — Fine-grained Distribution Refined DETR (primary detector models)
- [OpenAI Whisper](https://github.com/openai/whisper) — Speech transcription
- [Salesforce BLIP](https://github.com/salesforce/BLIP) — Image captioning
- [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) — Audio event detection
- [Ollama](https://ollama.com) — Local LLM inference
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO (available via `--backend legacy_yolo`)
- **Dr. Mark Eramian** and the **Image Lab, Department of Computer Science, University of Saskatchewan** — mentorship and research guidance.

---

## References

- **D-FINE**: Peng, Y., et al. (2024). D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement. [arXiv:2410.13842](https://arxiv.org/abs/2410.13842)
- **RF-DETR**: Roboflow Research (2024). [github.com/roboflow/RF-DETR](https://github.com/roboflow/RF-DETR)
- **Whisper**: Radford, A., et al. (2022). [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
- **BLIP**: Li, J., et al. (2022). [arXiv:2201.12086](https://arxiv.org/abs/2201.12086)
- **PANNs**: Kong, Q., et al. (2020). IEEE/ACM TASLP.
- **YOLO (YOLOv11)**: Khanam, R., & Hussain, M. (2024). [arXiv:2410.17725](https://arxiv.org/abs/2410.17725)

---

*Developed by [Arash Sajjadi](https://www.linkedin.com/in/arash-sajjadi/) as a local AI research initiative.*
