# AI-Powered Video Analyzer

**Offline, privacy-first AI video understanding.**

Analyze local video files using a fully offline AI pipeline: object detection, scene captioning, speech transcription, audio event detection, and LLM summarization — no cloud, no data upload, no telemetry.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)

---

## What It Does

| Capability | Model | Notes |
|---|---|---|
| Object detection | YOLO (ultralytics) or VisionServeX | Configurable backend |
| Scene captioning | BLIP (Salesforce) | Generates natural-language frame descriptions |
| Speech transcription | Whisper (OpenAI) | Multilingual, local |
| Audio event detection | PANNs (CNN14) | Requires CNN14 checkpoint |
| LLM summarization | Any Ollama model | Runs fully offline |
| Adaptive sampling | Built-in | Scene-change and motion-aware frame selection |

---

## Privacy Guarantee

All inference runs on your local machine. No frames, audio, or text leave your computer. Internet access is only needed for **initial model downloads**.

---

## Installation

### Option 1 — pip (recommended)

```bash
# Minimal install (no heavy models)
pip install -e .

# Full local AI stack (all pipeline components)
pip install -r pip_requirements.txt

# Or use extras
pip install -e ".[full]"           # Full local stack
pip install -e ".[full,gui]"       # Full stack + GUI dependencies
pip install -e ".[dev]"            # Add test/lint tools
```

### Option 2 — Conda

```bash
conda env create -f environment.yml
conda activate ai-video-analyzer
```

### Option 3 — VisionServeX optional backend

```bash
# Install base package first, then add VisionServeX
pip install -e ".[full]"
pip install "visionservex[hf,rfdetr]"
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
ollama pull partai/dorna-llama3:latest
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
| Whisper | Auto-downloaded on first use | `~/.cache/whisper/` |
| BLIP | Auto-downloaded on first use | `~/.cache/huggingface/` |
| YOLO | Auto-downloaded on first use | `~/.ultralytics/` |
| PANNs CNN14 | Manual download required | See below |

### PANNs CNN14 checkpoint

```bash
mkdir -p models
# Download cnn14.pth from the PANNs repository:
# https://github.com/qiuqiangkong/audioset_tagging_cnn
# Place it at:
models/cnn14.pth
```

### Windows — Tesseract OCR (optional)

Install from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki). OCR is used in some modes for text extraction.

---

## Usage

### CLI

```bash
# Basic analysis
ai-video-analyzer analyze /path/to/video.mp4

# With options
ai-video-analyzer analyze /path/to/video.mp4 \
    --strategy adaptive \
    --target-fps 1.0 \
    --backend auto \
    --whisper-model base \
    --ollama-model phi4:latest \
    --output-dir ./results

# Compatibility shim (matches the README's original example)
python video_processing.py --video /path/to/video.mp4 --save

# Use the VisionServeX backend
ai-video-analyzer analyze video.mp4 --backend visionservex --detector-model rf-detr-base

# Skip heavy models for a quick structural test
ai-video-analyzer analyze video.mp4 --no-captioning --no-audio-events --no-summarization --backend none

# Check your environment
ai-video-analyzer doctor

# Benchmark frame sampling and detection
ai-video-analyzer benchmark video.mp4 --strategy adaptive
```

### GUI

```bash
# Launch the Tkinter GUI
python video_processing_gui.py

# Or via CLI subcommand
ai-video-analyzer gui
```

The GUI lets you:
- Load a video file
- Choose transcription language
- Pick an Ollama model from a live list
- Configure frame sampling rate
- Start processing and view results

### Python API

```python
from ai_powered_video_analyzer.config import AnalysisConfig
from ai_powered_video_analyzer.core import analyze_video

config = AnalysisConfig(
    video_path="my_video.mp4",
    frame_strategy="adaptive",
    target_fps=1.0,
    backend="auto",          # "auto" | "yolo" | "visionservex" | "none"
    whisper_model="base",
    ollama_model="phi4:latest",
    output_dir="./results",
)

report = analyze_video(config)
print(report.summary)
print(report.to_json())
```

### VisionServeX Backend

If VisionServeX is installed, it can be used as a drop-in replacement for the YOLO backend:

```python
config = AnalysisConfig(
    video_path="video.mp4",
    backend="visionservex",
    detector_model="rf-detr-base",   # or any model in your VisionServeX registry
)
```

```bash
ai-video-analyzer analyze video.mp4 --backend visionservex --detector-model rf-detr-base
```

If VisionServeX is not installed, you will see a helpful error:

```
ImportError: VisionServeX backend requested but not installed.
Install with: pip install 'visionservex[hf,rfdetr]'
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
    ├─ Object detection  → detections per frame
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

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO
- [OpenAI Whisper](https://github.com/openai/whisper) — Speech transcription
- [Salesforce BLIP](https://github.com/salesforce/BLIP) — Image captioning
- [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) — Audio event detection
- [Ollama](https://ollama.com) — Local LLM inference
- **Dr. Mark Eramian** and the **Image Lab, Department of Computer Science, University of Saskatchewan** — mentorship and research guidance.

---

## References

- **YOLO (YOLOv11)**: Khanam, R., & Hussain, M. (2024). [arXiv:2410.17725](https://arxiv.org/abs/2410.17725)
- **Whisper**: Radford, A., et al. (2022). [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
- **BLIP**: Li, J., et al. (2022). [arXiv:2201.12086](https://arxiv.org/abs/2201.12086)
- **PANNs**: Kong, Q., et al. (2020). IEEE/ACM TASLP.

---

*Developed by [Arash Sajjadi](https://www.linkedin.com/in/arash-sajjadi/) as a local AI research initiative.*
