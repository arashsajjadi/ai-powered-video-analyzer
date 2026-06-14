# AI-Powered Video Analyzer

Offline, privacy-first AI video analysis. Runs entirely on your local machine — no cloud, no data upload, no telemetry.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)

---

## What it does

| Stage | Technology | Notes |
|---|---|---|
| Object detection | VisionServeX D-FINE (dfine-s default) | Primary backend; real-video benchmarked |
| Scene captioning | BLIP (Salesforce) | Natural-language frame descriptions |
| Speech transcription | Whisper (OpenAI) | Multilingual, offline |
| Audio events | PANNs CNN14 | Requires CNN14 checkpoint |
| LLM summarization | Ollama (any local model) | Fully offline |
| Frame sampling | Built-in adaptive sampler | Scene-change and motion-aware |

---

## Quick start

```bash
# 1. Install
git clone https://github.com/arashsajjadi/ai-powered-video-analyzer.git
cd ai-powered-video-analyzer
python -m pip install -U pip
python -m pip install -e ".[vision]"

# 2. Check dependencies
ai-video-analyzer doctor

# 3. Analyze a video
ai-video-analyzer analyze "/path/to/video.mp4" --preset balanced
```

---

## Installation

### Core install (detection only)

```bash
python -m pip install -e ".[vision]"
```

This installs the frame sampler, VisionServeX D-FINE detection backend, and the CLI. No YOLO, no ultralytics, no cloud dependencies.

### Full install (all pipeline stages)

```bash
python -m pip install -r pip_requirements.txt
```

Includes: VisionServeX, Whisper, BLIP, PANNs, Ollama client, PyTorch, librosa.

### Development

```bash
python -m pip install -e ".[dev]"
pytest -q
```

---

## External tools

### ffmpeg — required for audio extraction

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Ollama — optional, for LLM summarization

```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull phi4:latest
```

---

## First analysis

```bash
ai-video-analyzer analyze "/path/to/video.mp4" --preset balanced
```

**Fast test** (no transcription or summarization, 40 frames max):

```bash
ai-video-analyzer analyze "/path/to/video.mp4" \
    --preset fast \
    --max-frames 40 \
    --no-captioning \
    --no-audio-events \
    --no-summarization
```

**With all stages:**

```bash
ai-video-analyzer analyze "/path/to/video.mp4" \
    --preset balanced \
    --whisper-model base \
    --ollama-model phi4:latest \
    --output-dir ./results
```

---

## Doctor command

```bash
ai-video-analyzer doctor
```

Prints the status of every dependency: Python version, VisionServeX, ffmpeg, GPU, Ollama models, Whisper, BLIP, PANNs.

---

## Output files

Each run writes to `--output-dir` (default: current directory):

| File | Contents |
|---|---|
| `<video>_analysis.json` | Detections, captions, transcript, audio events, timings |
| `<video>_analysis.md` | Human-readable Markdown report |
| `report.txt` | Legacy plain-text report |

---

## Detection presets

VisionServeX D-FINE presets (benchmarked on RTX 5080 with real videos):

| Preset | Model | ~ms/frame | When to use |
|---|---|---|---|
| `fast` | dfine-n | 17–21 | Large videos, speed matters |
| `balanced` | dfine-s | 17–27 | **Default** — best all-round |
| `quality` | dfine-m | 21–32 | Best COCO accuracy |
| `quality+` | dfine-l | higher | Maximum accuracy, slower |

```bash
ai-video-analyzer analyze video.mp4 --preset fast
ai-video-analyzer analyze video.mp4 --preset quality

# Use a specific model ID instead of a preset
ai-video-analyzer analyze video.mp4 --model dfine-x
```

**Known limitation**: COCO-80 does not include fire, smoke, or weather. D-FINE models will return approximate visual matches (e.g. food textures for fire). For fire/smoke detection, use the BLIP captioning stage (`--no-captioning` is off by default).

```bash
# List available models in your local VisionServeX registry
ai-video-analyzer list-models
```

---

## Optional: LLM summarization with Ollama

When Ollama is running, the pipeline generates a natural-language summary at the end.

```bash
# Start Ollama (if not already running)
ollama serve

# List available models
ai-video-analyzer analyze video.mp4 --list-ollama-models

# Choose a style
ai-video-analyzer analyze video.mp4 --preset balanced --summary-style evidence
```

Summary styles:

| Style | Description |
|---|---|
| `concise` | One paragraph, plain language (default) |
| `evidence` | Grounded in detected objects and events |
| `technical` | Includes model names and confidence scores |
| `narrative` | Story-form prose |

Ollama is **optional** — the detection and captioning stages work without it.

---

## PANNs model (audio event detection)

PANNs CNN14 is not auto-downloaded. Download it manually:

```bash
mkdir -p models
# Download cnn14.pth from:
# https://github.com/qiuqiangkong/audioset_tagging_cnn
# Place it at: models/cnn14.pth
```

If not present, audio event detection is silently skipped.

---

## Benchmarking

```bash
# Benchmark one video with the balanced preset
ai-video-analyzer benchmark "/path/to/video.mp4" --preset balanced

# Compare all three presets side by side
ai-video-analyzer benchmark "/path/to/video.mp4" --compare
```

Output:

```
Benchmark: video.mp4
  Strategy : adaptive
  Preset   : balanced

Results:
  Video duration     : 8.2s
  Frames selected    : 8  (sampling: 0.08s)
  Model load+warmup  : 0.52s
  Detection runtime  : 0.15s  (18.6ms/frame, 53.8fps)
  Total detections   : 48
  Top labels         : dog(9), person(7), cat(5)
  Model              : dfine-s
```

---

## Troubleshooting

### VisionServeX not installed

```
Error: VisionServeX backend requested but not installed or failed to load.
Install with: pip install 'visionservex[hf,rfdetr]'
```

Fix:

```bash
python -m pip install -e ".[vision]"
# or
pip install "visionservex[hf,rfdetr]"
```

### ffmpeg not found

```bash
sudo apt install ffmpeg     # Ubuntu/Debian
brew install ffmpeg         # macOS
```

### Ollama not running

```bash
ollama serve
ollama pull phi4:latest
```

### GPU not detected

```bash
python -c "import torch; print(torch.cuda.is_available())"
ai-video-analyzer analyze video.mp4 --device cpu
```

### Run doctor for a full diagnosis

```bash
ai-video-analyzer doctor
```

---

## Legacy CLI

The original `video_processing.py` interface is preserved:

```bash
python video_processing.py --video "/path/to/video.mp4" --save
python video_processing.py --video video.mp4 --preset fast --no-summarization
```

The Tkinter GUI is also preserved:

```bash
python video_processing_gui.py
# or
ai-video-analyzer gui
```

**Legacy optional backend**: The original YOLO-based detection path from `video_processing_gui.py` is available as `--backend legacy_yolo` with the `[legacy-yolo]` extra (`pip install -e ".[legacy-yolo]"`). This path is not recommended for new use. The default and supported detection backend is VisionServeX D-FINE.

---

## Pipeline architecture

```
Video file
    │
    ├─ Adaptive frame sampling (scene + motion aware)
    │       └─ FrameRecord list (frame, timestamp, reason)
    │
    ├─ Object detection    → VisionServeX D-FINE (per frame)
    ├─ Scene captioning    → BLIP (per frame, optional)
    │
    ├─ Audio extraction
    │       ├─ Transcription (Whisper, optional)
    │       └─ Audio events (PANNs CNN14, optional)
    │
    └─ LLM summarization  → Ollama (optional)
            │
            └─ AnalysisReport
                    ├─ <video>_analysis.json
                    ├─ <video>_analysis.md
                    └─ report.txt
```

---

## For coding agents

- Install: `python -m pip install -e ".[vision]"`
- Test: `pytest -q`
- No media committed — videos are local-only test files
- VisionServeX reference project at `/home/arash/PycharmProjects/VisionServeX` is read-only
- Benchmark videos: `~/Videos/TETS\ VIDEOS/` (local, not committed)
- Config: `ai_powered_video_analyzer/config.py` — `AnalysisConfig` dataclass
- Pipeline entry: `ai_powered_video_analyzer/core.py` — `analyze_video(config)`
- Detection backend: `ai_powered_video_analyzer/backends/visionservex_backend.py`

---

## Development

```bash
git clone https://github.com/arashsajjadi/ai-powered-video-analyzer.git
cd ai-powered-video-analyzer
python -m pip install -e ".[dev]"
pytest -q
ruff check .
```

---

## Credits

- [VisionServeX](https://github.com/arashsajjadi/VisionServeX) — local CV model gateway; primary detection backend
- [D-FINE](https://github.com/Peterande/D-FINE) — Fine-grained Distribution Refined DETR (Peng et al., 2024)
- [OpenAI Whisper](https://github.com/openai/whisper) — Speech transcription
- [Salesforce BLIP](https://github.com/salesforce/BLIP) — Image captioning
- [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) — Audio event detection
- [Ollama](https://ollama.com) — Local LLM inference
- **Dr. Mark Eramian** and the **Image Lab, Department of Computer Science, University of Saskatchewan** — research mentorship

---

## References

- D-FINE: Peng et al. (2024). [arXiv:2410.13842](https://arxiv.org/abs/2410.13842)
- Whisper: Radford et al. (2022). [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
- BLIP: Li et al. (2022). [arXiv:2201.12086](https://arxiv.org/abs/2201.12086)
- PANNs: Kong et al. (2020). IEEE/ACM TASLP.

---

## License

MIT — see [LICENSE](LICENSE).

*Developed by [Arash Sajjadi](https://www.linkedin.com/in/arash-sajjadi/), University of Saskatchewan.*
