# AI-Powered Video Analyzer

Offline, privacy-first AI video analysis. Runs entirely on your local machine — no cloud, no data upload, no telemetry.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)

---

## What it does

Analyzes a video file through a local AI pipeline and produces structured outputs:

| Stage | Technology | Default |
|---|---|---|
| Object detection | VisionServeX D-FINE | Required — primary backend |
| Frame sampling | Built-in (adaptive/scene-change) | Enabled |
| Scene captioning | BLIP (Salesforce) | Optional — needs `[full]` |
| Speech transcription | Whisper (OpenAI) | Optional — needs `[full]` |
| Audio events | PANNs CNN14 | Optional — needs `[full]` |
| LLM summarization | Ollama (any local model) | Optional — needs Ollama |

All processing is local. Nothing is sent to any server.

---

## Quick start

```bash
# 1. Clone and install (detection only)
git clone https://github.com/arashsajjadi/ai-powered-video-analyzer.git
cd ai-powered-video-analyzer
python -m pip install -U pip
python -m pip install -e ".[vision]"

# 2. Check your environment
ai-video-analyzer doctor

# 3. Analyze a video
ai-video-analyzer analyze "/path/to/video.mp4" --preset balanced
```

---

## Installation

### Detection only (recommended starting point)

```bash
python -m pip install -e ".[vision]"
```

Installs: frame sampler, VisionServeX D-FINE backend, CLI. No YOLO, no cloud dependencies.

### Full pipeline (all optional stages)

```bash
python -m pip install -r pip_requirements.txt
```

Adds: Whisper, BLIP, PANNs, PyTorch, librosa, Ollama client.

### Development

```bash
python -m pip install -e ".[dev]"
pytest -q
```

---

## External tools

### ffmpeg — required for audio stages

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

# Start and pull a model
ollama serve
ollama pull phi4:latest
```

---

## First run

```bash
# Check that everything is installed
ai-video-analyzer doctor

# Run detection only (fast, no optional deps needed)
ai-video-analyzer analyze "/path/to/video.mp4" \
    --preset balanced \
    --no-captioning \
    --no-audio-events \
    --no-summarization

# Full analysis (requires full install + Ollama)
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

Checks: Python version, OpenCV, VisionServeX, D-FINE model registry, ffmpeg, PyTorch/GPU, Whisper, BLIP, PANNs, Ollama, moviepy, Tesseract.

Required dependencies exit with `✗`. Optional dependencies show `✓` with an install hint when missing.
Exits 0 if all required dependencies are present.

---

## Detection presets

VisionServeX D-FINE models (COCO-80 classes, benchmarked on RTX 5080):

| Preset | Model | ~ms/frame | Notes |
|---|---|---|---|
| `fast` | dfine-n | 17–21 | Speed-first; good accuracy |
| `balanced` | dfine-s | 17–27 | **Default** — best all-round |
| `quality` | dfine-m | 21–32 | Highest COCO accuracy |
| `quality+` | dfine-l | higher | Maximum accuracy, slowest |

```bash
ai-video-analyzer analyze video.mp4 --preset fast
ai-video-analyzer analyze video.mp4 --preset quality

# Override with a specific model ID
ai-video-analyzer analyze video.mp4 --model dfine-s

# List all available models
ai-video-analyzer list-models
```

**Known limitation**: COCO-80 does not include fire, smoke, or weather classes.
D-FINE will return approximate visual matches (e.g. food textures for fire/smoke frames).
The BLIP captioning stage handles these cases in natural language.

---

## Output files

Each run writes to `--output-dir` (default: current directory):

| File | Contents |
|---|---|
| `<video>_analysis.json` | Full structured data: detections, captions, transcript, timings, preset |
| `<video>_analysis.md` | Human-readable Markdown report with tables |
| `report.txt` | Plain-text legacy report |

The JSON report includes: `preset`, `frame_strategy`, `timings`, `top labels`, and `limitations`.

---

## Benchmarking

```bash
# Single preset
ai-video-analyzer benchmark "/path/to/video.mp4" --preset balanced

# Compare fast / balanced / quality side-by-side
ai-video-analyzer benchmark "/path/to/video.mp4" --compare
```

Benchmark output:

```
  Video duration     : 8.2s
  Frames selected    : 8  (sampling: 0.08s)
  Model load+warmup  : 0.52s
  Detection runtime  : 0.15s  (18.6ms/frame, 53.8fps)
  Total detections   : 48
  Top labels         : dog(9), person(7)
  Model              : dfine-s
```

Real-world results in `reports/benchmarks/`.

---

## Optional: LLM summarization

When Ollama is running, the pipeline generates a factual evidence-grounded summary.

```bash
# Start Ollama
ollama serve
ollama pull phi4:latest

# Run with summarization
ai-video-analyzer analyze video.mp4 --preset balanced --ollama-model phi4:latest

# List installed models
ai-video-analyzer analyze video.mp4 --list-ollama-models
```

The summary prompt separates **observed facts** from **plausible interpretation**, and says "Insufficient evidence" instead of guessing when data is weak.

---

## Troubleshooting

### VisionServeX not found

```bash
pip install 'visionservex[hf,rfdetr]'
# or
pip install -e ".[vision]"
```

### ffmpeg not found

```bash
sudo apt install ffmpeg    # Ubuntu/Debian
brew install ffmpeg        # macOS
```

### GPU not detected

```bash
python -c "import torch; print(torch.cuda.is_available())"
ai-video-analyzer analyze video.mp4 --device cpu
```

### No objects detected

Run doctor to check the detection backend:

```bash
ai-video-analyzer doctor
```

If `dfine-s` shows as unavailable, try listing what is registered:

```bash
ai-video-analyzer list-models
```

### Run a full diagnostics check

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
ai-video-analyzer gui
```

**Legacy optional backend**: YOLO-based detection is available as `--backend legacy_yolo`
with the `[legacy-yolo]` extra (`pip install -e ".[legacy-yolo]"`).
This is not recommended for new use. VisionServeX D-FINE is the default and supported backend.

---

## Pipeline overview

```
Video file
    │
    ├─ Adaptive frame sampling  (scene + motion aware)
    │
    ├─ Object detection         → VisionServeX D-FINE (per frame)
    ├─ Scene captioning         → BLIP (optional)
    │
    ├─ Audio extraction
    │       ├─ Transcription    → Whisper (optional)
    │       └─ Audio events     → PANNs CNN14 (optional)
    │
    └─ LLM summarization        → Ollama (optional)
            │
            └─ AnalysisReport
                    ├─ <video>_analysis.json
                    ├─ <video>_analysis.md
                    └─ report.txt
```

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

## For coding agents

See [`docs/LLM_AGENT_GUIDE.md`](docs/LLM_AGENT_GUIDE.md) for the complete agent guide.

Quick reference:

- Install: `python -m pip install -e ".[vision]"`
- Test: `pytest -q`
- Doctor: `ai-video-analyzer doctor`
- No media committed — videos and model weights are local-only
- VisionServeX reference at `/home/arash/PycharmProjects/VisionServeX` is **read-only**
- Config: `ai_powered_video_analyzer/config.py` — `AnalysisConfig` dataclass
- Pipeline entry: `ai_powered_video_analyzer/core.py` — `analyze_video(config)`
- Detection backend: `ai_powered_video_analyzer/backends/visionservex_backend.py`

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
