# User Guide — AI-Powered Video Analyzer

This guide walks you through everything from first install to advanced usage.
No prior machine learning experience required.

---

## 1. What is this project?

AI-Powered Video Analyzer takes a video file and produces a structured report about what it contains:

- Objects detected in each frame (person, car, dog, etc.)
- Natural-language scene descriptions
- Speech transcription (if audio is present)
- Audio event tags (speech, music, noise, etc.)
- A written summary of the whole video

All processing runs **locally on your computer**. Nothing is sent to the internet.

---

## 2. How offline/local privacy works

Every stage of the pipeline uses locally-installed software:

| What | Where it runs |
|---|---|
| Object detection | VisionServeX (local Python package) |
| Scene captioning | BLIP model (downloaded once, runs locally) |
| Speech transcription | Whisper model (downloaded once, runs locally) |
| Audio events | PANNs CNN14 checkpoint (you download manually) |
| LLM summarization | Ollama (local server, models stored on your machine) |

Your video never leaves your machine.

---

## 3. Installation from scratch

### Prerequisites

- Python 3.10, 3.11, 3.12, or 3.13
- Git
- pip (comes with Python)

### Step 1: Clone the repository

```bash
git clone https://github.com/arashsajjadi/ai-powered-video-analyzer.git
cd ai-powered-video-analyzer
```

### Step 2: Upgrade pip

```bash
python -m pip install -U pip
```

### Step 3: Install the package

For detection only (simplest):

```bash
python -m pip install -e ".[vision]"
```

For the full pipeline (detection + captioning + transcription + audio):

```bash
python -m pip install -r pip_requirements.txt
```

---

## 4. Installing VisionServeX

VisionServeX is the primary detection backend. It is installed automatically with the `[vision]` extra:

```bash
python -m pip install -e ".[vision]"
```

If you need to install it separately:

```bash
pip install "visionservex[hf,rfdetr]"
```

Verify it is detected:

```bash
ai-video-analyzer doctor
```

You should see a `✓` next to `VisionServeX` and `dfine-s (balanced preset)`.

---

## 5. How to run doctor

The `doctor` command checks every dependency and tells you what is installed and what is missing:

```bash
ai-video-analyzer doctor
```

Example output:

```
=== ai-video-analyzer doctor ===

  ✓  Python                                 3.13.12
  ✓  OpenCV (cv2)                           4.13.0
  ✓  VisionServeX                           3.11.0 (primary detection backend)
  ✓    dfine-s (balanced preset)            status=beta, impl=wired
  ✓  ffmpeg                                 found
  ✓  Whisper (optional)                     not installed — install for transcription: pip install openai-whisper
  ✓  Ollama                                 3 model(s): phi4:latest

Core dependencies OK.

Next step:
  ai-video-analyzer analyze "/path/to/video.mp4" --preset balanced
```

Items marked `✓` with an optional note are working fine. Only items marked `✗` are real failures.

---

## 6. Analyzing a first video

```bash
ai-video-analyzer analyze "/path/to/your/video.mp4" --preset balanced
```

This runs object detection on the video and writes two output files to the current directory:

- `<video-name>_analysis.json` — full structured data
- `<video-name>_analysis.md` — human-readable Markdown report

The terminal will print:

```
Analysis complete.
  Video     : /path/to/your/video.mp4
  Duration  : 12.4s
  Frames    : 12 analyzed / 372 total
  Preset    : balanced  model=dfine-s
  Detections: 87
  Captions  : 0

Output files:
  JSON    : ./video_analysis.json
  Markdown: ./video_analysis.md
```

---

## 7. Fast detection-only mode

Skip all optional stages for the fastest possible run:

```bash
ai-video-analyzer analyze "/path/to/video.mp4" \
    --preset fast \
    --max-frames 40 \
    --no-captioning \
    --no-audio-events \
    --no-summarization
```

This requires only the `[vision]` install (`pip install -e ".[vision]"`) and runs in seconds.

---

## 8. Full analysis mode

Run all stages including captioning, transcription, and LLM summarization:

```bash
# Requirements: pip install -r pip_requirements.txt  +  ollama serve
ai-video-analyzer analyze "/path/to/video.mp4" \
    --preset balanced \
    --whisper-model base \
    --ollama-model phi4:latest \
    --output-dir ./results
```

All outputs go to `./results/`.

---

## 9. Output file explanation

### JSON report (`<video>_analysis.json`)

Contains everything in machine-readable form:

```json
{
  "video_path": "/path/to/video.mp4",
  "duration_sec": 12.4,
  "fps": 29.97,
  "sampled_frame_count": 12,
  "preset": "balanced",
  "frame_strategy": "adaptive",
  "timings": {"frame_sampling": 0.08, "object_detection": 0.23},
  "detections": [
    {"label": "person", "score": 0.92, "timestamp_sec": 1.0, ...},
    ...
  ],
  "captions": [...],
  "transcript": "...",
  "summary": "...",
  "limitations": [...]
}
```

### Markdown report (`<video>_analysis.md`)

Human-readable with:
- Run configuration (preset, model, frame strategy, timings)
- Detected objects table with counts and max confidence
- Scene captions table with timestamps
- Speech transcript
- Audio events
- Limitations and notes

---

## 10. Detection presets

| Preset | Model | Speed | Accuracy | When to use |
|---|---|---|---|---|
| `fast` | dfine-n | ~52 fps | Good | Real-time or long videos |
| `balanced` | dfine-s | ~37 fps | Very good | **Default** |
| `quality` | dfine-m | ~47 fps | Best COCO | Short videos, critical accuracy |
| `quality+` | dfine-l | slower | Maximum | Highest possible accuracy |

All presets use COCO-80 classes (80 common object categories).

```bash
# Change preset
ai-video-analyzer analyze video.mp4 --preset fast
ai-video-analyzer analyze video.mp4 --preset quality

# Override with a specific model from the registry
ai-video-analyzer analyze video.mp4 --model dfine-s

# List what models are available
ai-video-analyzer list-models
```

---

## 11. What to do when detections are weak

Detection quality can be limited by:

1. **Low confidence threshold** — lower the threshold to see more detections:
   ```bash
   ai-video-analyzer analyze video.mp4 --confidence 0.2
   ```

2. **COCO-80 limitation** — fire, smoke, weather, food textures for fire are expected misclassifications.
   Use BLIP captioning for these subjects:
   ```bash
   ai-video-analyzer analyze video.mp4 --preset balanced
   # (captioning is on by default — reads scene descriptions)
   ```

3. **Too few frames** — increase frame sampling:
   ```bash
   ai-video-analyzer analyze video.mp4 --target-fps 2.0
   ```

4. **Wrong device** — force GPU if not auto-detected:
   ```bash
   ai-video-analyzer analyze video.mp4 --device cuda
   ```

---

## 12. How BLIP captioning helps

BLIP generates natural-language descriptions of individual frames. This is useful when:

- The video contains fire, smoke, weather, or other content not in COCO-80
- You want written scene descriptions alongside object labels
- You want to supplement detection with qualitative context

BLIP captioning is enabled by default. To disable it (faster run):

```bash
ai-video-analyzer analyze video.mp4 --no-captioning
```

Requires `pip install -r pip_requirements.txt` (includes `transformers`).

---

## 13. How Ollama summarization works

After detection and captioning run, the pipeline constructs a structured text input:

```
Detected Objects: person(9), car(5), dog(3)
Scene Captions:
- A person walking a dog on a leash
- A car parked near a tree
```

This is sent to a local Ollama model with a prompt that asks for:
- **Observed facts**: what the data explicitly shows
- **Interpretation**: what is likely happening (labeled as such)
- An honest "Insufficient evidence" response when data is weak

Ollama must be running:

```bash
ollama serve
```

---

## 14. How to choose a local LLM

Recommended models for summarization:

| Model | Quality | Size | Command |
|---|---|---|---|
| `phi4:latest` | Very good | ~8GB | `ollama pull phi4:latest` |
| `llama3:latest` | Good | ~4GB | `ollama pull llama3` |
| `qwen2.5:7b` | Good | ~4GB | `ollama pull qwen2.5:7b` |

Auto-discovery selects the first available model matching a preference list.
Override with `--ollama-model`:

```bash
ai-video-analyzer analyze video.mp4 --ollama-model llama3:latest
```

---

## 15. How to benchmark

Run a speed benchmark to measure detection performance on your hardware:

```bash
# Single preset
ai-video-analyzer benchmark "/path/to/video.mp4" --preset balanced

# Compare all three main presets
ai-video-analyzer benchmark "/path/to/video.mp4" --compare
```

The benchmark measures: frame sampling time, model load+warmup time, detection time per frame, and top detected labels.

---

## 16. Common errors and fixes

### `Error: VisionServeX backend requested but not installed`

```bash
pip install -e ".[vision]"
# or
pip install "visionservex[hf,rfdetr]"
```

### `Error: Video file not found`

Check the path:

```bash
ls "/path/to/video.mp4"
```

### `No objects detected`

Run doctor and verify VisionServeX and dfine-s are OK:

```bash
ai-video-analyzer doctor
```

### `Ollama timed out`

Check Ollama is running and the model is pulled:

```bash
ollama serve
ollama list
ollama pull phi4:latest
```

### `ffmpeg not found`

```bash
sudo apt install ffmpeg     # Ubuntu/Debian
brew install ffmpeg         # macOS
```

---

## 17. Privacy: what not to upload or commit

This project is built for **local, offline use**. Follow these rules when working with the code:

- **Do not commit videos**: `.mp4`, `.mov`, `.avi`, etc.
- **Do not commit model weights**: `.pt`, `.pth`, `.onnx`, `.bin`, `.safetensors`
- **Do not commit screenshots or frames**: extracted images from private video
- **Do not commit private audio**: `.wav` extracted from personal recordings
- **Do not commit API keys or `.env` files**

The `.gitignore` in this repo covers all the above.

---

## 18. Known limitations

| Limitation | Details |
|---|---|
| COCO-80 only | D-FINE detects 80 common categories. Fire, smoke, weather are not included. |
| PANNs requires manual setup | Download CNN14 checkpoint to `models/cnn14.pth` from the PANNs GitHub. |
| Whisper memory usage | Large Whisper models need 4–8 GB RAM. Use `--whisper-model base` for low-memory setups. |
| BLIP is slow without GPU | BLIP captioning may be slow on CPU-only setups. Disable with `--no-captioning`. |
| Ollama timeout | Long videos or slow LLMs may exceed the 120s timeout. Use smaller models or shorter videos. |
| `quality+` (dfine-l) | Benchmarked results for dfine-l are not included. Slower than dfine-m with marginal gain on most videos. |
