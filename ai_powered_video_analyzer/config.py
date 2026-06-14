"""Configuration dataclasses and defaults for the video analyzer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

FrameStrategy = Literal["uniform", "adaptive", "scene_change", "motion_aware", "hybrid"]
BackendName = Literal["auto", "visionservex", "none", "legacy_yolo", "yolo"]
DeviceName = Literal["auto", "cpu", "cuda", "mps"]
DetectorPreset = Literal["fast", "balanced", "quality", "quality+"]
SummaryStyle = Literal["concise", "evidence", "technical", "narrative"]


def default_pann_path() -> str:
    candidates = [
        os.path.join("models", "cnn14.pth"),
        os.path.expanduser("~/panns_data/cnn14.pth"),
        "/app/models/cnn14.pth",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join("models", "cnn14.pth")


@dataclass
class AnalysisConfig:
    """Top-level configuration for a video analysis run."""

    video_path: str = ""
    output_dir: str = "."

    # Frame sampling
    frame_strategy: FrameStrategy = "adaptive"
    target_fps: float = 1.0
    min_fps: float = 0.25
    max_fps: float = 4.0
    scene_threshold: float = 30.0
    motion_threshold: float = 5.0
    max_frames: int = 2000

    # Detection backend — VisionServeX is the default since v0.3.0
    backend: BackendName = "visionservex"
    detector_model: str = ""
    detector_preset: DetectorPreset = "balanced"
    detection_confidence: float = 0.3

    # Device
    device: DeviceName = "auto"

    # Transcription
    whisper_model: str = "base"
    transcription_language: str | None = None

    # Captioning
    blip_model: str = "Salesforce/blip-image-captioning-base"
    enable_captioning: bool = True

    # Audio
    pann_model_path: str = ""
    enable_audio_events: bool = True

    # Summarization
    ollama_model: str = ""
    summary_style: SummaryStyle = "concise"
    enable_summarization: bool = True

    # Output
    save_json: bool = True
    save_markdown: bool = True
    save_annotated_video: bool = False

    # Behavior
    strict: bool = False
    debug: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        if not self.pann_model_path:
            self.pann_model_path = default_pann_path()
        # Default Ollama model: auto-discover from local installation
        if not self.ollama_model:
            self.ollama_model = _discover_ollama_model()


def _discover_ollama_model() -> str:
    """Return the first available Ollama model, or a safe default string."""
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            models = [ln.split()[0] for ln in lines[1:] if ln.strip()]
            if models:
                # Prefer known-good summarizers if present
                for preferred in ("phi4:latest", "phi4", "qwen:14b", "llama3"):
                    if any(m.startswith(preferred) for m in models):
                        return next(m for m in models if m.startswith(preferred))
                return models[0]
    except Exception:
        pass
    return "phi4:latest"
