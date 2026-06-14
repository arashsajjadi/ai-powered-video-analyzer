"""Configuration dataclasses and defaults for the video analyzer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

FrameStrategy = Literal["uniform", "adaptive", "scene_change", "motion_aware", "hybrid"]
BackendName = Literal["auto", "yolo", "visionservex", "none"]
DeviceName = Literal["auto", "cpu", "cuda", "mps"]


def default_pann_path() -> str:
    """Return the best-guess path for the CNN14 PANNs checkpoint."""
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

    # Detection backend
    backend: BackendName = "auto"
    detector_model: str = ""
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
    ollama_model: str = "phi4:latest"
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
