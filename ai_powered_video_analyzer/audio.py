"""Audio extraction and event detection (PANNs)."""

from __future__ import annotations

import os
import shutil
import tempfile

from ai_powered_video_analyzer.logging_utils import get_logger, timed_stage

log = get_logger(__name__)

# Confidence threshold for PANNs audio event labels
_PANNS_MIN_CONFIDENCE = 0.1
_SEGMENT_SECONDS = 5


def extract_audio(video_path: str, output_path: str) -> bool:
    """Extract audio track to WAV. Returns False if no audio track exists."""
    try:
        from moviepy.editor import VideoFileClip  # type: ignore[import]
    except ImportError:
        log.warning("moviepy not installed; skipping audio extraction.")
        return False

    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            log.info("Video has no audio track: %s", video_path)
            clip.reader.close()
            return False
        clip.audio.write_audiofile(output_path, logger=None)
        clip.reader.close()
        clip.audio.reader.close_proc()
        return True
    except Exception as exc:
        log.error("Audio extraction failed: %s", exc)
        return False


def detect_audio_events(
    audio_path: str, pann_model_path: str
) -> dict[str, list[str]]:
    """
    Detect audio events using PANNs CNN14.

    Returns a dict mapping event labels to lists of HH:MM:SS timestamps.
    Returns {'unavailable': []} if the model file or library is missing.
    """
    if not os.path.exists(pann_model_path):
        log.warning("PANNs model not found at %s; skipping audio event detection.", pann_model_path)
        return {"model_not_found": []}

    try:
        import librosa  # type: ignore[import]
        import numpy as np
        import torch  # type: ignore[import]
        from panns_inference import AudioTagging, labels as pann_labels  # type: ignore[import]
    except ImportError as exc:
        log.warning("PANNs dependency missing (%s); skipping audio event detection.", exc)
        return {"unavailable": []}

    try:
        with timed_stage("audio_events", log):
            waveform, sr = librosa.load(audio_path, sr=32000)
            if np.max(np.abs(waveform)) > 0:
                waveform = waveform / np.max(np.abs(waveform))

            panns_model = AudioTagging(checkpoint_path=pann_model_path)
            seg_len = _SEGMENT_SECONDS * sr
            events: dict[str, list[str]] = {}

            for i in range(0, len(waveform), seg_len):
                segment = waveform[i : i + seg_len]
                if len(segment) == 0:
                    continue
                tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
                output = panns_model.inference(tensor)

                if isinstance(output, dict):
                    clipwise = np.array(output.get("clipwise_output", []), dtype=float)
                else:
                    clipwise = np.array(output, dtype=float)

                if clipwise.size == 0 or np.max(clipwise) < _PANNS_MIN_CONFIDENCE:
                    continue

                top_idx = int(np.argmax(clipwise))
                label = pann_labels[top_idx] if top_idx < len(pann_labels) else "Unknown"
                ts = _sec_to_hhmmss(i / sr)
                events.setdefault(label, []).append(ts)

            return events if events else {"silence": []}
    except Exception as exc:
        log.error("Audio event detection error: %s", exc)
        return {"error": []}


def _sec_to_hhmmss(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
