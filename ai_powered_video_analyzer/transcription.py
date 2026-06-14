"""Speech transcription via OpenAI Whisper (local, offline)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ai_powered_video_analyzer.logging_utils import get_logger, timed_stage

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

_whisper_model_cache: dict[str, object] = {}


def load_whisper(model_name: str = "base") -> object:
    """Load (or return cached) Whisper model."""
    if model_name not in _whisper_model_cache:
        try:
            import whisper  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is not installed. Install with: pip install openai-whisper"
            ) from exc
        log.info("Loading Whisper model: %s", model_name)
        _whisper_model_cache[model_name] = whisper.load_model(model_name)
    return _whisper_model_cache[model_name]


def transcribe(
    audio_path: str,
    model_name: str = "base",
    language: str | None = None,
) -> tuple[str, str]:
    """
    Transcribe audio to text.

    Returns (text, detected_language). On failure returns ("", "unknown").
    """
    if not os.path.exists(audio_path):
        log.error("Audio file not found: %s", audio_path)
        return "", "unknown"

    # Map Tesseract-style language codes to ISO 639-1 codes that Whisper accepts
    _LANG_MAP = {"fas": "fa", "eng": "en", "spa": "es", "fra": "fr",
                 "deu": "de", "ara": "ar", "ita": "it", "jpn": "ja",
                 "kor": "ko", "rus": "ru"}
    whisper_lang = _LANG_MAP.get(language or "", language) if language else None

    try:
        with timed_stage("transcription", log):
            model = load_whisper(model_name)
            result = model.transcribe(audio_path, task="transcribe", language=whisper_lang)
            text = result.get("text", "")
            detected = result.get("language", "unknown") or "unknown"
            log.info("Transcription complete. Detected language: %s", detected)
            return text, detected
    except Exception as exc:
        log.error("Transcription failed: %s", exc)
        return "", "unknown"


def clear_cache() -> None:
    """Release loaded Whisper models from memory."""
    import gc
    _whisper_model_cache.clear()
    gc.collect()
    try:
        import torch  # type: ignore[import]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
