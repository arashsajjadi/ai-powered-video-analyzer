"""Image captioning via BLIP (Salesforce, local/offline)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from ai_powered_video_analyzer.logging_utils import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

_blip_cache: dict[str, tuple[object, object]] = {}


def load_blip(model_id: str = "Salesforce/blip-image-captioning-base") -> tuple[object, object]:
    """Load (or return cached) BLIP processor and model."""
    if model_id not in _blip_cache:
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor  # type: ignore[import]
            import torch  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. Install with: pip install transformers"
            ) from exc

        log.info("Loading BLIP model: %s", model_id)
        device = _get_device()
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id)
        model.to(device)  # type: ignore[arg-type]
        _blip_cache[model_id] = (processor, model)
    return _blip_cache[model_id]


def caption_frame(
    frame_bgr: np.ndarray,
    model_id: str = "Salesforce/blip-image-captioning-base",
) -> str | None:
    """
    Generate a natural-language caption for a BGR frame.

    Returns None on error or empty caption.
    """
    try:
        from PIL import Image  # type: ignore[import]
        import torch  # type: ignore[import]

        processor, model = load_blip(model_id)
        device = next(model.parameters()).device  # type: ignore[arg-type]

        rgb = frame_bgr[:, :, ::-1].copy()
        pil_img = Image.fromarray(rgb)
        inputs = processor(pil_img, return_tensors="pt").to(device)

        output_ids = model.generate(
            **inputs,
            max_length=60,
            min_length=20,
            repetition_penalty=1.05,
            num_beams=5,
            length_penalty=0.6,
        )
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        caption = _dedup_words(caption).strip()
        return caption if caption else None

    except Exception as exc:
        log.warning("BLIP caption failed: %s", exc)
        return None


def _dedup_words(text: str) -> str:
    """Remove adjacent duplicate words (case-insensitive)."""
    words = text.split()
    if not words:
        return text
    out = [words[0]]
    for w in words[1:]:
        if w.lower() != out[-1].lower():
            out.append(w)
    return " ".join(out)


def _get_device() -> str:
    try:
        import torch  # type: ignore[import]
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def clear_cache() -> None:
    _blip_cache.clear()
    try:
        import torch  # type: ignore[import]
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
