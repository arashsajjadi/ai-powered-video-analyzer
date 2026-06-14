"""
VisionServeX detection backend — primary object detector since v0.3.0.

Supports D-FINE and RF-DETR model families through VisionServeX:
  dfine-n / dfine-s / dfine-m / dfine-l / dfine-x
  rfdetr-nano / rfdetr-small / rfdetr-medium / rfdetr-base / rfdetr-large

Install:
    pip install "visionservex[hf,rfdetr]"

Usage:
    from ai_powered_video_analyzer.backends import load_backend
    backend = load_backend("visionservex", preset="balanced")

Evidence from real-video benchmarks (RTX 5080, 2026-06-14):
  dfine-s (~13ms/frame): dog(0.67), person(0.83) on dog video  ✓
  dfine-m (~15ms/frame): dog(0.88), person(0.90) on dog video  ✓
  rfdetr-nano (~14ms/frame): bicycle/sheep/horse on dog video   ✗ (hallucinated)

Preset → model_id mapping:
  fast      → dfine-n   (smallest D-FINE, ~16ms/frame on CUDA)
  balanced  → dfine-s   (default, ~13ms/frame on CUDA, good accuracy)
  quality   → dfine-m   (better confidence, ~15ms/frame on CUDA)
  quality+  → dfine-l   (large D-FINE, slower, highest COCO accuracy)

Limitation: All presets use COCO-80 classes. Abstract/conceptual categories
(fire, smoke, weather) are not directly detectable; the captioning stage
(BLIP) covers those cases.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from ai_powered_video_analyzer.backends.base import BaseBackend, Detection
from ai_powered_video_analyzer.logging_utils import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Preset → model_id mapping. Only core_load/wired models are listed here.
_PRESET_MODELS: dict[str, str] = {
    "fast": "dfine-n",
    "balanced": "dfine-s",
    "quality": "dfine-m",
    "quality+": "dfine-l",
}

_DEFAULT_PRESET = "balanced"
_DEFAULT_MODEL = _PRESET_MODELS[_DEFAULT_PRESET]

# Models known to work reliably in benchmarks
_RECOMMENDED_MODELS = frozenset(_PRESET_MODELS.values()) | {
    "dfine-x", "rfdetr-small", "rfdetr-medium", "rfdetr-base",
}


def resolve_model_id(model_id: str = "", preset: str = "") -> str:
    """Return the concrete model ID from either a direct model_id or a preset name."""
    if model_id:
        return model_id
    if preset:
        if preset not in _PRESET_MODELS:
            available = ", ".join(sorted(_PRESET_MODELS))
            raise ValueError(
                f"Unknown detector preset '{preset}'. Available: {available}"
            )
        return _PRESET_MODELS[preset]
    return _DEFAULT_MODEL


class VisionServeXBackend(BaseBackend):
    """
    Primary detection backend using VisionServeX VisionModel.

    Parameters
    ----------
    model_id : str
        Exact model ID from VisionServeX registry (e.g. "dfine-s").
        Overrides `preset` if both are given.
    preset : str
        One of fast/balanced/quality/quality+ — resolved to a model_id.
    device : str
        "auto" | "cpu" | "cuda" | "mps"
    """

    backend_name = "visionservex"

    def __init__(
        self,
        model_id: str = "",
        preset: str = _DEFAULT_PRESET,
        device: str = "auto",
    ) -> None:
        try:
            from visionservex import VisionModel  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "VisionServeX is not installed. Install with:\n"
                "  pip install 'visionservex[hf,rfdetr]'"
            ) from exc

        self.model_id = resolve_model_id(model_id, preset)
        self._preset = preset
        self._device_arg = device
        resolved_device = None if device == "auto" else device

        log.info(
            "Loading VisionServeX model: %s (preset=%s, device=%s)",
            self.model_id, preset, device,
        )
        t0 = time.perf_counter()
        self._model = VisionModel(self.model_id, device=resolved_device)
        load_time = time.perf_counter() - t0
        log.info("VisionServeX model loaded in %.2fs", load_time)
        self._load_time_sec = load_time

    def predict(self, frame: np.ndarray, confidence: float = 0.3) -> list[Detection]:
        """Run detection on a single BGR frame."""
        try:
            from PIL import Image  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Pillow is required: pip install Pillow") from exc

        pil_image = Image.fromarray(frame[:, :, ::-1])  # BGR → RGB
        try:
            result = self._model.predict(pil_image)
        except Exception as exc:
            log.warning("VisionServeX predict failed: %s", exc)
            return []

        return _parse_detections(result, confidence, self.backend_name, self.model_id)

    def predict_batch(
        self, frames: list[np.ndarray], confidence: float = 0.3
    ) -> list[list[Detection]]:
        """Run detection on a batch of BGR frames (sequentially — VisionServeX predict is already GPU-batched internally)."""
        return [self.predict(f, confidence) for f in frames]

    def warmup(self) -> None:
        """Pre-warm CUDA kernels with a synthetic frame."""
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.predict(dummy)
            log.debug("VisionServeX warmup complete.")
        except Exception as exc:
            log.debug("VisionServeX warmup failed (non-fatal): %s", exc)

    def unload(self) -> None:
        if hasattr(self._model, "unload"):
            self._model.unload()
        try:
            import gc
            gc.collect()
            import torch  # type: ignore[import]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def is_available(self) -> bool:
        try:
            import visionservex  # noqa: F401  # type: ignore[import]
            return True
        except ImportError:
            return False

    @property
    def info(self) -> dict:
        d = super().info
        d["preset"] = self._preset
        d["load_time_sec"] = round(self._load_time_sec, 3)
        return d


def _parse_detections(
    result: object,
    confidence: float,
    backend_name: str,
    model_id: str,
) -> list[Detection]:
    raw_dets = getattr(result, "detections", None)
    if raw_dets is None:
        return []
    detections: list[Detection] = []
    for det in raw_dets:
        score = float(getattr(det, "score", 0.0))
        if score < confidence:
            continue
        box = getattr(det, "box", None)
        if box is None:
            continue
        detections.append(
            Detection(
                label=str(getattr(det, "label", "object")),
                score=score,
                x1=float(box.x1),
                y1=float(box.y1),
                x2=float(box.x2),
                y2=float(box.y2),
                backend=backend_name,
                model_id=model_id,
            )
        )
    return detections


def list_available_detect_models() -> list[str]:
    """Return model IDs for all wired/detect models in the local VisionServeX registry."""
    try:
        from visionservex.registry import default_registry  # type: ignore[import]
        reg = default_registry()
        return sorted(
            m.id for m in reg.list()
            if getattr(m, "task", "") == "detect"
            and getattr(m, "implementation_status", "") == "wired"
        )
    except Exception as exc:
        log.debug("Could not query VisionServeX registry: %s", exc)
        return []
