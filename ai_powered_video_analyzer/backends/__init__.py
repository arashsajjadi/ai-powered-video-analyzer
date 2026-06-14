"""
Detection backend system for ai-powered-video-analyzer.

Primary backend since v0.3.0: VisionServeX (D-FINE / RF-DETR families).
Ultralytics/YOLO has been moved to legacy_yolo_backend.py and is NOT part
of the default pipeline.

Backend selection priority for `--backend auto`:
  1. VisionServeX   — if installed
  2. Null           — returns empty detections (non-strict mode)

Explicit backend selection:
  --backend visionservex   (preferred)
  --backend none           (disables detection)
  --backend legacy_yolo    (requires ultralytics, not recommended)
"""

from __future__ import annotations

from ai_powered_video_analyzer.backends.base import BaseBackend, Detection
from ai_powered_video_analyzer.backends.null_backend import NullBackend

__all__ = [
    "BaseBackend",
    "Detection",
    "NullBackend",
    "load_backend",
    "PRESET_MODELS",
]

# Exported preset → model mapping for documentation and CLI help
PRESET_MODELS: dict[str, str] = {
    "fast": "dfine-n",
    "balanced": "dfine-s",
    "quality": "dfine-m",
    "quality+": "dfine-l",
}


def load_backend(
    name: str,
    model_id: str = "",
    preset: str = "",
    device: str = "auto",
) -> BaseBackend:
    """
    Load and return the requested detection backend.

    Parameters
    ----------
    name    : "auto" | "visionservex" | "none" | "legacy_yolo"
    model_id: explicit VisionServeX model ID (e.g. "dfine-s")
    preset  : detector preset (fast/balanced/quality/quality+)
    device  : device string passed to the backend

    For 'auto': selects VisionServeX if installed, else Null.
    """
    from ai_powered_video_analyzer.logging_utils import get_logger
    log = get_logger(__name__)

    def _try_vsx() -> BaseBackend | None:
        try:
            from ai_powered_video_analyzer.backends.visionservex_backend import VisionServeXBackend
            b = VisionServeXBackend(model_id=model_id, preset=preset or "balanced", device=device)
            log.info("VisionServeX backend active (model=%s)", b.model_id)
            return b
        except ImportError:
            log.warning(
                "VisionServeX is not installed. Install for object detection:\n"
                "  pip install 'visionservex[hf,rfdetr]'"
            )
            return None
        except Exception as exc:
            log.warning("VisionServeX backend failed to load: %s", exc)
            return None

    if name == "visionservex":
        b = _try_vsx()
        if b is None:
            raise ImportError(
                "VisionServeX backend requested but not installed or failed to load.\n"
                "Install with: pip install 'visionservex[hf,rfdetr]'"
            )
        return b

    if name == "none":
        log.info("Null backend selected — object detection disabled.")
        return NullBackend()

    if name == "legacy_yolo":
        log.warning("Legacy YOLO backend requested. This is not the recommended path since v0.3.0.")
        try:
            from ai_powered_video_analyzer.backends.legacy_yolo_backend import LegacyYOLOBackend
            return LegacyYOLOBackend(model_id=model_id or "yolo11x.pt", device=device)
        except ImportError as exc:
            raise ImportError(
                "Legacy YOLO backend requires ultralytics: pip install ultralytics\n"
                "Recommended alternative: pip install 'visionservex[hf,rfdetr]'"
            ) from exc

    if name in ("yolo",):
        log.warning(
            "'--backend yolo' is deprecated since v0.3.0 and redirects to legacy_yolo. "
            "Use '--backend visionservex' instead."
        )
        return load_backend("legacy_yolo", model_id=model_id, device=device)

    # auto
    if name != "auto":
        log.warning("Unknown backend '%s', falling back to auto.", name)

    b = _try_vsx()
    if b is not None:
        return b

    log.warning(
        "No detection backend available. Object detection disabled. "
        "Install VisionServeX to enable detection: pip install 'visionservex[hf,rfdetr]'"
    )
    return NullBackend()
