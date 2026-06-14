"""Vision detection backends for ai-powered-video-analyzer."""

from __future__ import annotations

from ai_powered_video_analyzer.backends.base import BaseBackend, Detection
from ai_powered_video_analyzer.backends.null_backend import NullBackend

__all__ = ["BaseBackend", "Detection", "NullBackend", "load_backend"]


def load_backend(name: str, model_id: str = "", device: str = "auto") -> "BaseBackend":
    """
    Factory that loads the requested backend, falling back gracefully.

    Priority for 'auto':
      1. visionservex if installed
      2. yolo if ultralytics is installed
      3. null (no detection, non-strict mode only)
    """
    from ai_powered_video_analyzer.logging_utils import get_logger

    log = get_logger(__name__)

    def _try_vsx() -> "BaseBackend | None":
        try:
            from ai_powered_video_analyzer.backends.visionservex_backend import VisionServeXBackend
            b = VisionServeXBackend(model_id=model_id or "mock-detect", device=device)
            log.info("Using VisionServeX backend (model=%s)", b.model_id)
            return b
        except ImportError:
            log.debug("VisionServeX not installed, skipping.")
            return None
        except Exception as exc:
            log.warning("VisionServeX backend unavailable: %s", exc)
            return None

    def _try_yolo() -> "BaseBackend | None":
        try:
            from ai_powered_video_analyzer.backends.yolo_backend import YOLOBackend
            b = YOLOBackend(model_id=model_id or "yolo11x.pt", device=device)
            log.info("Using YOLO backend (model=%s)", b.model_id)
            return b
        except ImportError:
            log.debug("ultralytics not installed, skipping.")
            return None
        except Exception as exc:
            log.warning("YOLO backend unavailable: %s", exc)
            return None

    if name == "visionservex":
        b = _try_vsx()
        if b is None:
            raise ImportError(
                "VisionServeX backend requested but not installed. "
                "Install with: pip install 'visionservex[hf,rfdetr]'"
            )
        return b

    if name == "yolo":
        b = _try_yolo()
        if b is None:
            raise ImportError(
                "YOLO backend requested but ultralytics is not installed. "
                "Install with: pip install ultralytics"
            )
        return b

    if name == "none":
        log.info("Using null detection backend (no object detection).")
        return NullBackend()

    # auto
    b = _try_vsx() or _try_yolo()
    if b is None:
        log.warning(
            "No detection backend available. Object detection disabled. "
            "Install ultralytics or visionservex to enable detection."
        )
        return NullBackend()
    return b
