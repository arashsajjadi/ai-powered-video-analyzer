"""Optional VisionServeX detection backend.

Install the backend:
    pip install 'visionservex[hf,rfdetr]'

Usage:
    from ai_powered_video_analyzer.backends import load_backend
    backend = load_backend("visionservex", model_id="rf-detr-base")
"""

from __future__ import annotations

import numpy as np

from ai_powered_video_analyzer.backends.base import BaseBackend, Detection
from ai_powered_video_analyzer.logging_utils import get_logger

log = get_logger(__name__)


class VisionServeXBackend(BaseBackend):
    """Wraps VisionServeX VisionModel for detection tasks."""

    backend_name = "visionservex"

    def __init__(self, model_id: str = "mock-detect", device: str = "auto") -> None:
        try:
            from visionservex import VisionModel  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "VisionServeX is not installed. "
                "Install with: pip install 'visionservex[hf,rfdetr]'"
            ) from exc

        self.model_id = model_id
        resolved_device = None if device == "auto" else device
        log.info("Loading VisionServeX model: %s (device=%s)", model_id, device)
        self._model = VisionModel(model_id, device=resolved_device)

    def predict(self, frame: np.ndarray, confidence: float = 0.3) -> list[Detection]:
        from PIL import Image  # type: ignore[import]

        pil_image = Image.fromarray(frame[:, :, ::-1])  # BGR → RGB
        try:
            result = self._model.predict(pil_image)
        except Exception as exc:
            log.warning("VisionServeX predict failed: %s", exc)
            return []

        return self._parse_result(result, confidence)

    def _parse_result(self, result: object, confidence: float) -> list[Detection]:
        detections: list[Detection] = []
        raw_dets = getattr(result, "detections", None)
        if raw_dets is None:
            return detections
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
                    backend=self.backend_name,
                    model_id=self.model_id,
                )
            )
        return detections

    def warmup(self) -> None:
        if hasattr(self._model, "warmup"):
            self._model.warmup()

    def unload(self) -> None:
        if hasattr(self._model, "unload"):
            self._model.unload()

    def is_available(self) -> bool:
        try:
            import visionservex  # noqa: F401 # type: ignore[import]
            return True
        except ImportError:
            return False
