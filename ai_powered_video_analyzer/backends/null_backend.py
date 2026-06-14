"""No-op detection backend for testing and graceful degradation."""

from __future__ import annotations

import numpy as np

from ai_powered_video_analyzer.backends.base import BaseBackend, Detection


class NullBackend(BaseBackend):
    """Returns empty detections — useful when no detector is available."""

    model_id = "null"
    backend_name = "null"

    def predict(self, frame: np.ndarray, confidence: float = 0.3) -> list[Detection]:
        return []

    def is_available(self) -> bool:
        return True
