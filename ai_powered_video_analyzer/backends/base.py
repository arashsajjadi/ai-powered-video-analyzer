"""Abstract base class for all detection backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class Detection:
    """Normalized detection output shared across all backends."""

    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float
    frame_index: int = 0
    timestamp_sec: float = 0.0
    backend: str = ""
    model_id: str = ""

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "bbox_xyxy": [round(self.x1, 1), round(self.y1, 1), round(self.x2, 1), round(self.y2, 1)],
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "backend": self.backend,
            "model_id": self.model_id,
        }

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


class BaseBackend(ABC):
    """Contract for all detection backends."""

    model_id: str = ""
    backend_name: str = ""

    @abstractmethod
    def predict(self, frame: np.ndarray, confidence: float = 0.3) -> list[Detection]:
        """Run inference on a single BGR frame, return normalized detections."""

    def predict_batch(
        self, frames: list[np.ndarray], confidence: float = 0.3
    ) -> list[list[Detection]]:
        """Batch inference; default implementation calls predict() sequentially."""
        return [self.predict(f, confidence) for f in frames]

    def warmup(self) -> None:
        """Optional warmup call before processing starts."""

    def unload(self) -> None:
        """Release model resources."""

    def is_available(self) -> bool:
        """Return True if this backend can actually run."""
        return True

    @property
    def info(self) -> dict:
        return {"backend": self.backend_name, "model_id": self.model_id}
