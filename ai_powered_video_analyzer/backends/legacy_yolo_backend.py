"""
Legacy YOLO/Ultralytics backend — kept for backward compatibility only.

This module is NO LONGER part of the default detection pipeline.
Since v0.3.0, the primary backend is VisionServeX with D-FINE/RF-DETR models
(see visionservex_backend.py).

Why VisionServeX replaced YOLO as the default:
- D-FINE models (dfine-s, dfine-m) are faster and more accurate on benchmarks.
- VisionServeX provides a unified, license-aware, local-first model gateway.
- Multiple detector families are available without changing code.
- No separate ultralytics package dependency required.

To use this legacy backend (not recommended):
    from ai_powered_video_analyzer.backends.legacy_yolo_backend import LegacyYOLOBackend
    backend = LegacyYOLOBackend(model_id="yolo11x.pt")

Ultralytics is NOT a package dependency; install it separately if needed:
    pip install ultralytics
"""

from __future__ import annotations

import numpy as np

from ai_powered_video_analyzer.backends.base import BaseBackend, Detection
from ai_powered_video_analyzer.logging_utils import get_logger

log = get_logger(__name__)

# COCO class names (kept for reference)
_COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

# Keep old name alias for any code that imported YOLOBackend from this path
YOLOBackend = None  # replaced — see LegacyYOLOBackend below


class LegacyYOLOBackend(BaseBackend):
    """Legacy YOLO backend via ultralytics. Not part of the default pipeline since v0.3.0."""

    backend_name = "legacy_yolo"

    def __init__(self, model_id: str = "yolo11x.pt", device: str = "auto") -> None:
        log.warning(
            "LegacyYOLOBackend: Ultralytics YOLO is no longer the default detector. "
            "Use VisionServeX: --backend visionservex --detector-preset balanced"
        )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. This legacy backend requires it. "
                "Alternatively, use the VisionServeX backend: pip install 'visionservex[hf,rfdetr]'"
            ) from exc

        self.model_id = model_id
        log.info("Loading legacy YOLO model: %s", model_id)
        self._model = YOLO(model_id)

    def predict(self, frame: np.ndarray, confidence: float = 0.3) -> list[Detection]:
        results = self._model(frame, conf=confidence, verbose=False)
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None or r.boxes.data is None:
                continue
            for det in r.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = det
                label = _COCO_CLASSES.get(int(cls), f"class_{int(cls)}")
                detections.append(
                    Detection(
                        label=label, score=float(conf),
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        backend=self.backend_name, model_id=self.model_id,
                    )
                )
        return detections

    def is_available(self) -> bool:
        try:
            import ultralytics  # noqa: F401
            return True
        except ImportError:
            return False
