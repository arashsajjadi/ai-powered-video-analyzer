"""YOLO detection backend using ultralytics."""

from __future__ import annotations

import numpy as np

from ai_powered_video_analyzer.backends.base import BaseBackend, Detection
from ai_powered_video_analyzer.logging_utils import get_logger

log = get_logger(__name__)

# COCO class names used by YOLOv8/v11
COCO_CLASSES = {
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


class YOLOBackend(BaseBackend):
    """Wraps ultralytics YOLO for detection."""

    backend_name = "yolo"

    def __init__(self, model_id: str = "yolo11x.pt", device: str = "auto") -> None:
        from ultralytics import YOLO  # deferred import

        self.model_id = model_id
        self._device = _resolve_device(device)
        log.info("Loading YOLO model: %s on %s", model_id, self._device)
        self._model = YOLO(model_id)

    def predict(self, frame: np.ndarray, confidence: float = 0.3) -> list[Detection]:
        results = self._model(frame, conf=confidence, verbose=False)
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None or r.boxes.data is None:
                continue
            for det in r.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = det
                label = COCO_CLASSES.get(int(cls), f"class_{int(cls)}")
                detections.append(
                    Detection(
                        label=label,
                        score=float(conf),
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        backend=self.backend_name,
                        model_id=self.model_id,
                    )
                )
        return detections

    def predict_batch(
        self, frames: list[np.ndarray], confidence: float = 0.3
    ) -> list[list[Detection]]:
        results_all = self._model(frames, conf=confidence, verbose=False)
        out: list[list[Detection]] = []
        for r in results_all:
            frame_dets: list[Detection] = []
            if r.boxes is not None and r.boxes.data is not None:
                for det in r.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, cls = det
                    label = COCO_CLASSES.get(int(cls), f"class_{int(cls)}")
                    frame_dets.append(
                        Detection(
                            label=label,
                            score=float(conf),
                            x1=float(x1), y1=float(y1),
                            x2=float(x2), y2=float(y2),
                            backend=self.backend_name,
                            model_id=self.model_id,
                        )
                    )
            out.append(frame_dets)
        return out

    def is_available(self) -> bool:
        try:
            import ultralytics  # noqa: F401
            return True
        except ImportError:
            return False


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
