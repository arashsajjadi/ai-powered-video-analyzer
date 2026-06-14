"""Adaptive frame sampling for efficient video analysis.

Strategies
----------
uniform       - take every N-th frame based on target_fps
adaptive      - combine scene-change and motion detection heuristics
scene_change  - pure histogram-difference scene detection
motion_aware  - pure frame-difference motion detection
hybrid        - scene_change + motion_aware, deduplicated
"""

from __future__ import annotations

import dataclasses
from typing import Iterator

import numpy as np

from ai_powered_video_analyzer.logging_utils import get_logger

log = get_logger(__name__)


@dataclasses.dataclass
class FrameRecord:
    """A single sampled frame with metadata."""

    frame_index: int
    timestamp_sec: float
    frame: np.ndarray
    reason: str
    scene_score: float = 0.0
    motion_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "reason": self.reason,
            "scene_score": self.scene_score,
            "motion_score": self.motion_score,
        }


class FrameSampler:
    """
    Samples frames from a video according to the configured strategy.

    Parameters
    ----------
    strategy         : sampling strategy name
    target_fps       : desired output frame rate
    min_fps          : hard lower bound (ensures at least this many frames/sec)
    max_fps          : hard upper bound (prevents burst sampling)
    scene_threshold  : mean pixel-difference threshold to declare a scene change
    motion_threshold : mean pixel-difference threshold to declare significant motion
    max_frames       : stop after this many sampled frames regardless of video length
    """

    def __init__(
        self,
        strategy: str = "adaptive",
        target_fps: float = 1.0,
        min_fps: float = 0.25,
        max_fps: float = 4.0,
        scene_threshold: float = 30.0,
        motion_threshold: float = 5.0,
        max_frames: int = 2000,
    ) -> None:
        self.strategy = strategy
        self.target_fps = target_fps
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.scene_threshold = scene_threshold
        self.motion_threshold = motion_threshold
        self.max_frames = max_frames

    def sample(self, video_path: str) -> list[FrameRecord]:
        """Read video and return sampled FrameRecord list."""
        import cv2  # deferred so tests can mock

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            records = list(self._sample_stream(cap, fps))
        finally:
            cap.release()
        log.info(
            "Frame sampler [%s]: %d frames sampled from %s",
            self.strategy,
            len(records),
            video_path,
        )
        return records

    def _sample_stream(self, cap: object, video_fps: float) -> Iterator[FrameRecord]:
        import cv2

        min_interval = max(1, round(video_fps / self.max_fps))
        max_interval = max(1, round(video_fps / self.min_fps))
        target_interval = max(1, round(video_fps / self.target_fps))

        prev_gray: np.ndarray | None = None
        frame_idx = 0
        sampled = 0
        last_sampled = -999

        while sampled < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            ts = frame_idx / video_fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            scene_score = 0.0
            if prev_gray is not None:
                scene_score = float(
                    np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)))
                )

            gap = frame_idx - last_sampled
            selected, reason = self._decide(
                gap, min_interval, max_interval, target_interval, scene_score
            )

            if selected:
                yield FrameRecord(
                    frame_index=frame_idx,
                    timestamp_sec=ts,
                    frame=frame.copy(),
                    reason=reason,
                    scene_score=round(scene_score, 2),
                    motion_score=round(scene_score, 2),
                )
                last_sampled = frame_idx
                sampled += 1

            prev_gray = gray
            frame_idx += 1

    def _decide(
        self,
        gap: int,
        min_interval: int,
        max_interval: int,
        target_interval: int,
        scene_score: float,
    ) -> tuple[bool, str]:
        if gap < min_interval:
            return False, ""

        strategy = self.strategy
        st = self.scene_threshold
        mt = self.motion_threshold

        if strategy == "uniform":
            return (gap >= target_interval, "uniform") if gap >= target_interval else (False, "")

        if strategy == "scene_change":
            if scene_score >= st:
                return True, "scene_change"
            if gap >= max_interval:
                return True, "max_interval_fallback"
            return False, ""

        if strategy == "motion_aware":
            if scene_score >= mt:
                return True, "motion"
            if gap >= max_interval:
                return True, "max_interval_fallback"
            return False, ""

        if strategy in ("adaptive", "hybrid"):
            if scene_score >= st:
                return True, "scene_change"
            if scene_score >= mt:
                return True, "motion"
            if gap >= target_interval:
                return True, "periodic"
            return False, ""

        # Unknown strategy — fall back to uniform
        return (gap >= target_interval, "uniform") if gap >= target_interval else (False, "")
