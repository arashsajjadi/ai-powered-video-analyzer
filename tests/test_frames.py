"""Frame sampler unit tests — uses synthetic in-memory frames, no real video."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai_powered_video_analyzer.frames import FrameRecord, FrameSampler


def _make_gray(value: int = 128, h: int = 32, w: int = 32) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


class TestFrameSamplerDecide:
    """Unit-test _decide() without any I/O."""

    def _sampler(self, strategy="adaptive", scene_thr=30.0, motion_thr=5.0, target_fps=1.0):
        return FrameSampler(
            strategy=strategy,
            target_fps=target_fps,
            min_fps=0.25,
            max_fps=4.0,
            scene_threshold=scene_thr,
            motion_threshold=motion_thr,
        )

    def test_uniform_selects_at_target(self):
        s = self._sampler("uniform")
        # With video_fps=25 and target_fps=1, target_interval=25
        selected, reason = s._decide(25, 7, 100, 25, 0.0)
        assert selected
        assert reason == "uniform"

    def test_uniform_skips_before_target(self):
        s = self._sampler("uniform")
        selected, _ = s._decide(10, 1, 100, 25, 0.0)
        assert not selected

    def test_scene_change_triggers_on_high_score(self):
        s = self._sampler("scene_change", scene_thr=30.0)
        selected, reason = s._decide(5, 1, 100, 25, 40.0)
        assert selected
        assert reason == "scene_change"

    def test_scene_change_falls_back_to_max_interval(self):
        s = self._sampler("scene_change")
        selected, reason = s._decide(100, 1, 100, 25, 5.0)
        assert selected
        assert reason == "max_interval_fallback"

    def test_adaptive_combines_scene_and_periodic(self):
        s = self._sampler("adaptive", scene_thr=30.0, motion_thr=5.0)
        # High scene score — should select
        sel, reason = s._decide(5, 1, 100, 25, 35.0)
        assert sel and reason == "scene_change"
        # Low score but at periodic interval
        sel2, reason2 = s._decide(25, 1, 100, 25, 0.0)
        assert sel2 and reason2 == "periodic"

    def test_min_interval_blocks_too_frequent(self):
        s = self._sampler("adaptive")
        # min_interval is 7 (video_fps=25/max_fps=4 ≈ 6), gap=3 < 7 → skip
        selected, _ = s._decide(3, 7, 100, 25, 999.0)
        assert not selected


class TestFrameRecord:
    def test_to_dict(self):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        r = FrameRecord(frame_index=5, timestamp_sec=0.2, frame=frame,
                        reason="scene_change", scene_score=35.1, motion_score=35.1)
        d = r.to_dict()
        assert d["frame_index"] == 5
        assert d["timestamp_sec"] == 0.2
        assert d["reason"] == "scene_change"
        assert "frame" not in d  # frame array must NOT be in the dict (not serializable)


class TestFrameSamplerSampleMocked:
    """Tests that call sample() with a mocked cv2 in sys.modules."""

    def _cap_from_frames(self, grays: list[np.ndarray], fps: float = 25.0):
        """Build a mock VideoCapture that yields BGR frames from gray arrays."""
        frames_bgr = [np.stack([g, g, g], axis=-1) for g in grays]
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = fps
        reads = [(True, f) for f in frames_bgr] + [(False, None)]
        cap.read.side_effect = reads
        return cap

    def _make_mock_cv2(self, cap):
        import sys
        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.COLOR_BGR2GRAY = 6
        mock_cv2.cvtColor.side_effect = lambda f, _: f[:, :, 0]
        return mock_cv2

    def test_uniform_10_frames(self):
        import sys
        grays = [_make_gray(i * 10) for i in range(10)]
        cap = self._cap_from_frames(grays, fps=10.0)
        mock_cv2 = self._make_mock_cv2(cap)
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            s = FrameSampler(strategy="uniform", target_fps=1.0, min_fps=0.1, max_fps=5.0)
            records = s.sample("fake.mp4")
        # With 10 frames @ 10fps and target=1fps, interval=10 → 1 frame selected
        assert len(records) >= 1

    def test_returns_frame_record_objects(self):
        import sys
        grays = [_make_gray(100)] * 50
        cap = self._cap_from_frames(grays, fps=25.0)
        mock_cv2 = self._make_mock_cv2(cap)
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            s = FrameSampler(strategy="uniform", target_fps=1.0)
            records = s.sample("fake.mp4")
        for r in records:
            assert isinstance(r, FrameRecord)
            assert r.frame is not None

    def test_max_frames_respected(self):
        import sys
        grays = [_make_gray(i % 256) for i in range(100)]
        cap = self._cap_from_frames(grays, fps=25.0)
        mock_cv2 = self._make_mock_cv2(cap)
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            s = FrameSampler(strategy="uniform", target_fps=25.0, max_frames=5)
            records = s.sample("fake.mp4")
        assert len(records) <= 5
