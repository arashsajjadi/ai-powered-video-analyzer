"""Backend unit tests — no real model weights needed (uses mocks)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_frame(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestNullBackend:
    def test_returns_empty(self):
        from ai_powered_video_analyzer.backends.null_backend import NullBackend
        b = NullBackend()
        result = b.predict(_make_frame())
        assert result == []

    def test_batch_returns_list_of_empty(self):
        from ai_powered_video_analyzer.backends.null_backend import NullBackend
        b = NullBackend()
        result = b.predict_batch([_make_frame(), _make_frame()])
        assert result == [[], []]

    def test_is_available(self):
        from ai_powered_video_analyzer.backends.null_backend import NullBackend
        assert NullBackend().is_available()


class TestDetection:
    def test_to_dict(self):
        from ai_powered_video_analyzer.backends.base import Detection
        d = Detection(label="cat", score=0.9, x1=10, y1=20, x2=50, y2=60,
                      frame_index=1, timestamp_sec=0.5, backend="test", model_id="m")
        data = d.to_dict()
        assert data["label"] == "cat"
        assert data["score"] == 0.9
        assert data["frame_index"] == 1
        assert "bbox_xyxy" in data

    def test_center(self):
        from ai_powered_video_analyzer.backends.base import Detection
        d = Detection(label="x", score=1.0, x1=0, y1=0, x2=100, y2=100)
        assert d.cx == 50.0
        assert d.cy == 50.0


class TestVisionServeXBackend:
    def test_raises_import_error_when_not_installed(self):
        import sys
        with patch.dict(sys.modules, {"visionservex": None}):
            with pytest.raises(ImportError, match="VisionServeX is not installed"):
                from ai_powered_video_analyzer.backends.visionservex_backend import VisionServeXBackend
                VisionServeXBackend()

    def test_parse_result_filters_by_confidence(self):
        from ai_powered_video_analyzer.backends.visionservex_backend import VisionServeXBackend

        mock_vsx = MagicMock()
        with patch("visionservex.VisionModel", return_value=mock_vsx):
            # Manually create backend without calling __init__
            b = object.__new__(VisionServeXBackend)
            b._model = mock_vsx
            b.model_id = "mock"

            mock_box = MagicMock()
            mock_box.x1 = 10.0
            mock_box.y1 = 20.0
            mock_box.x2 = 50.0
            mock_box.y2 = 60.0

            low_conf = MagicMock()
            low_conf.score = 0.1
            low_conf.label = "noise"
            low_conf.box = mock_box

            high_conf = MagicMock()
            high_conf.score = 0.95
            high_conf.label = "person"
            high_conf.box = mock_box

            result = MagicMock()
            result.detections = [low_conf, high_conf]

            dets = b._parse_result(result, confidence=0.5)
            assert len(dets) == 1
            assert dets[0].label == "person"


class TestLoadBackend:
    def test_load_none_backend(self):
        from ai_powered_video_analyzer.backends import load_backend
        b = load_backend("none")
        from ai_powered_video_analyzer.backends.null_backend import NullBackend
        assert isinstance(b, NullBackend)

    def test_load_visionservex_raises_when_not_installed(self):
        import sys
        with patch.dict(sys.modules, {"visionservex": None}):
            from ai_powered_video_analyzer.backends import load_backend
            with pytest.raises(ImportError, match="VisionServeX backend requested"):
                load_backend("visionservex")

    def test_auto_falls_back_to_null_when_nothing_installed(self):
        import sys
        with patch.dict(sys.modules, {"visionservex": None, "ultralytics": None}):
            from ai_powered_video_analyzer.backends import load_backend
            import importlib
            import ai_powered_video_analyzer.backends as bmod
            importlib.reload(bmod)
            b = bmod.load_backend("none")
            from ai_powered_video_analyzer.backends.null_backend import NullBackend
            assert isinstance(b, NullBackend)
