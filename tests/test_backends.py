"""Backend unit tests — no real model weights needed (uses mocks)."""

from __future__ import annotations

import sys
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


class TestPresetResolution:
    """Verify preset → model_id mapping is consistent."""

    def test_balanced_resolves_to_dfine_s(self):
        from ai_powered_video_analyzer.backends.visionservex_backend import resolve_model_id
        assert resolve_model_id(preset="balanced") == "dfine-s"

    def test_fast_resolves_to_dfine_n(self):
        from ai_powered_video_analyzer.backends.visionservex_backend import resolve_model_id
        assert resolve_model_id(preset="fast") == "dfine-n"

    def test_quality_resolves_to_dfine_m(self):
        from ai_powered_video_analyzer.backends.visionservex_backend import resolve_model_id
        assert resolve_model_id(preset="quality") == "dfine-m"

    def test_quality_plus_resolves_to_dfine_l(self):
        from ai_powered_video_analyzer.backends.visionservex_backend import resolve_model_id
        assert resolve_model_id(preset="quality+") == "dfine-l"

    def test_explicit_model_id_takes_precedence(self):
        from ai_powered_video_analyzer.backends.visionservex_backend import resolve_model_id
        assert resolve_model_id(model_id="rfdetr-base", preset="fast") == "rfdetr-base"

    def test_invalid_preset_raises(self):
        from ai_powered_video_analyzer.backends.visionservex_backend import resolve_model_id
        with pytest.raises(ValueError, match="Unknown detector preset"):
            resolve_model_id(preset="turbo")

    def test_no_args_returns_default(self):
        from ai_powered_video_analyzer.backends.visionservex_backend import resolve_model_id, _DEFAULT_MODEL
        assert resolve_model_id() == _DEFAULT_MODEL

    def test_preset_models_exported_from_backends_init(self):
        from ai_powered_video_analyzer.backends import PRESET_MODELS
        assert "balanced" in PRESET_MODELS
        assert PRESET_MODELS["balanced"] == "dfine-s"
        assert len(PRESET_MODELS) >= 4


class TestVisionServeXBackend:
    def test_raises_import_error_when_not_installed(self):
        with patch.dict(sys.modules, {"visionservex": None}):
            with pytest.raises(ImportError, match="VisionServeX is not installed"):
                from ai_powered_video_analyzer.backends import visionservex_backend
                import importlib
                importlib.reload(visionservex_backend)
                visionservex_backend.VisionServeXBackend()

    def test_parse_detections_filters_by_confidence(self):
        from ai_powered_video_analyzer.backends.visionservex_backend import _parse_detections

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

        dets = _parse_detections(result, confidence=0.5, backend_name="visionservex", model_id="dfine-s")
        assert len(dets) == 1
        assert dets[0].label == "person"
        assert dets[0].score == 0.95

    def test_parse_detections_no_box_skipped(self):
        from ai_powered_video_analyzer.backends.visionservex_backend import _parse_detections

        det = MagicMock()
        det.score = 0.9
        det.label = "cat"
        det.box = None

        result = MagicMock()
        result.detections = [det]

        dets = _parse_detections(result, confidence=0.3, backend_name="visionservex", model_id="dfine-s")
        assert dets == []

    def test_parse_detections_returns_empty_for_no_detections_attr(self):
        from ai_powered_video_analyzer.backends.visionservex_backend import _parse_detections
        result = MagicMock(spec=[])
        dets = _parse_detections(result, confidence=0.3, backend_name="visionservex", model_id="dfine-s")
        assert dets == []


class TestLoadBackend:
    def test_load_none_backend(self):
        from ai_powered_video_analyzer.backends import load_backend
        b = load_backend("none")
        from ai_powered_video_analyzer.backends.null_backend import NullBackend
        assert isinstance(b, NullBackend)

    def test_load_visionservex_raises_when_not_installed(self):
        with patch.dict(sys.modules, {"visionservex": None}):
            from ai_powered_video_analyzer.backends import load_backend
            with pytest.raises(ImportError, match="VisionServeX backend requested"):
                load_backend("visionservex")

    def test_auto_falls_back_to_null_when_vsx_not_installed(self):
        with patch.dict(sys.modules, {"visionservex": None}):
            from ai_powered_video_analyzer.backends import load_backend
            b = load_backend("auto")
            from ai_powered_video_analyzer.backends.null_backend import NullBackend
            assert isinstance(b, NullBackend)

    def test_load_backend_passes_preset(self):
        """load_backend('none') with preset param must not crash."""
        from ai_powered_video_analyzer.backends import load_backend
        b = load_backend("none", preset="quality")
        from ai_powered_video_analyzer.backends.null_backend import NullBackend
        assert isinstance(b, NullBackend)

    def test_default_backend_is_visionservex_in_config(self):
        """AnalysisConfig default backend must be visionservex (not auto or yolo)."""
        from ai_powered_video_analyzer.config import AnalysisConfig
        cfg = AnalysisConfig()
        assert cfg.backend == "visionservex"

    def test_default_preset_is_balanced_in_config(self):
        from ai_powered_video_analyzer.config import AnalysisConfig
        cfg = AnalysisConfig()
        assert cfg.detector_preset == "balanced"


class TestLegacyYOLONotRequired:
    """Ultralytics must NOT be a hard dependency since v0.3.0."""

    def test_ultralytics_not_required_for_null_backend(self):
        with patch.dict(sys.modules, {"ultralytics": None}):
            from ai_powered_video_analyzer.backends.null_backend import NullBackend
            b = NullBackend()
            assert b.predict(_make_frame()) == []

    def test_ultralytics_not_required_for_config(self):
        with patch.dict(sys.modules, {"ultralytics": None}):
            from ai_powered_video_analyzer.config import AnalysisConfig
            cfg = AnalysisConfig()
            assert cfg.backend == "visionservex"

    def test_legacy_yolo_import_gives_correct_class_name(self):
        try:
            from ai_powered_video_analyzer.backends.legacy_yolo_backend import LegacyYOLOBackend
            assert LegacyYOLOBackend.backend_name == "legacy_yolo"
        except ImportError:
            pytest.skip("ultralytics not installed — expected for v0.3.0 default install")

    def test_yolo_backend_alias_is_none(self):
        from ai_powered_video_analyzer.backends.legacy_yolo_backend import YOLOBackend
        assert YOLOBackend is None

    def test_load_legacy_yolo_raises_without_ultralytics(self):
        with patch.dict(sys.modules, {"ultralytics": None}):
            from ai_powered_video_analyzer.backends import load_backend
            with pytest.raises(ImportError, match="ultralytics"):
                load_backend("legacy_yolo")
