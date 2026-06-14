"""Config dataclass tests."""

from __future__ import annotations


def test_default_config_is_valid():
    from ai_powered_video_analyzer.config import AnalysisConfig
    c = AnalysisConfig()
    assert c.frame_strategy == "adaptive"
    assert c.target_fps == 1.0
    assert c.backend == "auto"
    assert c.device == "auto"


def test_pann_path_defaults_to_string():
    from ai_powered_video_analyzer.config import AnalysisConfig
    c = AnalysisConfig()
    assert isinstance(c.pann_model_path, str)
    assert len(c.pann_model_path) > 0


def test_config_override():
    from ai_powered_video_analyzer.config import AnalysisConfig
    c = AnalysisConfig(target_fps=2.0, backend="none", whisper_model="tiny")
    assert c.target_fps == 2.0
    assert c.backend == "none"
    assert c.whisper_model == "tiny"
