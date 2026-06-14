"""Report schema and serialization tests."""

from __future__ import annotations

import json

import pytest

from ai_powered_video_analyzer.reports import AnalysisReport


def _sample_report() -> AnalysisReport:
    return AnalysisReport(
        video_path="sample.mp4",
        duration_sec=10.0,
        fps=25.0,
        width=640,
        height=480,
        frame_count=250,
        sampled_frame_count=10,
        backend="null",
        model_ids={"detector": "null", "whisper": "base", "blip": "disabled", "ollama": "phi4:latest"},
        transcript="Hello world",
        transcript_language="en",
        audio_events={"Speech": ["00:00:01"]},
        detections=[
            {"label": "person", "score": 0.9, "bbox_xyxy": [10, 20, 50, 60],
             "frame_index": 0, "timestamp_sec": 0.0, "backend": "null", "model_id": "null"}
        ],
        captions=[{"frame_index": 0, "timestamp_sec": 0.0, "caption": "A person stands in a room."}],
        summary="A person was seen in the video.",
        limitations=["BLIP captioning disabled."],
    )


class TestAnalysisReport:
    def test_to_dict_has_required_keys(self):
        r = _sample_report()
        d = r.to_dict()
        for key in ["video_path", "duration_sec", "fps", "transcript", "summary",
                    "detections", "captions", "audio_events", "limitations"]:
            assert key in d, f"Missing key: {key}"

    def test_to_json_is_valid(self):
        r = _sample_report()
        j = r.to_json()
        data = json.loads(j)
        assert data["video_path"] == "sample.mp4"
        assert data["sampled_frame_count"] == 10

    def test_to_markdown_contains_sections(self):
        r = _sample_report()
        md = r.to_markdown()
        assert "# AI-Powered Video Analysis Report" in md
        assert "Summary" in md
        assert "Speech Transcript" in md
        assert "Audio Events" in md
        assert "Detected Objects" in md

    def test_to_json_round_trip(self):
        r = _sample_report()
        j = r.to_json()
        data = json.loads(j)
        assert data["transcript"] == "Hello world"
        assert data["audio_events"]["Speech"] == ["00:00:01"]
        assert len(data["detections"]) == 1
        assert data["detections"][0]["label"] == "person"

    def test_limitations_preserved(self):
        r = _sample_report()
        md = r.to_markdown()
        assert "BLIP captioning disabled." in md


class TestSaveReport:
    def test_saves_json_and_markdown(self, tmp_path):
        from ai_powered_video_analyzer.reports import save_report
        r = _sample_report()
        paths = save_report(r, str(tmp_path), stem="test_report")
        assert "json" in paths
        assert "markdown" in paths
        assert (tmp_path / "test_report.json").exists()
        assert (tmp_path / "test_report.md").exists()

    def test_saved_json_is_parseable(self, tmp_path):
        from ai_powered_video_analyzer.reports import save_report
        r = _sample_report()
        save_report(r, str(tmp_path), stem="test_report")
        data = json.loads((tmp_path / "test_report.json").read_text())
        assert data["duration_sec"] == 10.0
