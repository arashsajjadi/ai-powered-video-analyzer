"""Report generation: JSON sidecar and Markdown summary."""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

from ai_powered_video_analyzer.logging_utils import get_logger

log = get_logger(__name__)


@dataclasses.dataclass
class AnalysisReport:
    """Complete analysis output for one video."""

    video_path: str
    duration_sec: float
    fps: float
    width: int
    height: int
    frame_count: int
    sampled_frame_count: int
    backend: str
    model_ids: dict[str, str]

    transcript: str = ""
    transcript_language: str = "unknown"
    audio_events: dict[str, list[str]] = dataclasses.field(default_factory=dict)

    frame_observations: list[dict] = dataclasses.field(default_factory=list)
    detections: list[dict] = dataclasses.field(default_factory=list)
    captions: list[dict] = dataclasses.field(default_factory=list)

    summary: str = ""
    limitations: list[str] = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# AI-Powered Video Analysis Report")
        lines.append("")
        lines.append("## Video Metadata")
        lines.append(f"- **File**: `{self.video_path}`")
        lines.append(f"- **Duration**: {self.duration_sec:.1f}s")
        lines.append(f"- **Resolution**: {self.width}×{self.height} @ {self.fps:.2f} fps")
        lines.append(f"- **Total frames**: {self.frame_count}")
        lines.append(f"- **Analyzed frames**: {self.sampled_frame_count}")
        lines.append(f"- **Detection backend**: {self.backend}")
        lines.append("")

        if self.summary:
            lines.append("## Executive Summary")
            lines.append(self.summary)
            lines.append("")

        if self.transcript:
            lines.append("## Speech Transcript")
            lines.append(f"*Detected language: {self.transcript_language}*")
            lines.append("")
            lines.append(self.transcript)
            lines.append("")

        if self.audio_events:
            lines.append("## Audio Events")
            for event, times in self.audio_events.items():
                ts = ", ".join(times) if times else "N/A"
                lines.append(f"- **{event}**: {ts}")
            lines.append("")

        if self.detections:
            lines.append("## Detected Objects (sample, top 20)")
            from collections import Counter
            counts = Counter(d["label"] for d in self.detections)
            for label, count in counts.most_common(20):
                lines.append(f"- **{label}**: seen {count}× across sampled frames")
            lines.append("")

        if self.captions:
            lines.append("## Frame Captions (sample, up to 10)")
            for cap in self.captions[:10]:
                ts = cap.get("timestamp_sec", 0)
                text = cap.get("caption", "")
                lines.append(f"- `{ts:.1f}s`: {text}")
            lines.append("")

        if self.limitations:
            lines.append("## Limitations & Notes")
            for note in self.limitations:
                lines.append(f"- {note}")
            lines.append("")

        return "\n".join(lines)


def save_report(report: AnalysisReport, output_dir: str, stem: str = "report") -> dict[str, str]:
    """Write JSON and Markdown files. Returns dict of {format: path}."""
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}

    json_path = os.path.join(output_dir, f"{stem}.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.to_json())
        paths["json"] = json_path
        log.info("JSON report saved: %s", json_path)
    except Exception as exc:
        log.error("Failed to save JSON report: %s", exc)

    md_path = os.path.join(output_dir, f"{stem}.md")
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
        paths["markdown"] = md_path
        log.info("Markdown report saved: %s", md_path)
    except Exception as exc:
        log.error("Failed to save Markdown report: %s", exc)

    # Legacy compatibility: write report.txt
    txt_path = os.path.join(output_dir, "report.txt")
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
        paths["txt"] = txt_path
    except Exception:
        pass

    return paths
