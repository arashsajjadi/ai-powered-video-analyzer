"""Report generation: JSON sidecar and Markdown summary."""

from __future__ import annotations

import dataclasses
import json
import os
from collections import Counter
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

    # Run configuration (added v1.1)
    preset: str = "balanced"
    frame_strategy: str = "adaptive"
    timings: dict[str, float] = dataclasses.field(default_factory=dict)

    transcript: str = ""
    transcript_language: str = "unknown"
    audio_events: dict[str, list[str]] = dataclasses.field(default_factory=dict)

    frame_observations: list[dict] = dataclasses.field(default_factory=list)
    detections: list[dict] = dataclasses.field(default_factory=list)
    captions: list[dict] = dataclasses.field(default_factory=list)

    summary: str = ""
    limitations: list[str] = dataclasses.field(default_factory=list)

    def top_labels(self, n: int = 10) -> list[tuple[str, int]]:
        """Return the top-n detected labels by count."""
        return Counter(d["label"] for d in self.detections).most_common(n)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# AI-Powered Video Analysis Report")
        lines.append("")

        # --- Run configuration ---
        lines.append("## Run Configuration")
        lines.append(f"- **File**: `{self.video_path}`")
        lines.append(f"- **Duration**: {self.duration_sec:.1f}s")
        lines.append(f"- **Resolution**: {self.width}×{self.height} @ {self.fps:.2f} fps")
        lines.append(f"- **Total frames**: {self.frame_count}")
        lines.append(f"- **Frames analyzed**: {self.sampled_frame_count}")
        lines.append(f"  *(sampling strategy: {self.frame_strategy})*")
        lines.append(f"- **Detection backend**: {self.backend}")
        lines.append(f"- **Detector preset**: {self.preset}")
        lines.append(f"- **Detector model**: {self.model_ids.get('detector', 'unknown')}")
        if self.timings:
            total = sum(self.timings.values())
            timing_str = " | ".join(f"{k}: {v:.1f}s" for k, v in sorted(self.timings.items()))
            lines.append(f"- **Timing**: {timing_str} | total: {total:.1f}s")
        lines.append("")

        # --- Executive summary ---
        if self.summary:
            lines.append("## Summary")
            lines.append(self.summary)
            lines.append("")

        # --- Detected objects ---
        if self.detections:
            top = self.top_labels(20)
            lines.append(f"## Detected Objects ({len(self.detections)} total detections)")
            lines.append("")
            lines.append("| Label | Count | Max confidence |")
            lines.append("|-------|-------|---------------|")
            label_scores: dict[str, float] = {}
            for d in self.detections:
                lbl = d["label"]
                sc = d.get("score", 0.0)
                if sc > label_scores.get(lbl, 0.0):
                    label_scores[lbl] = sc
            for label, count in top:
                max_score = label_scores.get(label, 0.0)
                lines.append(f"| {label} | {count} | {max_score:.2f} |")
            lines.append("")
        else:
            lines.append("## Detected Objects")
            lines.append(
                "> **Warning**: No objects detected. "
                "Check that the detection backend is installed (`ai-video-analyzer doctor`) "
                "and the video file is valid."
            )
            lines.append("")

        # --- Frame captions ---
        if self.captions:
            lines.append(f"## Scene Descriptions (BLIP captions, {len(self.captions)} frames)")
            lines.append("")
            lines.append("| Time | Caption |")
            lines.append("|------|---------|")
            for cap in self.captions[:15]:
                ts = cap.get("timestamp_sec", 0)
                text = cap.get("caption", "")
                lines.append(f"| {ts:.1f}s | {text} |")
            lines.append("")

        # --- Transcript ---
        if self.transcript:
            lines.append("## Speech Transcript")
            lines.append(f"*Detected language: {self.transcript_language}*")
            lines.append("")
            lines.append(self.transcript)
            lines.append("")

        # --- Audio events ---
        if self.audio_events:
            lines.append("## Audio Events")
            for event, times in self.audio_events.items():
                ts = ", ".join(times) if times else "N/A"
                lines.append(f"- **{event}**: {ts}")
            lines.append("")

        # --- Limitations ---
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

    # Legacy plain-text compatibility
    txt_path = os.path.join(output_dir, "report.txt")
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
        paths["txt"] = txt_path
    except Exception:
        pass

    return paths
