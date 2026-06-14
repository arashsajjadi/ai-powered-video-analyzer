#!/usr/bin/env python
"""
video_processing.py — CLI compatibility shim for ai-powered-video-analyzer.

This script preserves the command-line interface that the README documents:
    python video_processing.py --video path/to/video.mp4 --save

For the full feature set, use the package CLI instead:
    ai-video-analyzer analyze path/to/video.mp4
    python -m ai_powered_video_analyzer.cli analyze path/to/video.mp4
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="video_processing.py",
        description=(
            "Offline AI video analysis — object detection, captioning, "
            "transcription, audio events, and LLM summarization."
        ),
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--save", action="store_true", help="Save annotated output video."
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "yolo", "visionservex", "none"],
        default="auto",
        help="Object detection backend (default: auto).",
    )
    parser.add_argument(
        "--strategy",
        choices=["uniform", "adaptive", "scene_change", "motion_aware", "hybrid"],
        default="adaptive",
        help="Frame sampling strategy (default: adaptive).",
    )
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--ollama-model", default="phi4:latest")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--language", default=None)
    parser.add_argument("--no-captioning", action="store_true")
    parser.add_argument("--no-audio-events", action="store_true")
    parser.add_argument("--no-summarization", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    from ai_powered_video_analyzer.config import AnalysisConfig
    from ai_powered_video_analyzer.core import analyze_video
    from ai_powered_video_analyzer.logging_utils import setup_logging

    setup_logging(debug=args.debug, verbose=args.verbose)

    config = AnalysisConfig(
        video_path=args.video,
        output_dir=args.output_dir,
        frame_strategy=args.strategy,
        target_fps=args.target_fps,
        backend=args.backend,
        whisper_model=args.whisper_model,
        ollama_model=args.ollama_model,
        device=args.device,
        enable_captioning=not args.no_captioning,
        enable_audio_events=not args.no_audio_events,
        enable_summarization=not args.no_summarization,
        save_annotated_video=args.save,
        transcription_language=args.language,
        save_json=True,
        save_markdown=True,
        debug=args.debug,
        verbose=args.verbose,
    )

    try:
        report = analyze_video(config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

    print(f"\nDone. {report.sampled_frame_count} frames analyzed, "
          f"{len(report.detections)} detections, backend={report.backend}.")
    if report.summary:
        print(f"\nSummary:\n{report.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
