"""CLI entry point for ai-powered-video-analyzer."""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-video-analyzer",
        description=(
            "ai-powered-video-analyzer — offline, privacy-first AI video understanding.\n"
            "Analyzes video locally using YOLO, BLIP, Whisper, PANNs, and Ollama."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=_get_version())

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- analyze ---
    p_analyze = subparsers.add_parser("analyze", help="Analyze a video file.")
    p_analyze.add_argument("video", help="Path to the input video file.")
    p_analyze.add_argument("--output-dir", default=".", help="Directory for output reports.")
    p_analyze.add_argument(
        "--strategy",
        choices=["uniform", "adaptive", "scene_change", "motion_aware", "hybrid"],
        default="adaptive",
        help="Frame sampling strategy (default: adaptive).",
    )
    p_analyze.add_argument("--target-fps", type=float, default=1.0, help="Target frames per second to analyze.")
    p_analyze.add_argument(
        "--backend",
        choices=["auto", "yolo", "visionservex", "none"],
        default="auto",
        help="Object detection backend (default: auto).",
    )
    p_analyze.add_argument("--detector-model", default="", help="Detector model ID (e.g. yolo11x.pt).")
    p_analyze.add_argument("--whisper-model", default="base", help="Whisper model size (default: base).")
    p_analyze.add_argument("--ollama-model", default="phi4:latest", help="Ollama model for summarization.")
    p_analyze.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Compute device (default: auto).",
    )
    p_analyze.add_argument("--no-captioning", action="store_true", help="Skip BLIP image captioning.")
    p_analyze.add_argument("--no-audio-events", action="store_true", help="Skip PANNs audio event detection.")
    p_analyze.add_argument("--no-summarization", action="store_true", help="Skip Ollama LLM summarization.")
    p_analyze.add_argument("--save-annotated-video", action="store_true", help="Write annotated output video.")
    p_analyze.add_argument("--language", default=None, help="Force transcription language (e.g. en, fa, es).")
    p_analyze.add_argument("--max-frames", type=int, default=2000, help="Maximum frames to analyze.")
    p_analyze.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    p_analyze.add_argument("--debug", action="store_true", help="Enable debug logging.")
    p_analyze.add_argument("--strict", action="store_true", help="Fail on any missing model or dependency.")

    # --- doctor ---
    p_doctor = subparsers.add_parser("doctor", help="Check system dependencies.")
    p_doctor.add_argument("--verbose", "-v", action="store_true", help="Verbose output.")

    # --- benchmark ---
    p_bench = subparsers.add_parser("benchmark", help="Benchmark the pipeline on a video.")
    p_bench.add_argument("video", help="Path to video file.")
    p_bench.add_argument("--backend", choices=["auto", "yolo", "visionservex", "none"], default="auto")
    p_bench.add_argument("--strategy", default="adaptive")
    p_bench.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    # --- gui ---
    subparsers.add_parser("gui", help="Launch the Tkinter GUI.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "doctor":
        return _cmd_doctor(args)

    if args.command == "analyze":
        return _cmd_analyze(args)

    if args.command == "benchmark":
        return _cmd_benchmark(args)

    if args.command == "gui":
        return _cmd_gui()

    parser.print_help()
    return 1


def _cmd_doctor(args: argparse.Namespace) -> int:
    from ai_powered_video_analyzer.diagnostics import print_doctor_report, run_doctor
    results = run_doctor(verbose=getattr(args, "verbose", False))
    print_doctor_report(results)
    return 0 if all(r.ok for r in results if "optional" not in r.name.lower()) else 1


def _cmd_analyze(args: argparse.Namespace) -> int:
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
        detector_model=args.detector_model,
        whisper_model=args.whisper_model,
        ollama_model=args.ollama_model,
        device=args.device,
        enable_captioning=not args.no_captioning,
        enable_audio_events=not args.no_audio_events,
        enable_summarization=not args.no_summarization,
        save_annotated_video=args.save_annotated_video,
        transcription_language=args.language,
        max_frames=args.max_frames,
        save_json=True,
        save_markdown=True,
        debug=args.debug,
        verbose=args.verbose,
        strict=args.strict,
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

    print(f"\n✓ Analysis complete.")
    print(f"  Frames analyzed : {report.sampled_frame_count}")
    print(f"  Detections      : {len(report.detections)}")
    print(f"  Captions        : {len(report.captions)}")
    print(f"  Backend used    : {report.backend}")
    if report.summary:
        print(f"\nSummary:\n{report.summary}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    import time as _time
    from ai_powered_video_analyzer.config import AnalysisConfig
    from ai_powered_video_analyzer.frames import FrameSampler
    from ai_powered_video_analyzer.backends import load_backend
    from ai_powered_video_analyzer.logging_utils import setup_logging

    setup_logging(verbose=True)

    print(f"\nBenchmarking: {args.video}")
    print(f"  Backend  : {args.backend}")
    print(f"  Strategy : {args.strategy}")

    # Uniform baseline
    sampler_u = FrameSampler(strategy="uniform", target_fps=1.0)
    t0 = _time.perf_counter()
    try:
        frames_u = sampler_u.sample(args.video)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    t_uniform = _time.perf_counter() - t0

    # Adaptive
    sampler_a = FrameSampler(strategy=args.strategy, target_fps=1.0)
    t0 = _time.perf_counter()
    frames_a = sampler_a.sample(args.video)
    t_adaptive = _time.perf_counter() - t0

    speedup = t_uniform / t_adaptive if t_adaptive > 0 else 0
    reduction = 1.0 - len(frames_a) / max(len(frames_u), 1)

    backend = load_backend(args.backend, device=args.device)
    t0 = _time.perf_counter()
    for r in frames_a[:50]:
        backend.predict(r.frame)
    t_detect = _time.perf_counter() - t0

    print(f"\n--- Results ---")
    print(f"  Uniform  frames: {len(frames_u):>6}   time: {t_uniform:.2f}s")
    print(f"  {args.strategy:<8} frames: {len(frames_a):>6}   time: {t_adaptive:.2f}s")
    print(f"  Frame reduction : {reduction:.1%}")
    print(f"  Sample speedup  : {speedup:.2f}×")
    print(f"  Detection time (first 50 frames): {t_detect:.2f}s")
    print(f"  Backend: {backend.backend_name} / model: {backend.model_id}")
    return 0


def _cmd_gui() -> int:
    try:
        import tkinter as tk
        from video_processing_gui import VideoProcessingGUI
        root = tk.Tk()
        app = VideoProcessingGUI(root)
        root.mainloop()
        return 0
    except ImportError as exc:
        print(f"GUI unavailable: {exc}", file=sys.stderr)
        return 1


def _get_version() -> str:
    from ai_powered_video_analyzer import __version__
    return f"ai-video-analyzer {__version__}"


if __name__ == "__main__":
    sys.exit(main())
