"""CLI entry point for ai-powered-video-analyzer."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    from ai_powered_video_analyzer import __version__
    parser = argparse.ArgumentParser(
        prog="ai-video-analyzer",
        description=(
            f"ai-powered-video-analyzer {__version__}\n"
            "Offline, privacy-first video analysis powered by VisionServeX.\n\n"
            "Quick start:\n"
            "  pip install -e '.[vision]'\n"
            "  ai-video-analyzer doctor\n"
            "  ai-video-analyzer analyze video.mp4 --preset balanced"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"ai-video-analyzer {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    _add_analyze_parser(subparsers)
    _add_doctor_parser(subparsers)
    _add_benchmark_parser(subparsers)
    _add_gui_parser(subparsers)
    _add_list_models_parser(subparsers)

    return parser


def _add_analyze_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "analyze",
        help="Analyze a video file.",
        description=(
            "Run the full offline AI pipeline on a video:\n"
            "  detection (VisionServeX) → captioning (BLIP) → transcription (Whisper)\n"
            "  → audio events (PANNs) → LLM summary (Ollama)\n\n"
            "All stages except detection are optional and can be skipped."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", help="Path to the input video file.")
    p.add_argument("--output-dir", "-o", default=".", help="Directory for output reports (default: current dir).")

    # Detection
    p.add_argument(
        "--preset",
        choices=["fast", "balanced", "quality", "quality+"],
        default="balanced",
        dest="preset",
        help=(
            "Detection quality preset (default: balanced):\n"
            "  fast      → dfine-n  (~18ms/frame, good accuracy)\n"
            "  balanced  → dfine-s  (~19ms/frame, best default) [DEFAULT]\n"
            "  quality   → dfine-m  (~21ms/frame, highest COCO accuracy)\n"
            "  quality+  → dfine-l  (larger model, maximum accuracy)"
        ),
    )
    p.add_argument("--model", dest="detector_model", default="",
                   help="Explicit VisionServeX model ID (overrides --preset).")
    p.add_argument("--device",
                   choices=["auto", "cpu", "cuda", "mps"],
                   default="auto",
                   help="Compute device (default: auto).")
    p.add_argument("--confidence", type=float, default=0.3,
                   help="Minimum detection confidence 0..1 (default: 0.3).")
    p.add_argument("--backend",
                   choices=["visionservex", "none", "legacy_yolo"],
                   default="visionservex",
                   help="Detection backend (default: visionservex).")

    # Frame sampling
    p.add_argument(
        "--strategy",
        choices=["uniform", "adaptive", "scene_change", "motion_aware", "hybrid"],
        default="adaptive",
        help="Frame sampling strategy (default: adaptive).",
    )
    p.add_argument("--target-fps", type=float, default=1.0,
                   help="Target frames-per-second to analyze (default: 1.0).")
    p.add_argument("--max-frames", type=int, default=2000,
                   help="Hard cap on analyzed frames per video (default: 2000).")

    # Optional heavy stages
    p.add_argument("--whisper-model", default="base",
                   help="Whisper model size: tiny|base|small|medium|large-v2 (default: base).")
    p.add_argument("--language", default=None,
                   help="Force transcription language code (e.g. en, fa).")
    p.add_argument("--ollama-model", default="",
                   help="Ollama model name for summarization (default: auto-discover).")
    p.add_argument("--summary-style",
                   choices=["concise", "evidence", "technical", "narrative"],
                   default="concise",
                   help="LLM summary style (default: concise).")

    # Skip flags
    p.add_argument("--no-captioning", action="store_true", help="Skip BLIP captioning.")
    p.add_argument("--no-audio-events", action="store_true", help="Skip PANNs audio events.")
    p.add_argument("--no-summarization", action="store_true", help="Skip Ollama summarization.")

    # Output / behavior
    p.add_argument("--list-ollama-models", action="store_true",
                   help="Print available Ollama models and exit.")
    p.add_argument("--save-video", action="store_true", help="Write annotated output video.")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if any required component is missing.")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    p.add_argument("--debug", action="store_true", help="Debug logging.")


def _add_doctor_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "doctor",
        help="Check all system dependencies and print a status report.",
    )
    p.add_argument("--verbose", "-v", action="store_true")


def _add_benchmark_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "benchmark",
        help="Benchmark detection speed and frame-sampling on a video.",
        description=(
            "Measures:\n"
            "  - Frame sampling: number of frames selected and elapsed time\n"
            "  - Detection: ms/frame, total detections, top labels\n"
            "  - Comparison across presets (with --compare)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("video", help="Path to video file.")
    p.add_argument("--preset", choices=["fast", "balanced", "quality", "quality+"],
                   default="balanced", dest="preset")
    p.add_argument("--model", dest="detector_model", default="")
    p.add_argument("--strategy", default="adaptive",
                   choices=["uniform", "adaptive", "scene_change", "motion_aware", "hybrid"])
    p.add_argument("--max-frames", type=int, default=100,
                   help="Max frames to run detection on (default: 100).")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--compare", action="store_true",
                   help="Compare fast/balanced/quality presets side-by-side.")


def _add_gui_parser(subparsers: argparse._SubParsersAction) -> None:
    subparsers.add_parser("gui", help="Launch the Tkinter GUI.")


def _add_list_models_parser(subparsers: argparse._SubParsersAction) -> None:
    subparsers.add_parser(
        "list-models",
        help="List available VisionServeX model presets and registry entries.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    dispatch = {
        "doctor": _cmd_doctor,
        "analyze": _cmd_analyze,
        "benchmark": _cmd_benchmark,
        "gui": _cmd_gui,
        "list-models": _cmd_list_models,
    }
    fn = dispatch.get(args.command)
    if fn is None:
        parser.print_help()
        return 1
    return fn(args)


def _cmd_doctor(args: argparse.Namespace) -> int:
    from ai_powered_video_analyzer.diagnostics import print_doctor_report, run_doctor
    results = run_doctor(verbose=getattr(args, "verbose", False))
    print_doctor_report(results)
    failures = [r for r in results if not r.ok and not r.optional]
    return 0 if not failures else 1


def _cmd_analyze(args: argparse.Namespace) -> int:
    from ai_powered_video_analyzer.logging_utils import setup_logging
    setup_logging(debug=args.debug, verbose=args.verbose)

    if args.list_ollama_models:
        from ai_powered_video_analyzer.summarization import list_ollama_models
        models = list_ollama_models()
        if models:
            print("Available Ollama models:")
            for m in models:
                print(f"  {m}")
        else:
            print("No Ollama models found. Is Ollama running?\n  Run: ollama serve")
        return 0

    from ai_powered_video_analyzer.config import AnalysisConfig
    from ai_powered_video_analyzer.core import analyze_video

    config = AnalysisConfig(
        video_path=args.video,
        output_dir=args.output_dir,
        frame_strategy=args.strategy,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        backend=args.backend,
        detector_model=args.detector_model,
        detector_preset=args.preset,
        detection_confidence=args.confidence,
        device=args.device,
        whisper_model=args.whisper_model,
        ollama_model=args.ollama_model or "",
        summary_style=args.summary_style,
        enable_captioning=not args.no_captioning,
        enable_audio_events=not args.no_audio_events,
        enable_summarization=not args.no_summarization,
        save_annotated_video=args.save_video,
        transcription_language=args.language,
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

    from pathlib import Path
    print(f"\nAnalysis complete.")
    print(f"  Video     : {report.video_path}")
    print(f"  Duration  : {report.duration_sec:.1f}s")
    print(f"  Frames    : {report.sampled_frame_count} analyzed / {report.frame_count} total")
    print(f"  Preset    : {report.preset}  model={report.model_ids.get('detector', '?')}")
    print(f"  Detections: {len(report.detections)}")
    print(f"  Captions  : {len(report.captions)}")
    if not report.detections:
        print("  WARNING   : No objects detected. Run `ai-video-analyzer doctor` to verify dependencies.")
    if not args.no_captioning and not report.captions:
        print("  NOTE      : No captions generated (BLIP may not be installed or captioning failed).")
    if not args.no_summarization and not report.summary:
        print("  NOTE      : No summary generated (start Ollama with `ollama serve`).")
    stem = Path(report.video_path).stem + "_analysis"
    json_out = Path(args.output_dir) / f"{stem}.json"
    md_out = Path(args.output_dir) / f"{stem}.md"
    print(f"\nOutput files:")
    print(f"  JSON    : {json_out}")
    print(f"  Markdown: {md_out}")
    if report.summary:
        print(f"\nSummary:\n{report.summary}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    import time as _time
    from collections import Counter
    from ai_powered_video_analyzer.frames import FrameSampler
    from ai_powered_video_analyzer.backends import load_backend

    if args.compare:
        return _run_preset_comparison(args.video, args.device, args.max_frames)

    print(f"Benchmark: {args.video}")
    print(f"  Strategy : {args.strategy}")
    print(f"  Preset   : {args.preset}")
    print()

    # Frame sampling
    sampler = FrameSampler(strategy=args.strategy, target_fps=1.0, max_frames=args.max_frames)
    t0 = _time.perf_counter()
    try:
        frames = sampler.sample(args.video)
    except Exception as exc:
        print(f"Error reading video: {exc}", file=sys.stderr)
        return 1
    t_sample = _time.perf_counter() - t0

    # Duration via cv2
    duration_sec = 0.0
    try:
        import cv2
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
        fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_sec = fc / fps
        cap.release()
    except Exception:
        pass

    # Detection
    t_load = _time.perf_counter()
    backend = load_backend("visionservex", preset=args.preset,
                           model_id=args.detector_model, device=args.device)
    backend.warmup()
    t_load = _time.perf_counter() - t_load

    all_dets = []
    t0 = _time.perf_counter()
    for r in frames:
        all_dets.extend(backend.predict(r.frame))
    t_detect = _time.perf_counter() - t0

    fps_analyzed = len(frames) / t_detect if t_detect > 0 else 0
    ms_per_frame = t_detect / len(frames) * 1000 if frames else 0
    top = Counter(d.label for d in all_dets).most_common(5)
    top_str = ", ".join(f"{l}({c})" for l, c in top) or "none"

    print(f"Results:")
    print(f"  Video duration     : {duration_sec:.1f}s")
    print(f"  Frames selected    : {len(frames)}  (sampling: {t_sample:.2f}s)")
    print(f"  Model load+warmup  : {t_load:.2f}s")
    print(f"  Detection runtime  : {t_detect:.2f}s  ({ms_per_frame:.1f}ms/frame, {fps_analyzed:.1f}fps)")
    print(f"  Total detections   : {len(all_dets)}")
    print(f"  Top labels         : {top_str}")
    print(f"  Model              : {backend.model_id}")
    return 0


def _run_preset_comparison(video_path: str, device: str, max_frames: int) -> int:
    import time as _time
    from collections import Counter
    from ai_powered_video_analyzer.frames import FrameSampler
    from ai_powered_video_analyzer.backends import load_backend

    sampler = FrameSampler(strategy="adaptive", target_fps=1.0, max_frames=max_frames)
    try:
        frames = sampler.sample(video_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Preset comparison on {video_path}  ({len(frames)} frames)\n")
    print(f"{'Preset':<10} {'Model':<12} {'ms/frame':>10} {'fps':>6} {'Dets':>6}  Top labels")
    print("-" * 72)

    for preset in ["fast", "balanced", "quality"]:
        try:
            b = load_backend("visionservex", preset=preset, device=device)
            b.warmup()
            all_dets = []
            t0 = _time.perf_counter()
            for r in frames:
                all_dets.extend(b.predict(r.frame))
            elapsed = _time.perf_counter() - t0
            ms = elapsed / len(frames) * 1000 if frames else 0
            fps = len(frames) / elapsed if elapsed > 0 else 0
            top = ", ".join(f"{l}({c})" for l, c in Counter(d.label for d in all_dets).most_common(3))
            print(f"  {preset:<8} {b.model_id:<12} {ms:>8.1f}ms {fps:>5.1f}  {len(all_dets):>5}  {top or 'none'}")
            b.unload()
        except Exception as exc:
            print(f"  {preset:<8} FAILED: {exc}")
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    try:
        import tkinter as tk
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from video_processing_gui import VideoProcessingGUI  # type: ignore[import]
        root = tk.Tk()
        VideoProcessingGUI(root)
        root.mainloop()
        return 0
    except ImportError as exc:
        print(f"GUI unavailable: {exc}", file=sys.stderr)
        return 1


def _cmd_list_models(args: argparse.Namespace) -> int:
    from ai_powered_video_analyzer.backends.visionservex_backend import list_available_detect_models
    from ai_powered_video_analyzer.backends import PRESET_MODELS
    print("\nDetection presets (--preset):")
    for preset, model_id in PRESET_MODELS.items():
        marker = "  [default]" if preset == "balanced" else ""
        print(f"  {preset:<10} → {model_id}{marker}")
    print("\nAll wired models in local VisionServeX registry:")
    models = list_available_detect_models()
    if models:
        for m in models:
            print(f"  {m}")
    else:
        print("  (Could not query registry — is visionservex installed?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
