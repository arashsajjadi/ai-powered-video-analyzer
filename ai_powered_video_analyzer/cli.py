"""CLI entry point for ai-powered-video-analyzer."""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-video-analyzer",
        description=(
            "ai-powered-video-analyzer v0.3.0 — offline, privacy-first AI video understanding.\n"
            "Primary detector: VisionServeX D-FINE/RF-DETR (since v0.3.0).\n"
            "Analysis stages: detection, captioning (BLIP), transcription (Whisper),\n"
            "audio events (PANNs), and LLM summarization (Ollama)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=_get_version())

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- analyze ---
    p_analyze = subparsers.add_parser("analyze", help="Analyze a video file.")
    p_analyze.add_argument("video", help="Path to the input video file.")
    p_analyze.add_argument("--output-dir", default=".", help="Output directory for reports.")
    p_analyze.add_argument(
        "--strategy",
        choices=["uniform", "adaptive", "scene_change", "motion_aware", "hybrid"],
        default="adaptive",
        help="Frame sampling strategy (default: adaptive).",
    )
    p_analyze.add_argument("--target-fps", type=float, default=1.0,
                           help="Target analyzed frames per second.")

    # Detection backend
    p_analyze.add_argument(
        "--backend",
        choices=["auto", "visionservex", "none", "legacy_yolo"],
        default="visionservex",
        help="Detection backend (default: visionservex).",
    )
    p_analyze.add_argument(
        "--detector-preset",
        choices=["fast", "balanced", "quality", "quality+"],
        default="balanced",
        help=(
            "Detector preset (default: balanced).\n"
            "  fast      → dfine-n  (~16ms/frame, good accuracy)\n"
            "  balanced  → dfine-s  (~13ms/frame, better accuracy) [DEFAULT]\n"
            "  quality   → dfine-m  (~15ms/frame, best COCO accuracy)\n"
            "  quality+  → dfine-l  (large model, highest accuracy)"
        ),
    )
    p_analyze.add_argument("--detector-model", default="",
                           help="Explicit VisionServeX model ID (overrides --detector-preset).")
    p_analyze.add_argument("--detection-confidence", type=float, default=0.3,
                           help="Minimum detection confidence threshold (default: 0.3).")
    p_analyze.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Compute device (default: auto).",
    )

    # Audio / captioning / transcription
    p_analyze.add_argument("--whisper-model", default="base",
                           help="Whisper model size (tiny/base/small/medium/large-v2).")
    p_analyze.add_argument("--language", default=None,
                           help="Force transcription language (e.g. en, fa, es).")
    p_analyze.add_argument("--no-captioning", action="store_true",
                           help="Skip BLIP image captioning.")
    p_analyze.add_argument("--no-audio-events", action="store_true",
                           help="Skip PANNs audio event detection.")

    # Summarization
    p_analyze.add_argument("--ollama-model", default="",
                           help="Ollama model for summarization (default: auto-discovered).")
    p_analyze.add_argument(
        "--summary-style",
        choices=["concise", "evidence", "technical", "narrative"],
        default="concise",
        help="Summarization style (default: concise).",
    )
    p_analyze.add_argument("--no-summarization", action="store_true",
                           help="Skip Ollama LLM summarization.")
    p_analyze.add_argument("--list-ollama-models", action="store_true",
                           help="Print available Ollama models and exit.")

    # Output
    p_analyze.add_argument("--save-annotated-video", action="store_true",
                           help="Write annotated output video.")
    p_analyze.add_argument("--max-frames", type=int, default=2000,
                           help="Maximum frames to analyze.")

    # Behavior
    p_analyze.add_argument("--verbose", "-v", action="store_true",
                           help="Enable verbose logging.")
    p_analyze.add_argument("--debug", action="store_true",
                           help="Enable debug logging.")
    p_analyze.add_argument("--strict", action="store_true",
                           help="Fail on any missing model or dependency.")

    # --- doctor ---
    p_doctor = subparsers.add_parser("doctor", help="Check system dependencies.")
    p_doctor.add_argument("--verbose", "-v", action="store_true")

    # --- benchmark ---
    p_bench = subparsers.add_parser("benchmark", help="Benchmark detection pipeline on a video.")
    p_bench.add_argument("video", help="Path to video file.")
    p_bench.add_argument("--backend", choices=["auto", "visionservex", "none", "legacy_yolo"],
                         default="visionservex")
    p_bench.add_argument("--detector-preset", choices=["fast", "balanced", "quality", "quality+"],
                         default="balanced")
    p_bench.add_argument("--detector-model", default="")
    p_bench.add_argument("--strategy", default="adaptive")
    p_bench.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p_bench.add_argument("--compare-detectors", action="store_true",
                         help="Compare fast/balanced/quality presets.")

    # --- eval-backends ---
    p_eval = subparsers.add_parser(
        "eval-backends", help="Evaluate available VisionServeX detector presets."
    )
    p_eval.add_argument("video", help="Video file for evaluation.")
    p_eval.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p_eval.add_argument("--output-dir", default="reports/benchmarks/v0.3.0/")

    # --- gui ---
    subparsers.add_parser("gui", help="Launch the Tkinter GUI.")

    # --- list-models ---
    subparsers.add_parser("list-models", help="List available VisionServeX detection models.")

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
    if args.command == "eval-backends":
        return _cmd_eval_backends(args)
    if args.command == "gui":
        return _cmd_gui()
    if args.command == "list-models":
        return _cmd_list_models()

    parser.print_help()
    return 1


# --- subcommand implementations ---

def _cmd_doctor(args: argparse.Namespace) -> int:
    from ai_powered_video_analyzer.diagnostics import print_doctor_report, run_doctor
    results = run_doctor(verbose=getattr(args, "verbose", False))
    print_doctor_report(results)
    critical_failures = [r for r in results if not r.ok and "optional" not in r.name.lower()]
    return 0 if not critical_failures else 1


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
            print("No Ollama models found. Is Ollama running? Run: ollama serve")
        return 0

    from ai_powered_video_analyzer.config import AnalysisConfig
    from ai_powered_video_analyzer.core import analyze_video

    config = AnalysisConfig(
        video_path=args.video,
        output_dir=args.output_dir,
        frame_strategy=args.strategy,
        target_fps=args.target_fps,
        backend=args.backend,
        detector_model=args.detector_model,
        detector_preset=args.detector_preset,
        detection_confidence=args.detection_confidence,
        device=args.device,
        whisper_model=args.whisper_model,
        ollama_model=args.ollama_model or "",
        summary_style=args.summary_style,
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
    print(f"  Video           : {report.video_path}")
    print(f"  Duration        : {report.duration_sec:.1f}s")
    print(f"  Frames analyzed : {report.sampled_frame_count} / {report.frame_count}")
    print(f"  Detections      : {len(report.detections)}")
    print(f"  Captions        : {len(report.captions)}")
    print(f"  Backend         : {report.backend}")
    if report.summary:
        print(f"\nSummary:\n{report.summary}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    import time as _time
    from ai_powered_video_analyzer.frames import FrameSampler
    from ai_powered_video_analyzer.backends import load_backend
    from ai_powered_video_analyzer.logging_utils import setup_logging

    setup_logging(verbose=True)

    if args.compare_detectors:
        return _run_detector_comparison(args.video, args.device)

    print(f"\nBenchmarking: {args.video}")
    print(f"  Backend  : {args.backend}")
    print(f"  Preset   : {args.detector_preset}")
    print(f"  Strategy : {args.strategy}")

    sampler_u = FrameSampler(strategy="uniform", target_fps=1.0)
    t0 = _time.perf_counter()
    try:
        frames_u = sampler_u.sample(args.video)
    except Exception as exc:
        print(f"Error reading video: {exc}", file=sys.stderr)
        return 1
    t_uniform = _time.perf_counter() - t0

    sampler_a = FrameSampler(strategy=args.strategy, target_fps=1.0)
    t0 = _time.perf_counter()
    frames_a = sampler_a.sample(args.video)
    t_adaptive = _time.perf_counter() - t0

    reduction = 1.0 - len(frames_a) / max(len(frames_u), 1)
    speedup = t_uniform / t_adaptive if t_adaptive > 0 else 0

    backend = load_backend(args.backend, preset=args.detector_preset,
                           model_id=args.detector_model, device=args.device)
    backend.warmup()

    t0 = _time.perf_counter()
    total_dets = 0
    for r in frames_a[:100]:
        dets = backend.predict(r.frame)
        total_dets += len(dets)
    t_detect = _time.perf_counter() - t0
    frames_tested = min(100, len(frames_a))
    ms_per_frame = t_detect / frames_tested * 1000 if frames_tested else 0

    print(f"\n--- Results ---")
    print(f"  Uniform   frames: {len(frames_u):>6}   time: {t_uniform:.2f}s")
    print(f"  {args.strategy:<9} frames: {len(frames_a):>6}   time: {t_adaptive:.2f}s")
    print(f"  Frame reduction : {reduction:.1%}")
    print(f"  Sample speedup  : {speedup:.2f}×")
    print(f"  Detection (first {frames_tested} frames): {t_detect:.2f}s  ({ms_per_frame:.1f}ms/frame)")
    print(f"  Total detections: {total_dets}")
    print(f"  Backend model   : {backend.model_id}")
    return 0


def _run_detector_comparison(video_path: str, device: str) -> int:
    import time as _time
    from ai_powered_video_analyzer.frames import FrameSampler
    from ai_powered_video_analyzer.backends import load_backend

    sampler = FrameSampler(strategy="adaptive", target_fps=1.0, max_frames=20)
    try:
        frames = sampler.sample(video_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nDetector comparison on {video_path} ({len(frames)} sampled frames)")
    print(f"{'Preset':<12} {'Model':<15} {'ms/frame':>10} {'Total dets':>12} {'Top labels'}")
    print("-" * 80)

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
            from collections import Counter
            top = Counter(d.label for d in all_dets).most_common(3)
            top_str = ", ".join(f"{l}({c})" for l, c in top) or "none"
            print(f"  {preset:<10} {b.model_id:<15} {ms:>8.1f}ms {len(all_dets):>10} {top_str}")
            b.unload()
        except Exception as exc:
            print(f"  {preset:<10} {'FAILED':<15} — {exc}")
    return 0


def _cmd_eval_backends(args: argparse.Namespace) -> int:
    import time as _time
    import json, os
    from ai_powered_video_analyzer.frames import FrameSampler
    from ai_powered_video_analyzer.backends import load_backend

    print(f"\nEvaluating detector backends on: {args.video}")
    sampler = FrameSampler(strategy="adaptive", target_fps=1.0, max_frames=30)
    try:
        frames = sampler.sample(args.video)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Sampled {len(frames)} frames.\n")

    results = []
    for preset in ["fast", "balanced", "quality"]:
        try:
            t_load = _time.perf_counter()
            b = load_backend("visionservex", preset=preset, device=args.device)
            b.warmup()
            load_time = _time.perf_counter() - t_load

            all_dets = []
            t0 = _time.perf_counter()
            for r in frames:
                all_dets.extend(b.predict(r.frame, confidence=0.3))
            infer_time = _time.perf_counter() - t0

            from collections import Counter
            top = Counter(d.label for d in all_dets).most_common(5)
            avg_ms = infer_time / len(frames) * 1000 if frames else 0
            row = {
                "preset": preset,
                "model": b.model_id,
                "load_time_sec": round(load_time, 2),
                "frames_tested": len(frames),
                "total_dets": len(all_dets),
                "avg_ms_per_frame": round(avg_ms, 1),
                "top_labels": dict(top),
            }
            results.append(row)
            top_str = ", ".join(f"{l}({c})" for l, c in top[:3]) or "none"
            print(
                f"  {preset:<12} {b.model_id:<15} load={load_time:.1f}s "
                f"avg={avg_ms:.1f}ms dets={len(all_dets)} top=[{top_str}]"
            )
            b.unload()
        except Exception as exc:
            print(f"  {preset:<12} FAILED: {exc}")
            results.append({"preset": preset, "error": str(exc)})

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "eval_backends.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")
    return 0


def _cmd_gui() -> int:
    try:
        import tkinter as tk
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from video_processing_gui import VideoProcessingGUI  # type: ignore[import]
        root = tk.Tk()
        VideoProcessingGUI(root)
        root.mainloop()
        return 0
    except ImportError as exc:
        print(f"GUI unavailable: {exc}", file=sys.stderr)
        return 1


def _cmd_list_models() -> int:
    from ai_powered_video_analyzer.backends.visionservex_backend import list_available_detect_models
    from ai_powered_video_analyzer.backends import PRESET_MODELS
    print("\nVisionServeX detection model presets:")
    for preset, model_id in PRESET_MODELS.items():
        print(f"  {preset:<12} → {model_id}")
    print("\nAll available detection models in local VisionServeX registry:")
    models = list_available_detect_models()
    if models:
        for m in models:
            print(f"  {m}")
    else:
        print("  (Could not query registry — is visionservex installed?)")
    return 0


def _get_version() -> str:
    from ai_powered_video_analyzer import __version__
    return f"ai-video-analyzer {__version__}"


if __name__ == "__main__":
    sys.exit(main())
