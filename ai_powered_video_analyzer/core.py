"""Main video analysis pipeline."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from ai_powered_video_analyzer.config import AnalysisConfig
from ai_powered_video_analyzer.logging_utils import get_logger, timed_stage
from ai_powered_video_analyzer.reports import AnalysisReport, save_report

log = get_logger(__name__)


def analyze_video(config: AnalysisConfig) -> AnalysisReport:
    """
    Run the complete offline video analysis pipeline.

    Stages
    ------
    1. Frame sampling
    2. Object detection (per frame, batched when possible)
    3. Image captioning (per frame)
    4. Audio extraction
    5. Speech transcription
    6. Audio event detection
    7. LLM summarization
    8. Report generation
    """
    timings: dict[str, float] = {}
    limitations: list[str] = []

    video_path = config.video_path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # --- Video metadata ---
    meta = _get_video_metadata(video_path)

    # --- Frame sampling ---
    t0 = time.perf_counter()
    from ai_powered_video_analyzer.frames import FrameSampler
    sampler = FrameSampler(
        strategy=config.frame_strategy,
        target_fps=config.target_fps,
        min_fps=config.min_fps,
        max_fps=config.max_fps,
        scene_threshold=config.scene_threshold,
        motion_threshold=config.motion_threshold,
        max_frames=config.max_frames,
    )
    with timed_stage("frame_sampling", log):
        frame_records = sampler.sample(video_path)
    timings["frame_sampling"] = time.perf_counter() - t0
    log.info("Sampled %d frames from %d total.", len(frame_records), meta["frame_count"])

    # --- Detection backend ---
    from ai_powered_video_analyzer.backends import load_backend
    backend = load_backend(config.backend, config.detector_model, config.device)
    if isinstance(backend, type) or not hasattr(backend, 'predict'):
        limitations.append("Object detection unavailable (no backend loaded).")

    # --- Object detection ---
    t0 = time.perf_counter()
    all_detections: list[dict] = []
    frame_observations: list[dict] = []

    with timed_stage("object_detection", log):
        frames = [r.frame for r in frame_records]
        batch_results = backend.predict_batch(frames, config.detection_confidence)
        for record, dets in zip(frame_records, batch_results):
            for d in dets:
                d.frame_index = record.frame_index
                d.timestamp_sec = record.timestamp_sec
                all_detections.append(d.to_dict())
            frame_observations.append({
                **record.to_dict(),
                "detections": [d.to_dict() for d in dets],
            })
    timings["object_detection"] = time.perf_counter() - t0

    if not all_detections:
        limitations.append(
            f"No objects detected (backend={backend.backend_name}, model={backend.model_id}). "
            "Ensure model weights are available."
        )

    # --- Captioning ---
    all_captions: list[dict] = []
    if config.enable_captioning:
        t0 = time.perf_counter()
        with timed_stage("captioning", log):
            try:
                from ai_powered_video_analyzer.captioning import caption_frame
                for record in frame_records:
                    cap = caption_frame(record.frame, config.blip_model)
                    if cap:
                        all_captions.append({
                            "frame_index": record.frame_index,
                            "timestamp_sec": record.timestamp_sec,
                            "caption": cap,
                        })
            except ImportError as exc:
                log.warning("Captioning unavailable: %s", exc)
                limitations.append("BLIP captioning unavailable (transformers not installed).")
        timings["captioning"] = time.perf_counter() - t0

    # --- Audio ---
    transcript = ""
    detected_lang = "unknown"
    audio_events: dict[str, list[str]] = {}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_audio = tmp.name

    try:
        t0 = time.perf_counter()
        with timed_stage("audio_extraction", log):
            from ai_powered_video_analyzer.audio import extract_audio
            audio_ok = extract_audio(video_path, tmp_audio)
        timings["audio_extraction"] = time.perf_counter() - t0

        if audio_ok and os.path.exists(tmp_audio):
            # Transcription
            t0 = time.perf_counter()
            with timed_stage("transcription", log):
                try:
                    from ai_powered_video_analyzer.transcription import transcribe
                    transcript, detected_lang = transcribe(
                        tmp_audio,
                        model_name=config.whisper_model,
                        language=config.transcription_language,
                    )
                except ImportError as exc:
                    log.warning("Transcription unavailable: %s", exc)
                    limitations.append("Whisper transcription unavailable.")
            timings["transcription"] = time.perf_counter() - t0

            # Audio events
            if config.enable_audio_events:
                t0 = time.perf_counter()
                with timed_stage("audio_events", log):
                    from ai_powered_video_analyzer.audio import detect_audio_events
                    audio_events = detect_audio_events(tmp_audio, config.pann_model_path)
                timings["audio_events"] = time.perf_counter() - t0
        else:
            limitations.append("No audio track found in video.")
    finally:
        if os.path.exists(tmp_audio):
            os.remove(tmp_audio)

    # --- Summarization ---
    summary = ""
    if config.enable_summarization:
        t0 = time.perf_counter()
        with timed_stage("summarization", log):
            try:
                from ai_powered_video_analyzer.summarization import summarize_report
                raw_report_text = _build_summary_input(
                    transcript, audio_events, all_detections, all_captions
                )
                summary = summarize_report(raw_report_text, model=config.ollama_model)
            except Exception as exc:
                log.warning("Summarization failed: %s", exc)
                limitations.append("LLM summarization failed or Ollama not running.")
        timings["summarization"] = time.perf_counter() - t0

    # --- Timing note ---
    log.info("Stage timings: %s", {k: f"{v:.2f}s" for k, v in timings.items()})

    report = AnalysisReport(
        video_path=video_path,
        duration_sec=meta["duration"],
        fps=meta["fps"],
        width=meta["width"],
        height=meta["height"],
        frame_count=meta["frame_count"],
        sampled_frame_count=len(frame_records),
        backend=backend.backend_name,
        model_ids={
            "detector": backend.model_id,
            "whisper": config.whisper_model,
            "blip": config.blip_model if config.enable_captioning else "disabled",
            "ollama": config.ollama_model if config.enable_summarization else "disabled",
        },
        transcript=transcript,
        transcript_language=detected_lang,
        audio_events=audio_events,
        frame_observations=frame_observations,
        detections=all_detections,
        captions=all_captions,
        summary=summary,
        limitations=limitations,
    )

    # --- Write reports ---
    if config.save_json or config.save_markdown:
        stem = Path(video_path).stem + "_analysis"
        save_report(report, config.output_dir, stem=stem)

    return report


def _get_video_metadata(video_path: str) -> dict:
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()
        return {"fps": fps, "frame_count": frame_count, "width": width, "height": height, "duration": duration}
    except ImportError:
        return {"fps": 25.0, "frame_count": 0, "width": 0, "height": 0, "duration": 0.0}


def _build_summary_input(
    transcript: str,
    audio_events: dict,
    detections: list[dict],
    captions: list[dict],
) -> str:
    from collections import Counter
    parts: list[str] = []
    if transcript:
        parts.append(f"Speech Transcript:\n{transcript}")
    if audio_events:
        ae_str = "; ".join(f"{k}" for k in list(audio_events.keys())[:10])
        parts.append(f"Audio Events: {ae_str}")
    if detections:
        counts = Counter(d["label"] for d in detections)
        top = ", ".join(f"{l}({n})" for l, n in counts.most_common(10))
        parts.append(f"Detected Objects: {top}")
    if captions:
        sample = [c["caption"] for c in captions[:5]]
        parts.append("Scene Captions:\n" + "\n".join(f"- {c}" for c in sample))
    return "\n\n".join(parts)
