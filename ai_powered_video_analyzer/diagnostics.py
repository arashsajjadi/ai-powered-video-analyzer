"""Environment diagnostics — the 'doctor' command."""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import NamedTuple


class CheckResult(NamedTuple):
    name: str
    ok: bool
    message: str
    optional: bool = False


def run_doctor(verbose: bool = False) -> list[CheckResult]:
    """Run all environment checks and return results."""
    return [
        # --- Required ---
        _check_python(),
        _check_opencv(),
        _check_visionservex(),
        _check_dfine_model(),
        # --- Recommended ---
        _check_ffmpeg(),
        _check_torch(),
        # --- Optional pipeline stages ---
        _check_whisper(),
        _check_transformers(),
        _check_panns(),
        _check_ollama(),
        # --- Legacy / informational ---
        _check_ultralytics_optional(),
        _check_moviepy(),
        _check_tesseract(),
    ]


def print_doctor_report(results: list[CheckResult]) -> None:
    """Print a human-readable doctor report to stdout."""
    PASS = "\033[92m✓\033[0m"
    FAIL = "\033[91m✗\033[0m"
    SKIP = "\033[93m·\033[0m"  # optional, not installed

    print("\n=== ai-video-analyzer doctor ===\n")
    has_failure = False
    for r in results:
        if r.ok:
            icon = PASS
        elif r.optional:
            icon = SKIP
        else:
            icon = FAIL
            has_failure = True
        print(f"  {icon}  {r.name:<38} {r.message}")

    print()
    if has_failure:
        print("Required dependencies missing. See ✗ items above.\n")
        print("Suggested fix:")
        print("  python -m pip install -e '.[vision]'   # detection only")
        print("  python -m pip install -r pip_requirements.txt  # full stack\n")
    else:
        print("Core dependencies OK.\n")
        print("Next step:")
        print("  ai-video-analyzer analyze \"/path/to/video.mp4\" --preset balanced\n")


# --- Required checks ---

def _check_python() -> CheckResult:
    v = sys.version_info
    ok = v >= (3, 10)
    return CheckResult(
        "Python",
        ok,
        f"{v.major}.{v.minor}.{v.micro}" + ("" if ok else " (3.10+ required)"),
    )


def _check_opencv() -> CheckResult:
    try:
        import cv2
        return CheckResult("OpenCV (cv2)", True, cv2.__version__)
    except ImportError:
        return CheckResult(
            "OpenCV (cv2)", False,
            "not installed — pip install opencv-python-headless",
        )


def _check_visionservex() -> CheckResult:
    """Primary detection backend — required for object detection."""
    try:
        import visionservex
        version = getattr(visionservex, "__version__", "installed")
        return CheckResult("VisionServeX", True, f"{version} (primary detection backend)")
    except ImportError:
        return CheckResult(
            "VisionServeX", False,
            "not installed — pip install 'visionservex[hf,rfdetr]'",
        )


def _check_dfine_model() -> CheckResult:
    """Probe the default D-FINE model (dfine-s/balanced preset) in the local registry."""
    try:
        from visionservex.registry import default_registry  # type: ignore[import]
        reg = default_registry()
        entries = {m.id: m for m in reg.list()}
        target = "dfine-s"
        if target in entries:
            e = entries[target]
            status = getattr(e, "status", "?")
            impl = getattr(e, "implementation_status", "?")
            return CheckResult("  dfine-s (balanced preset)", True, f"status={status}, impl={impl}")
        dfine_ids = [k for k in entries if k.startswith("dfine")]
        if dfine_ids:
            return CheckResult(
                "  dfine-s (balanced preset)", True,
                f"dfine-s absent but found: {', '.join(sorted(dfine_ids))}",
            )
        return CheckResult(
            "  dfine-s (balanced preset)", False,
            "not found in registry — run: ai-video-analyzer list-models",
        )
    except ImportError:
        return CheckResult(
            "  dfine-s (balanced preset)", False,
            "VisionServeX not installed — registry unavailable",
        )
    except Exception as exc:
        return CheckResult("  dfine-s (balanced preset)", False, f"registry probe failed: {exc}")


# --- Recommended checks ---

def _check_ffmpeg() -> CheckResult:
    found = shutil.which("ffmpeg") is not None
    return CheckResult(
        "ffmpeg",
        found,
        "found" if found else "not found — needed for audio  (sudo apt install ffmpeg)",
        optional=not found,
    )


def _check_torch() -> CheckResult:
    try:
        import torch
        device = "cpu"
        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        return CheckResult("PyTorch", True, f"{torch.__version__} [{device}]")
    except ImportError:
        return CheckResult(
            "PyTorch", True,
            "not installed (optional — VisionServeX loads its own torch)",
            optional=True,
        )


# --- Optional pipeline stage checks ---

def _check_whisper() -> CheckResult:
    try:
        import whisper
        return CheckResult("Whisper", True, getattr(whisper, "__version__", "installed"))
    except ImportError:
        return CheckResult(
            "Whisper (optional)", True,
            "not installed — install for transcription: pip install openai-whisper",
            optional=True,
        )


def _check_transformers() -> CheckResult:
    try:
        import transformers
        return CheckResult("transformers / BLIP", True, transformers.__version__)
    except ImportError:
        return CheckResult(
            "transformers / BLIP (optional)", True,
            "not installed — install for captioning: pip install transformers",
            optional=True,
        )


def _check_panns() -> CheckResult:
    try:
        import panns_inference  # noqa: F401
        return CheckResult("panns-inference", True, "installed")
    except ImportError:
        return CheckResult(
            "panns-inference (optional)", True,
            "not installed — install for audio events: pip install panns-inference",
            optional=True,
        )


def _check_ollama() -> CheckResult:
    if shutil.which("ollama") is None:
        return CheckResult(
            "Ollama (optional)", True,
            "CLI not found — install for LLM summaries: https://ollama.com",
            optional=True,
        )
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, timeout=5, text=True)
        if r.returncode == 0:
            lines = r.stdout.strip().splitlines()
            count = max(0, len(lines) - 1)
            models = [ln.split()[0] for ln in lines[1:] if ln.strip()]
            summary = f"{count} model(s): {', '.join(models[:3])}" if models else f"{count} models"
            return CheckResult("Ollama", True, summary)
        return CheckResult(
            "Ollama (optional)", True,
            "CLI found but not running — run: ollama serve",
            optional=True,
        )
    except Exception:
        return CheckResult("Ollama (optional)", True, "unable to query Ollama", optional=True)


# --- Legacy / informational ---

def _check_ultralytics_optional() -> CheckResult:
    try:
        import ultralytics
        return CheckResult(
            "ultralytics (legacy YOLO)", True,
            f"{ultralytics.__version__} (not the default backend; use --backend legacy_yolo)",
        )
    except ImportError:
        return CheckResult(
            "ultralytics (legacy YOLO)", True,
            "not installed (not needed — VisionServeX is the default detector)",
            optional=True,
        )


def _check_moviepy() -> CheckResult:
    try:
        import moviepy
        return CheckResult("moviepy", True, getattr(moviepy, "__version__", "installed"))
    except ImportError:
        return CheckResult(
            "moviepy (optional)", True,
            "not installed — install for video output: pip install moviepy",
            optional=True,
        )


def _check_tesseract() -> CheckResult:
    found = shutil.which("tesseract") is not None
    return CheckResult(
        "Tesseract OCR (optional)", True,
        "found" if found else "not found (optional OCR tool)",
        optional=not found,
    )
