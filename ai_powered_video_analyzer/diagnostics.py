"""Environment diagnostics — the 'doctor' command."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import NamedTuple


class CheckResult(NamedTuple):
    name: str
    ok: bool
    message: str


def run_doctor(verbose: bool = False) -> list[CheckResult]:
    """Run all environment checks and return results."""
    checks = [
        _check_python(),
        _check_ffmpeg(),
        _check_opencv(),
        _check_torch(),
        _check_ultralytics(),
        _check_whisper(),
        _check_transformers(),
        _check_panns(),
        _check_ollama(),
        _check_visionservex(),
        _check_tesseract(),
        _check_moviepy(),
    ]
    return checks


def print_doctor_report(results: list[CheckResult]) -> None:
    """Print a human-readable doctor report to stdout."""
    PASS = "\033[92m✓\033[0m"
    FAIL = "\033[91m✗\033[0m"
    WARN = "\033[93m!\033[0m"

    print("\n=== ai-video-analyzer doctor ===\n")
    all_ok = True
    for r in results:
        icon = PASS if r.ok else FAIL
        print(f"  {icon}  {r.name:<30} {r.message}")
        if not r.ok:
            all_ok = False

    print()
    if all_ok:
        print("All checks passed. You are ready to analyze videos.\n")
    else:
        print("Some checks failed. See messages above and install missing dependencies.\n")


def _check_python() -> CheckResult:
    v = sys.version_info
    ok = v >= (3, 10)
    return CheckResult(
        "Python version",
        ok,
        f"{v.major}.{v.minor}.{v.micro}" + ("" if ok else " (3.10+ required)"),
    )


def _check_ffmpeg() -> CheckResult:
    found = shutil.which("ffmpeg") is not None
    return CheckResult(
        "ffmpeg",
        found,
        "found" if found else "not found — install ffmpeg (e.g. sudo apt install ffmpeg)",
    )


def _check_opencv() -> CheckResult:
    try:
        import cv2
        return CheckResult("OpenCV (cv2)", True, cv2.__version__)
    except ImportError:
        return CheckResult("OpenCV (cv2)", False, "not installed — pip install opencv-python-headless")


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
        return CheckResult("PyTorch", False, "not installed — pip install torch")


def _check_ultralytics() -> CheckResult:
    try:
        import ultralytics
        return CheckResult("ultralytics (YOLO)", True, ultralytics.__version__)
    except ImportError:
        return CheckResult(
            "ultralytics (YOLO)", False,
            "not installed — pip install ultralytics (optional for YOLO backend)"
        )


def _check_whisper() -> CheckResult:
    try:
        import whisper
        return CheckResult("openai-whisper", True, getattr(whisper, "__version__", "installed"))
    except ImportError:
        return CheckResult("openai-whisper", False, "not installed — pip install openai-whisper")


def _check_transformers() -> CheckResult:
    try:
        import transformers
        return CheckResult("transformers (BLIP)", True, transformers.__version__)
    except ImportError:
        return CheckResult(
            "transformers (BLIP)", False, "not installed — pip install transformers"
        )


def _check_panns() -> CheckResult:
    try:
        import panns_inference  # noqa: F401
        return CheckResult("panns-inference", True, "installed")
    except ImportError:
        return CheckResult(
            "panns-inference", False,
            "not installed — pip install panns-inference (audio event detection)"
        )


def _check_ollama() -> CheckResult:
    if shutil.which("ollama") is None:
        return CheckResult(
            "Ollama", False,
            "CLI not found — install from https://ollama.com"
        )
    try:
        r = subprocess.run(
            ["ollama", "list"], capture_output=True, timeout=5, text=True
        )
        if r.returncode == 0:
            lines = r.stdout.strip().splitlines()
            model_count = max(0, len(lines) - 1)
            return CheckResult("Ollama", True, f"{model_count} model(s) available")
        return CheckResult("Ollama", False, "CLI found but not running — run: ollama serve")
    except Exception:
        return CheckResult("Ollama", False, "unable to query Ollama")


def _check_visionservex() -> CheckResult:
    try:
        import visionservex
        return CheckResult(
            "VisionServeX (optional)",
            True,
            getattr(visionservex, "__version__", "installed"),
        )
    except ImportError:
        return CheckResult(
            "VisionServeX (optional)",
            True,  # Optional — not a failure
            "not installed (optional) — pip install 'visionservex[hf,rfdetr]'",
        )


def _check_tesseract() -> CheckResult:
    found = shutil.which("tesseract") is not None
    return CheckResult(
        "Tesseract OCR (optional)",
        True,  # Optional
        "found" if found else "not found (optional) — sudo apt install tesseract-ocr",
    )


def _check_moviepy() -> CheckResult:
    try:
        import moviepy
        return CheckResult("moviepy", True, getattr(moviepy, "__version__", "installed"))
    except ImportError:
        return CheckResult("moviepy", False, "not installed — pip install moviepy")
