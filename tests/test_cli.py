"""CLI smoke tests — no model downloads required."""

from __future__ import annotations

import subprocess
import sys


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "ai_powered_video_analyzer.cli", *args],
        capture_output=True,
        text=True,
    )


def test_cli_help():
    result = _run("--help")
    assert result.returncode == 0
    assert "analyze" in result.stdout
    assert "doctor" in result.stdout
    assert "benchmark" in result.stdout


def test_cli_analyze_help():
    result = _run("analyze", "--help")
    assert result.returncode == 0
    assert "--video" in result.stdout or "video" in result.stdout


def test_cli_doctor_help():
    result = _run("doctor", "--help")
    assert result.returncode == 0


def test_cli_version():
    result = _run("--version")
    assert result.returncode == 0
    assert "0.2.0" in result.stdout


def test_video_processing_py_help():
    """The compatibility shim must accept --help."""
    import subprocess
    from pathlib import Path
    script = Path(__file__).parent.parent / "video_processing.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--video" in result.stdout


def test_cli_analyze_missing_file():
    """Analyzing a nonexistent file must exit with code 1."""
    result = _run("analyze", "/nonexistent/video_that_does_not_exist.mp4")
    assert result.returncode == 1
