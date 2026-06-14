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
    assert "video" in result.stdout.lower()


def test_cli_doctor_help():
    result = _run("doctor", "--help")
    assert result.returncode == 0


def test_cli_benchmark_help():
    result = _run("benchmark", "--help")
    assert result.returncode == 0
    assert "preset" in result.stdout.lower()


def test_cli_version():
    result = _run("--version")
    assert result.returncode == 0
    assert "1.0.0" in result.stdout


def test_cli_analyze_has_preset():
    result = _run("analyze", "--help")
    assert result.returncode == 0
    assert "--preset" in result.stdout


def test_cli_analyze_has_summary_style():
    result = _run("analyze", "--help")
    assert result.returncode == 0
    assert "summary" in result.stdout.lower()


def test_cli_list_models():
    result = _run("list-models")
    assert result.returncode in (0, 1)


def test_video_processing_py_help():
    """The compatibility shim must accept --help."""
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


def test_cli_benchmark_help_has_compare():
    result = _run("benchmark", "--help")
    assert result.returncode == 0
    assert "--compare" in result.stdout
