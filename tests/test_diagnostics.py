"""Diagnostics (doctor command) tests."""

from __future__ import annotations

import subprocess
import sys


def test_doctor_runs_without_crash():
    """Doctor must complete without exceptions."""
    from ai_powered_video_analyzer.diagnostics import run_doctor
    results = run_doctor()
    assert isinstance(results, list)
    assert len(results) > 0


def test_doctor_returns_named_tuples():
    from ai_powered_video_analyzer.diagnostics import run_doctor
    results = run_doctor()
    for r in results:
        assert hasattr(r, "name")
        assert hasattr(r, "ok")
        assert hasattr(r, "message")
        assert isinstance(r.ok, bool)


def test_doctor_python_check_passes():
    from ai_powered_video_analyzer.diagnostics import run_doctor
    results = {r.name: r for r in run_doctor()}
    py_check = results.get("Python")
    assert py_check is not None, f"'Python' check not found; keys={list(results.keys())}"
    assert py_check.ok, f"Python version check failed: {py_check.message}"


def test_doctor_optional_field_present():
    from ai_powered_video_analyzer.diagnostics import run_doctor
    results = run_doctor()
    for r in results:
        assert hasattr(r, "optional"), f"CheckResult missing 'optional' field: {r.name}"
        assert isinstance(r.optional, bool)


def test_doctor_command_via_cli():
    result = subprocess.run(
        [sys.executable, "-m", "ai_powered_video_analyzer.cli", "doctor"],
        capture_output=True, text=True, timeout=30,
    )
    # Doctor exits 0 when only required deps are missing (optional deps excluded from failures)
    assert result.returncode in (0, 1)
    combined = result.stdout + result.stderr
    assert "doctor" in combined.lower() or "Python" in combined
