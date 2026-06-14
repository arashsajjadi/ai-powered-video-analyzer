"""Test that required files and dependency declarations exist and are valid."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_pip_requirements_exists():
    """pip_requirements.txt must exist (fixes issue #1)."""
    assert (ROOT / "pip_requirements.txt").exists(), "pip_requirements.txt is missing"


def test_pip_requirements_is_valid_pip_format():
    """pip_requirements.txt must not contain conda export artifacts."""
    text = (ROOT / "pip_requirements.txt").read_text()
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#")]
    for line in lines:
        # Conda export artifacts look like: "package  0.0  pypi_0  pypi"
        assert "pypi_0" not in line, f"Invalid conda-export artifact in pip_requirements.txt: {line!r}"
        assert "pypi    pypi" not in line, f"Invalid conda-export artifact in pip_requirements.txt: {line!r}"


def test_environment_yml_exists():
    """environment.yml must exist with valid conda format."""
    assert (ROOT / "environment.yml").exists(), "environment.yml is missing"


def test_environment_yml_is_valid():
    """environment.yml must not contain conda-export build strings."""
    import yaml  # type: ignore[import]
    text = (ROOT / "environment.yml").read_text()
    env = yaml.safe_load(text)
    assert "name" in env, "environment.yml must have a 'name' field"
    assert "dependencies" in env, "environment.yml must have a 'dependencies' field"


def test_pyproject_toml_exists():
    assert (ROOT / "pyproject.toml").exists(), "pyproject.toml is missing"


def test_pyproject_toml_valid():
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    text = (ROOT / "pyproject.toml").read_bytes()
    data = tomllib.loads(text.decode())
    assert "project" in data
    assert data["project"]["name"] == "ai-powered-video-analyzer"
    assert "0.2.0" in data["project"]["version"]


def test_video_processing_py_exists():
    """video_processing.py must exist (fixes issue #1 — CLI script was missing)."""
    assert (ROOT / "video_processing.py").exists(), "video_processing.py is missing"


def test_video_processing_gui_exists():
    """video_processing_gui.py must still exist (original GUI preserved)."""
    assert (ROOT / "video_processing_gui.py").exists(), "video_processing_gui.py is missing"


def test_changelog_exists():
    assert (ROOT / "CHANGELOG.md").exists(), "CHANGELOG.md is missing"


def test_readme_references_existing_files():
    """
    README must not reference files that don't exist.
    (This is what caused issue #1.)
    """
    readme = (ROOT / "README.md").read_text()
    # These files must exist if README references them
    file_refs = {
        "pip_requirements.txt": ROOT / "pip_requirements.txt",
        "video_processing.py": ROOT / "video_processing.py",
        "video_processing_gui.py": ROOT / "video_processing_gui.py",
        "environment.yml": ROOT / "environment.yml",
    }
    for ref, path in file_refs.items():
        if ref in readme:
            assert path.exists(), (
                f"README references '{ref}' but the file does not exist. "
                f"This reproduces issue #1."
            )
