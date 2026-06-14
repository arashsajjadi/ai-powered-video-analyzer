"""Test that required files and dependency declarations exist and are valid."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_pip_requirements_exists():
    """pip_requirements.txt must exist (fixes original issue #1)."""
    assert (ROOT / "pip_requirements.txt").exists(), "pip_requirements.txt is missing"


def test_pip_requirements_is_valid_pip_format():
    """pip_requirements.txt must not contain conda export artifacts."""
    text = (ROOT / "pip_requirements.txt").read_text()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    for line in lines:
        assert "pypi_0" not in line, f"Invalid conda-export artifact: {line!r}"
        assert "pypi    pypi" not in line, f"Invalid conda-export artifact: {line!r}"


def test_no_broken_conda_requirements():
    """conda_requirements.txt (the broken Windows conda dump from issue #1) must not be tracked."""
    assert not (ROOT / "conda_requirements.txt").exists(), (
        "conda_requirements.txt still exists — remove it with: git rm conda_requirements.txt"
    )


def test_pyproject_toml_exists():
    assert (ROOT / "pyproject.toml").exists(), "pyproject.toml is missing"


def test_pyproject_toml_valid():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads((ROOT / "pyproject.toml").read_bytes().decode())
    assert "project" in data
    assert data["project"]["name"] == "ai-powered-video-analyzer"
    assert data["project"]["version"] == "1.0.0"


def test_pyproject_has_vision_extra():
    """[vision] extra must exist and include visionservex."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads((ROOT / "pyproject.toml").read_bytes().decode())
    extras = data["project"]["optional-dependencies"]
    assert "vision" in extras, "Missing [vision] extra in pyproject.toml"
    assert any("visionservex" in dep for dep in extras["vision"]), (
        "[vision] extra must contain visionservex"
    )


def test_pyproject_ultralytics_not_in_base_or_vision():
    """Ultralytics must not be a base or vision dependency (not the default backend)."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads((ROOT / "pyproject.toml").read_bytes().decode())
    base = data["project"].get("dependencies", [])
    vision = data["project"].get("optional-dependencies", {}).get("vision", [])
    for dep_list, name in [(base, "base"), (vision, "[vision]")]:
        for dep in dep_list:
            assert "ultralytics" not in dep.lower(), (
                f"ultralytics must not be in {name} dependencies (it is legacy)"
            )


def test_video_processing_py_exists():
    """video_processing.py must exist (CLI compatibility shim from issue #1)."""
    assert (ROOT / "video_processing.py").exists(), "video_processing.py is missing"


def test_video_processing_gui_exists():
    """video_processing_gui.py must still exist (original GUI preserved)."""
    assert (ROOT / "video_processing_gui.py").exists(), "video_processing_gui.py is missing"


def test_changelog_exists():
    assert (ROOT / "CHANGELOG.md").exists(), "CHANGELOG.md is missing"


def test_readme_references_existing_files():
    """README must not reference files that don't exist (reproduces original issue #1)."""
    readme = (ROOT / "README.md").read_text()
    must_exist = {
        "pip_requirements.txt": ROOT / "pip_requirements.txt",
        "video_processing.py": ROOT / "video_processing.py",
        "video_processing_gui.py": ROOT / "video_processing_gui.py",
    }
    for ref, path in must_exist.items():
        if ref in readme:
            assert path.exists(), f"README references '{ref}' but it does not exist"


def test_author_is_arash():
    """Package author must be Arash Sajjadi, not an AI assistant."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads((ROOT / "pyproject.toml").read_bytes().decode())
    authors = data["project"].get("authors", [])
    names = [a.get("name", "") for a in authors]
    assert any("Arash" in n for n in names), f"Author must be Arash Sajjadi, got: {names}"
    assert not any("Claude" in n or "Anthropic" in n for n in names), (
        "Claude/Anthropic must not be listed as an author"
    )
