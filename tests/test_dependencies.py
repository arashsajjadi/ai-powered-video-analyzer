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
    assert data["project"]["version"] == "1.1.0"


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


def test_docs_user_guide_exists():
    """docs/USER_GUIDE.md must exist (added in v1.1)."""
    assert (ROOT / "docs" / "USER_GUIDE.md").exists(), (
        "docs/USER_GUIDE.md is missing — create it with user-facing documentation"
    )


def test_docs_llm_agent_guide_exists():
    """docs/LLM_AGENT_GUIDE.md must exist (added in v1.1)."""
    assert (ROOT / "docs" / "LLM_AGENT_GUIDE.md").exists(), (
        "docs/LLM_AGENT_GUIDE.md is missing — create it with agent safety rules"
    )


def test_llm_agent_guide_contains_safety_rules():
    """LLM_AGENT_GUIDE.md must contain the key safety rules."""
    guide = (ROOT / "docs" / "LLM_AGENT_GUIDE.md").read_text()
    required = [
        "Do NOT commit",       # no media/weights rule
        "Do NOT credit",       # no Claude/Anthropic attribution
        "read-only",           # VisionServeX reference is read-only
        "Do NOT",              # must have prohibitions
    ]
    for phrase in required:
        assert phrase in guide, f"LLM_AGENT_GUIDE.md missing required phrase: {phrase!r}"


def test_no_claude_anthropic_in_source_files():
    """Python source files and pyproject.toml must not contain Claude/Anthropic attribution."""
    import subprocess
    # Only check source/config files — docs may reference these patterns for educational purposes
    result = subprocess.run(
        ["grep", "-rIl", "--include=*.py", "--include=*.toml",
         "-iE", r"noreply@anthropic\.com|co-authored-by.*claude|co-authored-by.*anthropic",
         str(ROOT)],
        capture_output=True, text=True,
    )
    matching_files = [f for f in result.stdout.strip().splitlines()
                      if ".git" not in f and "__pycache__" not in f
                      and "test_dependencies.py" not in f]  # this file contains the pattern as a test string
    assert not matching_files, (
        f"Source files with Claude/Anthropic attribution found: {matching_files}"
    )


def test_readme_no_yolo_in_credits():
    """README Credits section must not contain Ultralytics/YOLO."""
    readme = (ROOT / "README.md").read_text()
    # Find the Credits section
    if "## Credits" in readme:
        credits_start = readme.index("## Credits")
        # Find end of credits section (next ## heading)
        rest = readme[credits_start:]
        next_section = rest.find("\n## ", 4)
        credits_section = rest[:next_section] if next_section > 0 else rest
        assert "ultralytics" not in credits_section.lower(), (
            "Ultralytics/YOLO must not appear in README Credits section"
        )
        assert "yolo" not in credits_section.lower() or "legacy" in credits_section.lower(), (
            "YOLO in README Credits must be in a legacy context"
        )


def test_report_has_preset_and_timings():
    """AnalysisReport must include preset, frame_strategy, and timings fields (v1.1)."""
    from ai_powered_video_analyzer.reports import AnalysisReport
    r = AnalysisReport(
        video_path="test.mp4",
        duration_sec=5.0,
        fps=25.0,
        width=640,
        height=480,
        frame_count=125,
        sampled_frame_count=5,
        backend="visionservex",
        model_ids={"detector": "dfine-s"},
        preset="quality",
        frame_strategy="scene_change",
        timings={"detection": 0.5, "captioning": 1.2},
    )
    data = r.to_dict()
    assert data["preset"] == "quality"
    assert data["frame_strategy"] == "scene_change"
    assert data["timings"]["detection"] == 0.5
    md = r.to_markdown()
    assert "quality" in md
    assert "scene_change" in md
    assert "detection" in md
