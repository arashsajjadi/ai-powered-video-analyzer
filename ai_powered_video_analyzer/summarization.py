"""LLM summarization via Ollama (offline, local)."""

from __future__ import annotations

import re
import subprocess

from ai_powered_video_analyzer.logging_utils import get_logger, timed_stage

log = get_logger(__name__)

_SUMMARY_PROMPT = (
    "You are an offline video analysis assistant summarizing structured detection data. "
    "Based only on the data provided below, write a factual summary (under 150 words) of what the video shows. "
    "Follow these rules:\n"
    "- State only what the data explicitly supports.\n"
    "- If timestamps are available, reference them (e.g. 'At 3.2s, a dog appears').\n"
    "- Do not invent events, emotions, or context not present in the data.\n"
    "- If the evidence is weak or ambiguous, say so explicitly.\n"
    "- Do not mention model names, confidence scores, or technical details.\n"
    "- Separate observed facts from plausible interpretations using 'Observed:' and 'Interpretation:' labels.\n"
    "- If there is not enough evidence to summarize, respond with: 'Insufficient evidence for a meaningful summary.'"
)


def summarize_report(report_text: str, model: str = "phi4:latest") -> str:
    """
    Summarize a text report using a local Ollama LLM.

    Falls back to a generic placeholder if Ollama is unavailable.
    """
    clean = _clean_report(report_text)
    if not clean.strip():
        return ""

    with timed_stage("summarization", log):
        result = _call_ollama(_SUMMARY_PROMPT, clean, model)

    return result


def list_ollama_models() -> list[str]:
    """Return model names available in the local Ollama installation."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().splitlines()
        return [line.split()[0] for line in lines[1:] if line.strip()]
    except Exception:
        return []


def ollama_available() -> bool:
    """Check whether the `ollama` CLI is accessible."""
    try:
        r = subprocess.run(
            ["ollama", "list"],
            capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


def _call_ollama(prompt: str, input_text: str, model: str) -> str:
    combined = f"{prompt}\n\n{input_text}"
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=combined,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        if result.returncode != 0:
            log.error("Ollama exited %d: %s", result.returncode, result.stderr[:200])
            return _fallback_summary()
        output = re.sub(r"<think>.*?</think>", "", result.stdout, flags=re.DOTALL).strip()
        return output or _fallback_summary()
    except FileNotFoundError:
        log.warning("Ollama CLI not found. Install from https://ollama.com")
        return _fallback_summary()
    except subprocess.TimeoutExpired:
        log.warning("Ollama timed out.")
        return _fallback_summary()
    except Exception as exc:
        log.error("Ollama call failed: %s", exc)
        return _fallback_summary()


def _clean_report(text: str) -> str:
    text = re.sub(r"[۰-۹]+", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text


def _fallback_summary() -> str:
    return (
        "LLM summarization unavailable. "
        "Install and start Ollama to generate a natural-language summary: "
        "https://ollama.com — then run `ollama serve` and `ollama pull phi4:latest`."
    )
