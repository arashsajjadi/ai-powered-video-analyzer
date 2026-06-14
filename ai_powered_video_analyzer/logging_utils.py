"""Structured logging setup for ai-powered-video-analyzer."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Generator

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(
    debug: bool = False,
    verbose: bool = False,
    log_file: str | None = None,
) -> None:
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(level=level, format=_LOG_FORMAT, handlers=handlers, force=True)
    # Silence noisy third-party loggers at WARNING and below.
    for noisy in ("huggingface_hub", "transformers", "ultralytics", "moviepy"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def timed_stage(name: str, logger: logging.Logger | None = None) -> Generator[None, None, None]:
    """Context manager that logs elapsed time for a pipeline stage."""
    log = logger or logging.getLogger(__name__)
    log.info("Stage [%s] starting", name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        log.info("Stage [%s] completed in %.2fs", name, elapsed)
