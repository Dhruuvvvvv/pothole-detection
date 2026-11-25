"""Structured logging configuration for RoadGuard AI backend."""

from __future__ import annotations

import logging
import sys

from backend.app.core.config import settings


def configure_logging() -> logging.Logger:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("roadguard")


logger = configure_logging()
