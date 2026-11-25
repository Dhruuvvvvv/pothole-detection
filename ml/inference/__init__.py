"""Deep learning object detection inference engine with singleton weight caching (Module 2)."""

from __future__ import annotations

from .detector import (
    ModelNotConfiguredError,
    RoadDamageDetector,
    draw_assessed_detections,
    get_detector_singleton,
)

__all__ = [
    "ModelNotConfiguredError",
    "RoadDamageDetector",
    "draw_assessed_detections",
    "get_detector_singleton",
]
