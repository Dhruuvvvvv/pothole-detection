"""Module 3 — Transparent Bounding-Box & Confidence Visual Damage Assessment Engine."""

from .severity_engine import (
     AssessedDetection,
    PriorityHeuristicConfig,
    assess_detection,
    assess_detections_batch,
    summarize_category_assessments,
)

__all__ = [
    "AssessedDetection",
    "PriorityHeuristicConfig",
    "assess_detection",
    "assess_detections_batch",
    "summarize_category_assessments",
]
