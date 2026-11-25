"""Module 5 — Model Explainability, Classical-IP-vs-Detection Experiments, and Error Taxonomy."""

from .ip_vs_detection import (
    COMMON_ERROR_SOURCES,
    compute_yolo_eigencam,
    run_confidence_threshold_sweep,
    run_preprocessing_detection_experiment,
)

__all__ = [
    "COMMON_ERROR_SOURCES",
    "compute_yolo_eigencam",
    "run_confidence_threshold_sweep",
    "run_preprocessing_detection_experiment",
]
