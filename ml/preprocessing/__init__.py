"""Classical Digital Image Processing Pipeline (Module 1)."""

from .classical_ops import (
    OPERATION_CATALOG,
    apply_classical_operation,
    compare_filtering_techniques,
    get_operation_catalog,
)
from .metrics import (
    ImageQuantitativeMetrics,
    compute_channel_histograms,
    compute_difference_visualization,
    compute_quantitative_metrics,
)
from .experiment_plots import generate_classical_ip_experiment_suite

__all__ = [
    "ImageQuantitativeMetrics",
    "OPERATION_CATALOG",
    "apply_classical_operation",
    "compare_filtering_techniques",
    "compute_channel_histograms",
    "compute_difference_visualization",
    "compute_quantitative_metrics",
    "generate_classical_ip_experiment_suite",
    "get_operation_catalog",
]
