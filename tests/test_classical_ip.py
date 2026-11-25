"""Unit tests for Phase C — Classical Digital Image Processing Pipeline."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ml.preprocessing.classical_ops import (
    OPERATION_CATALOG,
    apply_classical_operation,
    compare_filtering_techniques,
)
from ml.preprocessing.experiment_plots import generate_classical_ip_experiment_suite
from ml.preprocessing.metrics import compute_quantitative_metrics


@pytest.fixture
def synthetic_asphalt_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    img = rng.integers(90, 140, size=(240, 320, 3), dtype=np.uint8)
    # Simulated dark pothole & longitudinal crack
    cv2.circle(img, (180, 140), 28, (25, 25, 28), -1)
    cv2.line(img, (60, 20), (75, 220), (20, 20, 20), 3)
    return img


def test_all_catalog_operations_execute_and_produce_metrics(synthetic_asphalt_image: np.ndarray) -> None:
    assert len(OPERATION_CATALOG) >= 20
    for entry in OPERATION_CATALOG:
        op_id = entry["id"]
        defaults = {p["name"]: p["default"] for p in entry["parameters"]}
        if op_id == "resize":
            defaults = {"width": 256, "height": 192, "interpolation": "bilinear"}

        proc, diff, tel = apply_classical_operation(synthetic_asphalt_image, op_id, defaults)
        assert proc is not None and proc.dtype == np.uint8
        assert diff is not None and diff.shape[:2] == proc.shape[:2]
        assert tel["processing_time_ms"] >= 0.0
        pm = tel["processed_metrics"]
        assert 0.0 <= pm["mean_intensity"] <= 255.0
        assert pm["edge_pixel_density"] >= 0.0

        if op_id != "resize":
            assert pm["psnr_db"] is not None
            assert pm["ssim_score"] is not None
        else:
            assert proc.shape[:2] == (192, 256)
            assert pm["psnr_db"] is None  # Different resolution -> reference PSNR omitted


def test_identical_reference_yields_perfect_ssim(synthetic_asphalt_image: np.ndarray) -> None:
    m = compute_quantitative_metrics(synthetic_asphalt_image, reference_image=synthetic_asphalt_image)
    assert m.ssim_score == 1.0
    assert m.psnr_db == 100.0


def test_filter_comparison_and_experiment_suite(
    synthetic_asphalt_image: np.ndarray,
    tmp_path: Path,
) -> None:
    comp = compare_filtering_techniques(synthetic_asphalt_image, kernel_size=5)
    assert len(comp) == 4
    ops = [c["operation"] for c in comp]
    assert ops == ["gaussian_blur", "median_blur", "bilateral_filter", "average_blur"]

    img_p = tmp_path / "road_lab_input.jpg"
    cv2.imwrite(str(img_p), synthetic_asphalt_image)
    suite = generate_classical_ip_experiment_suite(img_p, tmp_path / "suite_out")
    assert Path(suite["figure_path"]).is_file()
    assert Path(suite["metrics_csv_path"]).is_file()


def test_invalid_operation_raises(synthetic_asphalt_image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        apply_classical_operation(synthetic_asphalt_image, "non_existent_filter", {})
