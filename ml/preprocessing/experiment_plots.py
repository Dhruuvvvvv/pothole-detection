"""Batch experimental comparison plot generator for Classical Image Processing Lab."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml.preprocessing.classical_ops import apply_classical_operation


def generate_classical_ip_experiment_suite(
    image_path: Path | str,
    output_dir: Path | str,
) -> Dict[str, str]:
    """
    Run the 6 canonical classical IP transformations on an input road image,
    save a publication-grade comparison figure and quantitative CSV table.
    """
    in_path = Path(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(in_path))
    if bgr is None:
        raise ValueError(f"Could not decode image: {in_path}")

    variants = [
        ("Original", None, {}),
        ("Gaussian (k=5, σ=1.4)", "gaussian_blur", {"kernel_size": 5, "sigma": 1.4}),
        ("Median (k=5)", "median_blur", {"kernel_size": 5}),
        ("Bilateral (d=9)", "bilateral_filter", {"d": 9, "sigma_color": 75.0, "sigma_space": 75.0}),
        ("CLAHE (clip=2.5)", "clahe", {"clip_limit": 2.5, "tile_grid_size": 8, "preserve_color": True}),
        ("Gamma (γ=0.7)", "gamma_correction", {"gamma": 0.7}),
        ("Unsharp Mask", "unsharp_masking", {"kernel_size": 5, "sigma": 1.5, "amount": 1.4, "threshold": 3}),
        ("Canny Edges", "canny_edge", {"threshold1": 75, "threshold2": 175, "aperture_size": 3}),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=140)
    fig.patch.set_facecolor("#0B101E")
    csv_rows: List[Dict[str, object]] = []

    # Baseline metrics
    _, _, base_tel = apply_classical_operation(bgr, "normalize", {"mode": "minmax"})
    orig_m = base_tel["original_metrics"]

    for ax, (title, op_id, params) in zip(axes.flat, variants):
        if op_id is None:
            disp_bgr = bgr
            m = orig_m
            runtime_ms = 0.0
        else:
            disp_bgr, _, tel = apply_classical_operation(bgr, op_id, params)
            m = tel["processed_metrics"]
            runtime_ms = tel["processing_time_ms"]

        rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        subtitle = (
            f"μ={m['mean_intensity']:.1f} | σ={m['std_deviation']:.1f} | "
            f"LapVar={m['laplacian_variance']:.0f}"
        )
        if m.get("psnr_db") is not None and op_id is not None:
            subtitle += f"\nPSNR={m['psnr_db']:.1f}dB | SSIM={m['ssim_score']:.3f}"
        ax.set_title(f"{title}\n{subtitle}", color="#F1F5F9", fontsize=9.5, pad=6)
        ax.axis("off")

        csv_rows.append({
            "variant": title,
            "operation": op_id or "original",
            "runtime_ms": runtime_ms,
            **m,
        })

    plt.tight_layout()
    fig_path = out_dir / f"classical_ip_comparison_{in_path.stem}.png"
    plt.savefig(fig_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    csv_path = out_dir / f"classical_ip_metrics_{in_path.stem}.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    return {
        "figure_path": str(fig_path),
        "metrics_csv_path": str(csv_path),
    }
