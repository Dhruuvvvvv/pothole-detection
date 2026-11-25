"""Quantitative image quality and structural metrics for Classical Image Processing Lab."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class ImageQuantitativeMetrics:
    width: int
    height: int
    channels: int
    mean_intensity: float
    std_deviation: float
    rms_contrast: float
    michelson_contrast: float
    shannon_entropy: float
    laplacian_variance: float
    edge_pixel_density: float
    psnr_db: Optional[float] = None
    ssim_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_gray_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        gray = img
    elif img.shape[2] == 1:
        gray = img[:, :, 0]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _ensure_bgr_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        bgr = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        bgr = img
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    return bgr


def compute_quantitative_metrics(
    image: np.ndarray,
    reference_image: Optional[np.ndarray] = None,
) -> ImageQuantitativeMetrics:
    """
    Calculate quantitative digital image processing metrics:
      - Mean intensity & standard deviation
      - RMS contrast & Michelson contrast
      - Shannon entropy (bits)
      - Laplacian variance (focus / high-frequency energy)
      - Edge pixel density (Canny edge fraction)
      - PSNR (dB) and SSIM when a valid same-dimension reference image is provided
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot compute metrics on an empty image array")

    h, w = image.shape[:2]
    ch = 1 if image.ndim == 2 else int(image.shape[2])
    gray = _to_gray_uint8(image)
    gray_f = gray.astype(np.float64)

    mean_val = float(np.mean(gray_f))
    std_val = float(np.std(gray_f))
    rms_contrast = std_val

    i_max = float(np.max(gray_f))
    i_min = float(np.min(gray_f))
    denom = i_max + i_min
    michelson = float((i_max - i_min) / denom) if denom > 0 else 0.0

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    probs = hist / float(gray.size)
    nonzero_probs = probs[probs > 0]
    entropy = float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    psnr_val: Optional[float] = None
    ssim_val: Optional[float] = None

    if reference_image is not None and reference_image.shape[:2] == image.shape[:2]:
        ref_gray = _to_gray_uint8(reference_image)
        mse = float(np.mean((ref_gray.astype(np.float64) - gray_f) ** 2))
        if mse == 0.0:
            psnr_val = 100.0
            ssim_val = 1.0
        else:
            raw_psnr = float(peak_signal_noise_ratio(ref_gray, gray, data_range=255))
            psnr_val = round(raw_psnr, 3) if not math.isinf(raw_psnr) else 100.0
            win_size = min(7, h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1)
            if win_size >= 3:
                raw_ssim = float(structural_similarity(ref_gray, gray, data_range=255, win_size=win_size))
                ssim_val = round(raw_ssim, 4)

    return ImageQuantitativeMetrics(
        width=w,
        height=h,
        channels=ch,
        mean_intensity=round(mean_val, 3),
        std_deviation=round(std_val, 3),
        rms_contrast=round(rms_contrast, 3),
        michelson_contrast=round(michelson, 4),
        shannon_entropy=round(entropy, 4),
        laplacian_variance=round(lap_var, 3),
        edge_pixel_density=round(edge_density, 5),
        psnr_db=psnr_val,
        ssim_score=ssim_val,
    )


def compute_difference_visualization(
    original_img: np.ndarray,
    processed_img: np.ndarray,
) -> np.ndarray:
    """
    Generate a false-color absolute difference map between `original_img` and `processed_img`.
    Resizes `original_img` to `processed_img` resolution ONLY if a resize operation changed dimensions.
    """
    orig_bgr = _ensure_bgr_uint8(original_img)
    proc_bgr = _ensure_bgr_uint8(processed_img)

    if orig_bgr.shape[:2] != proc_bgr.shape[:2]:
        orig_bgr = cv2.resize(orig_bgr, (proc_bgr.shape[1], proc_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    diff_gray = cv2.cvtColor(cv2.absdiff(orig_bgr, proc_bgr), cv2.COLOR_BGR2GRAY)
    # Amplify subtle intensity deltas cleanly while preserving zero-difference black regions
    amplified = np.clip(diff_gray.astype(np.float32) * 2.5, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(amplified, cv2.COLORMAP_TURBO)
    return heatmap


def compute_channel_histograms(img: np.ndarray, bins: int = 64) -> List[Dict[str, float]]:
    """Compute binned intensity histograms for Luminance and RGB channels for frontend visualization."""
    bgr = _ensure_bgr_uint8(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    total = float(gray.size)
    hist_l = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel() / total
    hist_b = cv2.calcHist([bgr], [0], None, [bins], [0, 256]).ravel() / total
    hist_g = cv2.calcHist([bgr], [1], None, [bins], [0, 256]).ravel() / total
    hist_r = cv2.calcHist([bgr], [2], None, [bins], [0, 256]).ravel() / total

    step = 256.0 / bins
    rows: List[Dict[str, float]] = []
    for i in range(bins):
        rows.append({
            "intensity": round((i + 0.5) * step, 1),
            "luminance": round(float(hist_l[i]) * 100.0, 3),
            "red": round(float(hist_r[i]) * 100.0, 3),
            "green": round(float(hist_g[i]) * 100.0, 3),
            "blue": round(float(hist_b[i]) * 100.0, 3),
        })
    return rows
