"""OpenCV Classical Digital Image Processing Pipeline (Preprocessing, Filtering, Enhancement, Edges, Morphology)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ml.preprocessing.metrics import (
    compute_channel_histograms,
    compute_difference_visualization,
    compute_quantitative_metrics,
)


def _odd_kernel(val: Any, minimum: int = 3, maximum: int = 31) -> int:
    k = int(round(float(val)))
    k = max(minimum, min(maximum, k))
    if k % 2 == 0:
        k += 1 if k < maximum else -1
    return k


def _morph_structuring_element(shape_name: str, ksize: int) -> np.ndarray:
    k = max(1, min(31, int(ksize)))
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS,
    }
    st = shape_map.get(str(shape_name).lower(), cv2.MORPH_RECT)
    return cv2.getStructuringElement(st, (k, k))


OPERATION_CATALOG: List[Dict[str, Any]] = [
    # A. Preprocessing
    {
        "id": "resize",
        "name": "Image Resizing",
        "category": "Preprocessing",
        "description": "Spatial resampling using nearest-neighbor, bilinear, bicubic, or Lanczos interpolation.",
        "formula": "I'(x', y') = Interpolation(I, x'/sx, y'/sy)",
        "parameters": [
            {"name": "width", "label": "Target Width (px)", "type": "number", "min": 64, "max": 2048, "step": 32, "default": 640},
            {"name": "height", "label": "Target Height (px)", "type": "number", "min": 64, "max": 2048, "step": 32, "default": 640},
            {"name": "interpolation", "label": "Interpolation", "type": "select", "options": ["bilinear", "bicubic", "nearest", "lanczos"], "default": "bilinear"},
        ],
    },
    {
        "id": "normalize",
        "name": "Intensity Normalization",
        "category": "Preprocessing",
        "description": "Rescales pixel dynamic range via min-max [0, 255] or standardized Z-score mapping.",
        "formula": "I_norm = 255 * (I - I_min) / (I_max - I_min)",
        "parameters": [
            {"name": "mode", "label": "Normalization Mode", "type": "select", "options": ["minmax", "zscore"], "default": "minmax"},
        ],
    },
    {
        "id": "color_convert",
        "name": "Color Space Conversion",
        "category": "Preprocessing",
        "description": "Transforms road imagery between BGR/RGB, HSV, YCrCb luminance-chrominance, and CIE Lab spaces.",
        "formula": "BGR -> {RGB, HSV, YCrCb, CIE L*a*b*}",
        "parameters": [
            {"name": "space", "label": "Target Color Space", "type": "select", "options": ["rgb_swap", "hsv", "ycrcb", "lab"], "default": "ycrcb"},
        ],
    },
    {
        "id": "grayscale",
        "name": "Grayscale Conversion",
        "category": "Preprocessing",
        "description": "Projects 3-channel color road surface imagery onto single-channel luminance intensity.",
        "formula": "Y = 0.299 R + 0.587 G + 0.114 B",
        "parameters": [
            {"name": "method", "label": "Weighting Method", "type": "select", "options": ["bt601", "average"], "default": "bt601"},
        ],
    },
    {
        "id": "denoise_nlmeans",
        "name": "Non-Local Means Denoising",
        "category": "Preprocessing",
        "description": "Suppresses asphalt texture grain while preserving crack boundary discontinuities via patch similarity.",
        "formula": "NL[u](x) = sum_y w(x, y) u(y)",
        "parameters": [
            {"name": "h", "label": "Filter Strength (h)", "type": "number", "min": 1.0, "max": 25.0, "step": 1.0, "default": 8.0},
            {"name": "template_window", "label": "Template Patch Size", "type": "number", "min": 3, "max": 11, "step": 2, "default": 7},
            {"name": "search_window", "label": "Search Window Size", "type": "number", "min": 7, "max": 25, "step": 2, "default": 21},
        ],
    },
    {
        "id": "sharpen_kernel",
        "name": "Laplacian Kernel Sharpening",
        "category": "Preprocessing",
        "description": "Accentuates high-frequency crack edges by adding weighted second-order spatial derivatives.",
        "formula": "I_sharp = I - alpha * nabla^2 I",
        "parameters": [
            {"name": "strength", "label": "Sharpening Gain", "type": "number", "min": 0.2, "max": 3.0, "step": 0.1, "default": 1.0},
        ],
    },
    # B. Filtering
    {
        "id": "gaussian_blur",
        "name": "Gaussian Low-Pass Filter",
        "category": "Filtering",
        "description": "Isotropic linear smoothing using a 2D Gaussian impulse response to attenuate Gaussian sensor noise.",
        "formula": "G(x,y) = (1 / 2*pi*sigma^2) * exp(-(x^2+y^2)/(2*sigma^2))",
        "parameters": [
            {"name": "kernel_size", "label": "Kernel Size (odd)", "type": "number", "min": 3, "max": 31, "step": 2, "default": 5},
            {"name": "sigma", "label": "Sigma", "type": "number", "min": 0.1, "max": 10.0, "step": 0.2, "default": 1.4},
        ],
    },
    {
        "id": "median_blur",
        "name": "Median Non-Linear Filter",
        "category": "Filtering",
        "description": "Order-statistic filter highly effective against impulsive salt-and-pepper noise and coarse aggregate speckle.",
        "formula": "I'(x,y) = median_{(i,j) in W} I(x+i, y+j)",
        "parameters": [
            {"name": "kernel_size", "label": "Aperture Size (odd)", "type": "number", "min": 3, "max": 31, "step": 2, "default": 5},
        ],
    },
    {
        "id": "bilateral_filter",
        "name": "Bilateral Edge-Preserving Filter",
        "category": "Filtering",
        "description": "Combines spatial proximity and radiometric similarity weights to smooth asphalt grain without blurring crack edges.",
        "formula": "BF[I]_p = (1/W_p) sum_q G_sigma_s(||p-q||) G_sigma_r(|I_p-I_q|) I_q",
        "parameters": [
            {"name": "d", "label": "Neighborhood Diameter (d)", "type": "number", "min": 3, "max": 21, "step": 2, "default": 9},
            {"name": "sigma_color", "label": "Sigma Color", "type": "number", "min": 10.0, "max": 200.0, "step": 5.0, "default": 75.0},
            {"name": "sigma_space", "label": "Sigma Space", "type": "number", "min": 10.0, "max": 200.0, "step": 5.0, "default": 75.0},
        ],
    },
    {
        "id": "average_blur",
        "name": "Average (Box) Filter",
        "category": "Filtering",
        "description": "Uniform local neighborhood averaging baseline filter demonstrating frequency roll-off and edge blurring.",
        "formula": "I'(x,y) = (1 / k^2) sum_{i,j} I(x+i, y+j)",
        "parameters": [
            {"name": "kernel_size", "label": "Kernel Size", "type": "number", "min": 3, "max": 31, "step": 2, "default": 5},
        ],
    },
    # C. Image Enhancement
    {
        "id": "histogram_equalization",
        "name": "Global Histogram Equalization",
        "category": "Enhancement",
        "description": "Redistributes intensity probabilities via the cumulative distribution function (CDF) to maximize global contrast.",
        "formula": "s_k = (L - 1) * sum_{j=0}^k p_r(r_j)",
        "parameters": [
            {"name": "preserve_color", "label": "Equalize Luminance Only (YCrCb)", "type": "boolean", "default": True},
        ],
    },
    {
        "id": "clahe",
        "name": "CLAHE (Adaptive Histogram Equalization)",
        "category": "Enhancement",
        "description": "Contrast-Limited Adaptive Histogram Equalization enhances shadowed pavement cracks while bounding noise amplification.",
        "formula": "Clipped Tile CDF Interpolation (clipLimit, grid)",
        "parameters": [
            {"name": "clip_limit", "label": "Clip Limit", "type": "number", "min": 0.5, "max": 10.0, "step": 0.5, "default": 2.5},
            {"name": "tile_grid_size", "label": "Tile Grid Size", "type": "number", "min": 2, "max": 16, "step": 2, "default": 8},
            {"name": "preserve_color", "label": "Preserve Chrominance (LAB L-channel)", "type": "boolean", "default": True},
        ],
    },
    {
        "id": "gamma_correction",
        "name": "Power-Law Gamma Correction",
        "category": "Enhancement",
        "description": "Non-linear radiometric mapping (<1 brightens dark shadows/pothole interiors; >1 compresses bright glare).",
        "formula": "I_out = 255 * (I_in / 255)^gamma",
        "parameters": [
            {"name": "gamma", "label": "Gamma Exponent", "type": "number", "min": 0.2, "max": 3.0, "step": 0.1, "default": 0.7},
        ],
    },
    {
        "id": "contrast_stretching",
        "name": "Percentile Contrast Stretching",
        "category": "Enhancement",
        "description": "Piecewise linear intensity stretching anchored at configurable lower and upper histogram percentiles.",
        "formula": "I_out = clip((I - P_low) / (P_high - P_low), 0, 1) * 255",
        "parameters": [
            {"name": "low_percentile", "label": "Lower Percentile (%)", "type": "number", "min": 0.0, "max": 25.0, "step": 1.0, "default": 2.0},
            {"name": "high_percentile", "label": "Upper Percentile (%)", "type": "number", "min": 75.0, "max": 100.0, "step": 1.0, "default": 98.0},
        ],
    },
    {
        "id": "unsharp_masking",
        "name": "Unsharp Masking",
        "category": "Enhancement",
        "description": "Subtracts a Gaussian-smoothed low-pass component to boost fine hairline crack contrast.",
        "formula": "I_usm = I + amount * (I - G_sigma * I)",
        "parameters": [
            {"name": "kernel_size", "label": "Gaussian Kernel (odd)", "type": "number", "min": 3, "max": 21, "step": 2, "default": 5},
            {"name": "sigma", "label": "Gaussian Sigma", "type": "number", "min": 0.5, "max": 5.0, "step": 0.25, "default": 1.5},
            {"name": "amount", "label": "Mask Gain (Amount)", "type": "number", "min": 0.2, "max": 3.0, "step": 0.1, "default": 1.4},
            {"name": "threshold", "label": "Noise Gate Threshold", "type": "number", "min": 0, "max": 50, "step": 1, "default": 3},
        ],
    },
    # D. Edge Detection
    {
        "id": "sobel_edge",
        "name": "Sobel Gradient Operator",
        "category": "Edge Detection",
        "description": "First-order directional spatial derivative operator highlighting horizontal (D10) and vertical (D00) crack gradients.",
        "formula": "|G| = sqrt((S_x * I)^2 + (S_y * I)^2)",
        "parameters": [
            {"name": "ksize", "label": "Sobel Kernel Size", "type": "select", "options": ["3", "5", "7"], "default": "3"},
            {"name": "mode", "label": "Gradient Component", "type": "select", "options": ["magnitude", "dx", "dy"], "default": "magnitude"},
        ],
    },
    {
        "id": "canny_edge",
        "name": "Canny Multi-Stage Edge Detector",
        "category": "Edge Detection",
        "description": "Gaussian smoothing, gradient magnitude, non-maximum suppression, and dual-threshold hysteresis tracking.",
        "formula": "Hysteresis(NMS(|nabla(G * I)|), T_low, T_high)",
        "parameters": [
            {"name": "threshold1", "label": "Lower Hysteresis Threshold", "type": "number", "min": 10, "max": 300, "step": 5, "default": 75},
            {"name": "threshold2", "label": "Upper Hysteresis Threshold", "type": "number", "min": 20, "max": 500, "step": 5, "default": 175},
            {"name": "aperture_size", "label": "Sobel Aperture", "type": "select", "options": ["3", "5", "7"], "default": "3"},
        ],
    },
    {
        "id": "laplacian_edge",
        "name": "Laplacian / LoG Zero-Crossing Operator",
        "category": "Edge Detection",
        "description": "Isotropic second-order derivative operator with optional Gaussian pre-smoothing (Laplacian of Gaussian).",
        "formula": "nabla^2 I = d^2I/dx^2 + d^2I/dy^2",
        "parameters": [
            {"name": "ksize", "label": "Laplacian Kernel", "type": "select", "options": ["1", "3", "5"], "default": "3"},
            {"name": "pre_blur", "label": "Pre-Gaussian Blur (odd)", "type": "number", "min": 1, "max": 11, "step": 2, "default": 3},
        ],
    },
    # E. Morphological Operations
    {
        "id": "morph_erosion",
        "name": "Morphological Erosion",
        "category": "Morphology",
        "description": "Shrinks bright foreground regions and deepens dark crack valleys using a structuring element.",
        "formula": "(I (-) B)(x,y) = min_{(i,j) in B} I(x+i, y+j)",
        "parameters": [
            {"name": "kernel_size", "label": "Structuring Element Size", "type": "number", "min": 3, "max": 25, "step": 2, "default": 5},
            {"name": "kernel_shape", "label": "Element Shape", "type": "select", "options": ["rect", "ellipse", "cross"], "default": "ellipse"},
            {"name": "iterations", "label": "Iterations", "type": "number", "min": 1, "max": 8, "step": 1, "default": 1},
        ],
    },
    {
        "id": "morph_dilation",
        "name": "Morphological Dilation",
        "category": "Morphology",
        "description": "Expands bright structures or bridges broken binary edge segments across fragmented cracks.",
        "formula": "(I (+) B)(x,y) = max_{(i,j) in B} I(x-i, y-j)",
        "parameters": [
            {"name": "kernel_size", "label": "Structuring Element Size", "type": "number", "min": 3, "max": 25, "step": 2, "default": 5},
            {"name": "kernel_shape", "label": "Element Shape", "type": "select", "options": ["rect", "ellipse", "cross"], "default": "ellipse"},
            {"name": "iterations", "label": "Iterations", "type": "number", "min": 1, "max": 8, "step": 1, "default": 1},
        ],
    },
    {
        "id": "morph_opening",
        "name": "Morphological Opening",
        "category": "Morphology",
        "description": "Erosion followed by dilation; removes small bright road aggregate specks while preserving structure.",
        "formula": "I o B = (I (-) B) (+) B",
        "parameters": [
            {"name": "kernel_size", "label": "Structuring Element Size", "type": "number", "min": 3, "max": 25, "step": 2, "default": 5},
            {"name": "kernel_shape", "label": "Element Shape", "type": "select", "options": ["rect", "ellipse", "cross"], "default": "ellipse"},
            {"name": "iterations", "label": "Iterations", "type": "number", "min": 1, "max": 8, "step": 1, "default": 1},
        ],
    },
    {
        "id": "morph_closing",
        "name": "Morphological Closing",
        "category": "Morphology",
        "description": "Dilation followed by erosion; bridges narrow gaps in segmented crack networks and fills dark micro-voids.",
        "formula": "I . B = (I (+) B) (-) B",
        "parameters": [
            {"name": "kernel_size", "label": "Structuring Element Size", "type": "number", "min": 3, "max": 25, "step": 2, "default": 5},
            {"name": "kernel_shape", "label": "Element Shape", "type": "select", "options": ["rect", "ellipse", "cross"], "default": "ellipse"},
            {"name": "iterations", "label": "Iterations", "type": "number", "min": 1, "max": 8, "step": 1, "default": 1},
        ],
    },
    {
        "id": "morph_gradient",
        "name": "Morphological Gradient",
        "category": "Morphology",
        "description": "Difference between dilation and erosion, isolating pothole rims and crack perimeters.",
        "formula": "Grad(I) = (I (+) B) - (I (-) B)",
        "parameters": [
            {"name": "kernel_size", "label": "Structuring Element Size", "type": "number", "min": 3, "max": 25, "step": 2, "default": 5},
            {"name": "kernel_shape", "label": "Element Shape", "type": "select", "options": ["rect", "ellipse", "cross"], "default": "ellipse"},
            {"name": "iterations", "label": "Iterations", "type": "number", "min": 1, "max": 5, "step": 1, "default": 1},
        ],
    },
]


def get_operation_catalog() -> List[Dict[str, Any]]:
    return OPERATION_CATALOG


def execute_raw_operation(
    img_bgr: np.ndarray,
    operation: str,
    params: Dict[str, Any] | None = None,
) -> np.ndarray:
    """Execute a single OpenCV image-processing operation and return an 8-bit BGR or Grayscale ndarray."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Input image is empty")

    p = params or {}
    op = operation.strip().lower()

    if img_bgr.ndim == 2:
        bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    else:
        bgr = img_bgr.copy()

    # A. Preprocessing
    if op == "resize":
        w = max(32, min(4096, int(p.get("width", 640))))
        h = max(32, min(4096, int(p.get("height", 640))))
        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        interp = interp_map.get(str(p.get("interpolation", "bilinear")).lower(), cv2.INTER_LINEAR)
        return cv2.resize(bgr, (w, h), interpolation=interp)

    if op == "normalize":
        mode = str(p.get("mode", "minmax")).lower()
        if mode == "zscore":
            f = bgr.astype(np.float32)
            mean = float(np.mean(f))
            std = max(float(np.std(f)), 1e-6)
            z = (f - mean) / std
            # Map [-2.5, +2.5] sigma to [0, 255]
            mapped = np.clip((z + 2.5) / 5.0 * 255.0, 0, 255).astype(np.uint8)
            return mapped
        out = np.empty_like(bgr)
        cv2.normalize(bgr, out, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return out

    if op == "color_convert":
        space = str(p.get("space", "ycrcb")).lower()
        if space == "rgb_swap":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if space == "hsv":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        if space == "lab":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    if op == "grayscale":
        method = str(p.get("method", "bt601")).lower()
        if method == "average":
            gray = np.mean(bgr.astype(np.float32), axis=2).astype(np.uint8)
        else:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if op == "denoise_nlmeans":
        h_strength = max(1.0, min(30.0, float(p.get("h", 8.0))))
        tw = _odd_kernel(p.get("template_window", 7), 3, 15)
        sw = _odd_kernel(p.get("search_window", 21), 5, 31)
        return cv2.fastNlMeansDenoisingColored(bgr, None, h_strength, h_strength, tw, sw)

    if op == "sharpen_kernel":
        s = max(0.0, min(5.0, float(p.get("strength", 1.0))))
        kernel = np.array(
            [[0.0, -s, 0.0], [-s, 1.0 + 4.0 * s, -s], [0.0, -s, 0.0]],
            dtype=np.float32,
        )
        return cv2.filter2D(bgr, -1, kernel)

    # B. Filtering
    if op == "gaussian_blur":
        k = _odd_kernel(p.get("kernel_size", 5), 3, 31)
        sigma = max(0.0, min(20.0, float(p.get("sigma", 1.4))))
        return cv2.GaussianBlur(bgr, (k, k), sigmaX=sigma, sigmaY=sigma)

    if op == "median_blur":
        k = _odd_kernel(p.get("kernel_size", 5), 3, 31)
        return cv2.medianBlur(bgr, k)

    if op == "bilateral_filter":
        d = max(3, min(25, int(p.get("d", 9))))
        sc = max(1.0, min(250.0, float(p.get("sigma_color", 75.0))))
        ss = max(1.0, min(250.0, float(p.get("sigma_space", 75.0))))
        return cv2.bilateralFilter(bgr, d=d, sigmaColor=sc, sigmaSpace=ss)

    if op == "average_blur":
        k = max(2, min(31, int(p.get("kernel_size", 5))))
        return cv2.blur(bgr, (k, k))

    # C. Enhancement
    if op == "histogram_equalization":
        preserve = bool(p.get("preserve_color", True))
        if preserve:
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    if op == "clahe":
        clip = max(0.1, min(20.0, float(p.get("clip_limit", 2.5))))
        grid = max(2, min(32, int(p.get("tile_grid_size", 8))))
        preserve = bool(p.get("preserve_color", True))
        clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        if preserve:
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(clahe_obj.apply(gray), cv2.COLOR_GRAY2BGR)

    if op == "gamma_correction":
        gamma = max(0.05, min(5.0, float(p.get("gamma", 0.7))))
        lut = np.array([np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
        return cv2.LUT(bgr, lut)

    if op == "contrast_stretching":
        low_p = max(0.0, min(49.0, float(p.get("low_percentile", 2.0))))
        high_p = max(low_p + 1.0, min(100.0, float(p.get("high_percentile", 98.0))))
        out = np.empty_like(bgr)
        for c in range(3):
            ch = bgr[:, :, c].astype(np.float32)
            lo = float(np.percentile(ch, low_p))
            hi = float(np.percentile(ch, high_p))
            denom = max(hi - lo, 1.0)
            out[:, :, c] = np.clip((ch - lo) / denom * 255.0, 0, 255).astype(np.uint8)
        return out

    if op == "unsharp_masking":
        k = _odd_kernel(p.get("kernel_size", 5), 3, 25)
        sigma = max(0.1, min(10.0, float(p.get("sigma", 1.5))))
        amount = max(0.0, min(5.0, float(p.get("amount", 1.4))))
        threshold = max(0, min(100, int(p.get("threshold", 3))))
        blurred = cv2.GaussianBlur(bgr, (k, k), sigmaX=sigma)
        sharpened = cv2.addWeighted(bgr, 1.0 + amount, blurred, -amount, 0)
        if threshold > 0:
            low_contrast_mask = np.abs(bgr.astype(np.int16) - blurred.astype(np.int16)) < threshold
            np.copyto(sharpened, bgr, where=low_contrast_mask)
        return sharpened

    # D. Edge Detection
    if op == "sobel_edge":
        ksize = int(p.get("ksize", 3))
        if ksize not in {1, 3, 5, 7}:
            ksize = 3
        mode = str(p.get("mode", "magnitude")).lower()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        if mode == "dx":
            edge = cv2.convertScaleAbs(gx)
        elif mode == "dy":
            edge = cv2.convertScaleAbs(gy)
        else:
            mag = cv2.magnitude(gx, gy)
            max_m = max(float(np.max(mag)), 1.0)
            edge = np.clip(mag / max_m * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    if op == "canny_edge":
        t1 = max(1.0, min(500.0, float(p.get("threshold1", 75))))
        t2 = max(1.0, min(600.0, float(p.get("threshold2", 175))))
        ap = int(p.get("aperture_size", 3))
        if ap not in {3, 5, 7}:
            ap = 3
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=t1, threshold2=t2, apertureSize=ap, L2gradient=True)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if op == "laplacian_edge":
        ksize = int(p.get("ksize", 3))
        if ksize not in {1, 3, 5, 7}:
            ksize = 3
        pre_blur = _odd_kernel(p.get("pre_blur", 3), 1, 15)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if pre_blur > 1:
            gray = cv2.GaussianBlur(gray, (pre_blur, pre_blur), 0)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        edge = cv2.convertScaleAbs(lap)
        return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    # E. Morphological Operations
    if op in {"morph_erosion", "morph_dilation", "morph_opening", "morph_closing", "morph_gradient"}:
        k = _odd_kernel(p.get("kernel_size", 5), 3, 29)
        shape_name = str(p.get("kernel_shape", "ellipse"))
        iters = max(1, min(12, int(p.get("iterations", 1))))
        se = _morph_structuring_element(shape_name, k)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        if op == "morph_erosion":
            res = cv2.erode(gray, se, iterations=iters)
        elif op == "morph_dilation":
            res = cv2.dilate(gray, se, iterations=iters)
        elif op == "morph_opening":
            res = cv2.morphologyEx(gray, cv2.MORPH_OPEN, se, iterations=iters)
        elif op == "morph_closing":
            res = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se, iterations=iters)
        else:
            res = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, se, iterations=iters)
        return cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported classical image processing operation: '{operation}'")


def apply_classical_operation(
    img_bgr: np.ndarray,
    operation: str,
    params: Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Execute operation and return:
      (processed_bgr, difference_heatmap_bgr, full_telemetry_dict)
    """
    t0 = time.perf_counter()
    processed = execute_raw_operation(img_bgr, operation=operation, params=params)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    orig_metrics = compute_quantitative_metrics(img_bgr, reference_image=None)
    # Provide PSNR & SSIM against original when spatial dimensions match
    ref = img_bgr if img_bgr.shape[:2] == processed.shape[:2] else None
    proc_metrics = compute_quantitative_metrics(processed, reference_image=ref)
    diff_heatmap = compute_difference_visualization(img_bgr, processed)

    telemetry = {
        "operation": operation,
        "parameters": params or {},
        "processing_time_ms": elapsed_ms,
        "original_metrics": orig_metrics.to_dict(),
        "processed_metrics": proc_metrics.to_dict(),
        "original_histogram": compute_channel_histograms(img_bgr),
        "processed_histogram": compute_channel_histograms(processed),
    }
    return processed, diff_heatmap, telemetry


def compare_filtering_techniques(
    img_bgr: np.ndarray,
    kernel_size: int = 5,
) -> List[Dict[str, Any]]:
    """Run all 4 canonical spatial filters side-by-side and return images + quantitative metrics."""
    k = _odd_kernel(kernel_size, 3, 25)
    configs = [
        ("gaussian_blur", "Gaussian Filter", {"kernel_size": k, "sigma": 1.4}),
        ("median_blur", "Median Filter", {"kernel_size": k}),
        ("bilateral_filter", "Bilateral Filter", {"d": max(5, k), "sigma_color": 75.0, "sigma_space": 75.0}),
        ("average_blur", "Average (Box) Filter", {"kernel_size": k}),
    ]
    results: List[Dict[str, Any]] = []
    for op_id, label, p in configs:
        proc, diff, tel = apply_classical_operation(img_bgr, op_id, p)
        results.append({
            "operation": op_id,
            "label": label,
            "processed_bgr": proc,
            "diff_bgr": diff,
            "processing_time_ms": tel["processing_time_ms"],
            "metrics": tel["processed_metrics"],
        })
    return results
