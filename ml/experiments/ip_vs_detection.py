"""
Explainability & Controlled Experimentation Suite (Module 5).

Compares:
  1. Classical IP variants (Original, Gaussian, Median, CLAHE, Gamma, Sharpened) prior to YOLO inference
  2. Confidence threshold & resolution sweeps
  3. Common error sources in road surface imagery (shadows, lane markings, road repairs, etc.)
  4. EigenCAM spatial feature activation visualization for YOLO neck/backbone tensors
     (explicitly documented with its class-agnostic SVD limitation; never mislabelled as SHAP or class-specific Grad-CAM).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

from ml.evaluation.evaluator import evaluate_trained_model
from ml.inference.detector import RoadDamageDetector
from ml.preprocessing.classical_ops import execute_raw_operation


COMMON_ERROR_SOURCES: List[Dict[str, str]] = [
    {
        "id": "shadows",
        "name": "Cast Shadows (Trees, Bridges, Vehicles)",
        "error_mode": "False Positives (D10/D00) & False Negatives inside deep shade",
        "mechanism": "Sharp shadow boundaries produce high first-order gradient magnitudes resembling linear cracks, while low dynamic range inside shadows obscures fine alligator cracking.",
        "mitigation": "Evaluate CLAHE (clipLimit=2.0-2.5) and brightness/contrast training augmentation; verify empirically on validation mAP before deployment.",
    },
    {
        "id": "lane_markings",
        "name": "Worn Lane Markings & Crosswalks",
        "error_mode": "False Positives (D00 Longitudinal / D10 Transverse)",
        "mechanism": "Peeling thermoplastic paint edges create elongated high-contrast linear boundaries parallel or perpendicular to travel direction.",
        "mitigation": "Include intact and weathered road markings (D43/D44 negative examples) as background samples during YOLO training.",
    },
    {
        "id": "road_repairs",
        "name": "Bitumen Crack Sealing & Asphalt Patching",
        "error_mode": "False Positives (D00/D10/D20)",
        "mechanism": "Dark tar overbanding strips and rectangular cold-patch seams closely mimic active open cracks visually in 2D RGB imagery.",
        "mitigation": "Exclude 'Repair' labels during target mapping so sealed cracks are treated as non-damage background, though visual ambiguity remains without 3D profilometry.",
    },
    {
        "id": "low_lighting",
        "name": "Dusk, Tunnel & Overcast Low Illumination",
        "error_mode": "False Negatives across hairline cracks (D00/D10/D20)",
        "mechanism": "Sensor read noise and compressed shadow histograms reduce crack-to-asphalt Michelson contrast below convolutional filter activation thresholds.",
        "mitigation": "Compare Gamma (gamma=0.7) vs CLAHE preprocessing quantitatively on low-luminance validation slices.",
    },
    {
        "id": "motion_blur",
        "name": "High-Speed Vehicle Motion Blur",
        "error_mode": "False Negatives on Transverse (D10) & Alligator (D20) cracks",
        "mechanism": "Forward camera motion convolves road texture with a directional PSF, attenuating high spatial frequencies perpendicular to motion.",
        "mitigation": "Higher shutter speeds during capture; measure Laplacian variance (< 60 indicates blur) to flag degraded video frames.",
    },
    {
        "id": "small_defects",
        "name": "Far-Field & Incipient Horizon Defects",
        "error_mode": "False Negatives (Low Recall at IoU 0.50:0.95)",
        "mechanism": "Perspective foreshortening shrinks distant potholes/cracks to < 12x12 px, losing spatial resolution after stride-8/16 downsampling.",
        "mitigation": "Compare imgsz=640 vs imgsz=960 inference resolution trade-offs (mAP@50 vs FPS).",
    },
    {
        "id": "occlusion",
        "name": "Traffic, Standing Water & Debris Occlusion",
        "error_mode": "Fragmented Detections & ID Switches in Video",
        "mechanism": "Lead vehicles, puddles reflecting sky inside potholes, or fallen leaves partially hide defect geometry across consecutive frames.",
        "mitigation": "Use ByteTrack/SORT temporal association with coasting age buffer while avoiding claims that track IDs equal unique physical defects.",
    },
]


def run_confidence_threshold_sweep(
    detector: RoadDamageDetector,
    img_bgr: np.ndarray,
    thresholds: Optional[List[float]] = None,
    iou_threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    """Evaluate how detection count, mean confidence, and priority distribution vary across thresholds."""
    sweep_vals = thresholds or [0.15, 0.25, 0.35, 0.50, 0.65]
    results: List[Dict[str, Any]] = []
    for conf in sweep_vals:
        _, dets, summary = detector.predict_image(
            img_bgr=img_bgr,
            conf_threshold=conf,
            iou_threshold=iou_threshold,
        )
        results.append({
            "conf_threshold": conf,
            "total_detections": len(dets),
            "high_priority": summary["priority_bands"]["High"],
            "medium_priority": summary["priority_bands"]["Medium"],
            "low_priority": summary["priority_bands"]["Low"],
            "mean_confidence": round(
                sum(d.confidence for d in dets) / len(dets), 4
            ) if dets else 0.0,
            "inference_time_ms": summary["inference_time_ms"],
        })
    return results


def run_preprocessing_detection_experiment(
    checkpoint_path: Path | str,
    dataset_yaml: Path | str,
    work_dir: Path | str = "./outputs/experiments/ip_vs_detection",
    split: str = "val",
    image_size: int = 640,
) -> Dict[str, Any]:
    """
    Empirically evaluate whether classical image enhancement improves or degrades YOLO accuracy
    on a labelled split (Original vs Gaussian vs Median vs CLAHE vs Gamma vs Sharpened).
    Never claims improvement unless verified on ground-truth labels.
    """
    yaml_p = Path(dataset_yaml).resolve()
    cfg = yaml.safe_load(yaml_p.read_text(encoding="utf-8"))
    ds_root = Path(cfg.get("path", yaml_p.parent)).resolve()
    rel_img_dir = cfg.get(split, f"images/{split}")
    src_img_dir = (ds_root / rel_img_dir).resolve()
    src_lbl_dir = (ds_root / str(rel_img_dir).replace("images", "labels")).resolve()

    if not src_img_dir.is_dir() or not src_lbl_dir.is_dir():
        raise FileNotFoundError(f"Labelled split directories not found: {src_img_dir} / {src_lbl_dir}")

    variants = [
        ("original", None, {}),
        ("gaussian_k5", "gaussian_blur", {"kernel_size": 5, "sigma": 1.2}),
        ("median_k5", "median_blur", {"kernel_size": 5}),
        ("clahe_clip2.5", "clahe", {"clip_limit": 2.5, "tile_grid_size": 8, "preserve_color": True}),
        ("gamma_0.75", "gamma_correction", {"gamma": 0.75}),
        ("unsharp_mask", "unsharp_masking", {"kernel_size": 5, "sigma": 1.4, "amount": 1.2, "threshold": 3}),
    ]

    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    comparison_rows: List[Dict[str, Any]] = []

    image_files = sorted(p for p in src_img_dir.iterdir() if p.is_file())

    for var_name, op_id, op_params in variants:
        if op_id is None:
            target_yaml = yaml_p
        else:
            var_ds = work / f"temp_{var_name}"
            var_img_dir = var_ds / "images" / split
            var_lbl_dir = var_ds / "labels" / split
            var_img_dir.mkdir(parents=True, exist_ok=True)
            var_lbl_dir.mkdir(parents=True, exist_ok=True)

            for img_p in image_files:
                bgr = cv2.imread(str(img_p))
                if bgr is None:
                    continue
                proc = execute_raw_operation(bgr, op_id, op_params)
                cv2.imwrite(str(var_img_dir / img_p.name), proc)
                lbl_src = src_lbl_dir / f"{img_p.stem}.txt"
                if lbl_src.is_file():
                    shutil.copy2(lbl_src, var_lbl_dir / lbl_src.name)

            var_cfg = {
                "path": str(var_ds).replace("\\", "/"),
                "train": f"images/{split}",
                "val": f"images/{split}",
                "test": f"images/{split}",
                "nc": cfg["nc"],
                "names": cfg["names"],
            }
            target_yaml = var_ds / "data.yaml"
            target_yaml.write_text(yaml.safe_dump(var_cfg, sort_keys=False), encoding="utf-8")

        eval_res = evaluate_trained_model(
            checkpoint_path=checkpoint_path,
            dataset_yaml=target_yaml,
            split=split,
            image_size=image_size,
            output_dir=work / f"eval_{var_name}",
        )
        comparison_rows.append({
            "variant": var_name,
            "operation": op_id or "none",
            "precision": eval_res["precision"],
            "recall": eval_res["recall"],
            "f1_score": eval_res["f1_score"],
            "map50": eval_res["map50"],
            "map50_95": eval_res["map50_95"],
        })

    baseline_map50 = comparison_rows[0]["map50"] if comparison_rows else 0.0
    for row in comparison_rows:
        row["delta_map50_vs_original"] = round(row["map50"] - baseline_map50, 4)

    out_report = {
        "evaluated_split": split,
        "image_count": len(image_files),
        "results": comparison_rows,
        "scientific_note": (
            "Detectors trained exclusively on raw RGB images frequently experience slight domain shift "
            "when aggressive test-time filtering or histogram distortion is applied without retraining."
        ),
    }
    (work / "preprocessing_vs_detection_results.json").write_text(
        json.dumps(out_report, indent=2), encoding="utf-8"
    )
    return out_report


def compute_yolo_eigencam(
    detector: RoadDamageDetector,
    img_bgr: np.ndarray,
    target_layer_index: int = -2,
    image_size: int = 640,
) -> Tuple[np.ndarray, str]:
    """
    Compute a technically valid **EigenCAM** spatial activation overlay from an internal
    PyTorch feature layer of the Ultralytics YOLO architecture (Muhammad & Yeasin, 2020).

    Method & Limitations:
      - Hooks into a deep convolutional neck block prior to the detection head, extracts the
        [C, H_f, W_f] activation tensor, centers across spatial locations, and projects onto
        the first principal component via SVD.
      - Limitation: EigenCAM visualizes dominant spatial feature energy across all channels
        (class-agnostic saliency); unlike classification Grad-CAM, it does not isolate
        per-bounding-box class gradients through Non-Maximum Suppression.
    """
    if not detector.is_ready or detector._model is None:
        raise RuntimeError("Trained YOLO model must be loaded to compute EigenCAM activations.")

    pytorch_model = detector._model.model
    layers = list(pytorch_model.model.children())
    chosen_layer = layers[target_layer_index]

    captured: List[torch.Tensor] = []

    def _forward_hook(_module: Any, _inp: Any, out: Any) -> None:
        tensor = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
            captured.append(tensor.detach().float().cpu())

    handle = chosen_layer.register_forward_hook(_forward_hook)
    try:
        detector._model.predict(source=img_bgr, imgsz=image_size, verbose=False)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("Selected YOLO layer did not emit a 4D spatial feature map.")

    fmap = captured[-1][0]  # [C, Hf, Wf]
    c, hf, wf = fmap.shape
    reshaped = fmap.reshape(c, hf * wf)
    centered = reshaped - reshaped.mean(dim=1, keepdim=True)

    # First principal component projection across spatial locations
    u, _, _ = torch.linalg.svd(centered, full_matrices=False)
    cam = (u[:, 0:1].T @ centered).reshape(hf, wf).numpy()
    cam = np.maximum(cam, 0.0)
    if cam.max() > 1e-8:
        cam = cam / cam.max()

    h, w = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(np.clip(cam_resized * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.58, heatmap, 0.42, 0)

    limitation_note = (
        "EigenCAM (1st SVD principal projection of neck feature activations). "
        "Highlights dominant spatial feature responses; class-agnostic and lower spatial resolution than input pixels."
    )
    return overlay, limitation_note
