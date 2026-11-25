"""Held-out test set model evaluation, metric calculation, and chart generation."""

from __future__ import annotations

import csv
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from ml.constants import ID_TO_SPEC, TARGET_CODES


def _compute_f1(precision: float, recall: float) -> float:
    denom = precision + recall
    if denom <= 1e-9:
        return 0.0
    return round(2.0 * precision * recall / denom, 4)


def plot_training_history_curves(
    results_csv_path: Path | str,
    output_dir: Path | str,
) -> Optional[str]:
    """Generate training vs validation loss and mAP progression curves from Ultralytics results.csv."""
    csv_p = Path(results_csv_path)
    if not csv_p.is_file():
        return None

    epochs: List[int] = []
    train_box, val_box = [], []
    train_cls, val_cls = [], []
    map50, map50_95 = [], []

    with open(csv_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {k.strip(): v.strip() for k, v in row.items() if k and v}
            try:
                epochs.append(int(float(clean.get("epoch", len(epochs) + 1))))
                train_box.append(float(clean.get("train/box_loss", 0.0)))
                val_box.append(float(clean.get("val/box_loss", 0.0)))
                train_cls.append(float(clean.get("train/cls_loss", 0.0)))
                val_cls.append(float(clean.get("val/cls_loss", 0.0)))
                map50.append(float(clean.get("metrics/mAP50(B)", 0.0)))
                map50_95.append(float(clean.get("metrics/mAP50-95(B)", 0.0)))
            except ValueError:
                continue

    if not epochs:
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=140)
    fig.patch.set_facecolor("#0B101E")

    for ax in axes:
        ax.set_facecolor("#10172A")
        ax.tick_params(colors="#CBD5E1")
        for spine in ax.spines.values():
            spine.set_color("#1E293B")

    axes[0].plot(epochs, train_box, label="Train Box Loss", color="#3B82F6", linewidth=2)
    axes[0].plot(epochs, val_box, label="Val Box Loss", color="#F59E0B", linewidth=2, linestyle="--")
    axes[0].set_title("Bounding-Box Regression Loss", color="#F1F5F9")
    axes[0].set_xlabel("Epoch", color="#94A3B8")
    axes[0].legend(facecolor="#151E33", edgecolor="#1E293B", labelcolor="#F1F5F9")

    axes[1].plot(epochs, train_cls, label="Train Cls Loss", color="#06B6D4", linewidth=2)
    axes[1].plot(epochs, val_cls, label="Val Cls Loss", color="#EF4444", linewidth=2, linestyle="--")
    axes[1].set_title("Classification Loss", color="#F1F5F9")
    axes[1].set_xlabel("Epoch", color="#94A3B8")
    axes[1].legend(facecolor="#151E33", edgecolor="#1E293B", labelcolor="#F1F5F9")

    axes[2].plot(epochs, map50, label="Val mAP@50", color="#10B981", linewidth=2)
    axes[2].plot(epochs, map50_95, label="Val mAP@50-95", color="#8B5CF6", linewidth=2)
    axes[2].set_title("Validation Detection mAP", color="#F1F5F9")
    axes[2].set_xlabel("Epoch", color="#94A3B8")
    axes[2].legend(facecolor="#151E33", edgecolor="#1E293B", labelcolor="#F1F5F9")

    plt.tight_layout()
    dest = out / "training_loss_and_map_curves.png"
    plt.savefig(dest, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return str(dest)


def plot_ultralytics_results_png(
    results_csv_path: Path | str,
    output_png_path: Optional[Path | str] = None,
) -> Optional[str]:
    """
    Generate the standard 2x5 10-panel Ultralytics `results.png` figure directly from `results.csv`.
    Uses Ultralytics' official `plot_results` utility with a deterministic Matplotlib 2x5 fallback.
    """
    csv_p = Path(results_csv_path).resolve()
    if not csv_p.is_file():
        return None

    default_png = csv_p.with_name("results.png")
    target_png = Path(output_png_path).resolve() if output_png_path else default_png
    target_png.parent.mkdir(parents=True, exist_ok=True)

    generated = False
    try:
        from ultralytics.utils.plotting import plot_results

        plot_results(file=str(csv_p))
        if default_png.is_file():
            generated = True
            if target_png != default_png:
                shutil.copy2(default_png, target_png)
    except Exception:
        generated = False

    if not generated:
        rows: List[Dict[str, float]] = []
        with open(csv_p, "r", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                clean: Dict[str, float] = {}
                for k, v in raw.items():
                    if k and v:
                        try:
                            clean[k.strip()] = float(v.strip())
                        except ValueError:
                            continue
                if clean:
                    rows.append(clean)
        if not rows:
            return None

        epochs = [int(r.get("epoch", i + 1)) for i, r in enumerate(rows)]
        dfl_train_key = "train/dfl_loss" if "train/dfl_loss" in rows[0] else "train/l1_loss"
        dfl_val_key = "val/dfl_loss" if "val/dfl_loss" in rows[0] else "val/l1_loss"
        panels = [
            ("train/box_loss", "train/box_loss"),
            ("train/cls_loss", "train/cls_loss"),
            (dfl_train_key, dfl_train_key),
            ("metrics/precision(B)", "metrics/precision(B)"),
            ("metrics/recall(B)", "metrics/recall(B)"),
            ("val/box_loss", "val/box_loss"),
            ("val/cls_loss", "val/cls_loss"),
            (dfl_val_key, dfl_val_key),
            ("metrics/mAP50(B)", "metrics/mAP50(B)"),
            ("metrics/mAP50-95(B)", "metrics/mAP50-95(B)"),
        ]
        fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), dpi=160)
        for ax, (col, title) in zip(axes.flat, panels):
            vals = [r.get(col, 0.0) for r in rows]
            ax.plot(epochs, vals, marker=".", markersize=3, linewidth=1.6, color="#0284C7")
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(target_png, bbox_inches="tight")
        plt.close(fig)

    return str(target_png) if target_png.is_file() else None


def extract_checkpoint_training_artifacts(
    checkpoint_path: Path | str,
    output_dir: Path | str,
    experiment_name: str = "rdd2022_baseline",
    source_repository: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Extract authentic per-epoch training history (`ckpt['train_results']` and `ckpt['train_args']`)
    embedded inside an Ultralytics `.pt` checkpoint into `results.csv`, `results.png`,
    `training_loss_and_map_curves.png`, and `training_summary.json`.
    """
    ckpt_p = Path(checkpoint_path).resolve()
    if not ckpt_p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

    ckpt = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)
    train_results = ckpt.get("train_results")
    if not isinstance(train_results, dict) or not train_results:
        raise ValueError(f"Checkpoint {ckpt_p.name} does not contain embedded 'train_results' history.")

    columns = list(train_results.keys())
    num_rows = min(len(v) for v in train_results.values() if isinstance(v, (list, tuple)))
    if num_rows <= 0:
        raise ValueError(f"Checkpoint {ckpt_p.name} has empty 'train_results' series.")

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results_csv = out / "results.csv"

    history: List[Dict[str, float]] = []
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for idx in range(num_rows):
            row_vals = []
            row_dict: Dict[str, float] = {}
            for col in columns:
                val = train_results[col][idx]
                fval = float(val)
                if col == "epoch":
                    iv = int(round(fval))
                    row_vals.append(iv)
                    row_dict[col] = float(iv)
                else:
                    row_vals.append(round(fval, 6))
                    row_dict[col] = round(fval, 6)
            writer.writerow(row_vals)
            history.append(row_dict)

    results_png = plot_ultralytics_results_png(results_csv)
    loss_curves_png = plot_training_history_curves(results_csv, out)

    train_args = ckpt.get("train_args") or {}
    final_epoch = history[-1]
    best_epoch_row = max(history, key=lambda r: r.get("metrics/mAP50-95(B)", r.get("metrics/mAP50(B)", 0.0)))

    model_tag = str(train_args.get("name") or Path(str(train_args.get("model", "yolo11s"))).stem)
    if not model_tag.endswith(".pt"):
        resolved_arch = f"{model_tag}.pt"
    else:
        resolved_arch = model_tag

    summary: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "source_repository": source_repository,
        "resolved_architecture": resolved_arch,
        "optimizer": str(train_args.get("optimizer", "auto")),
        "epochs_configured": int(train_args.get("epochs", num_rows)),
        "epochs_completed": num_rows,
        "batch_size": int(train_args.get("batch", 16)),
        "image_size": int(train_args.get("imgsz", 640)),
        "run_directory": str(out),
        "best_checkpoint": str(ckpt_p),
        "results_csv": str(results_csv),
        "results_png": results_png,
        "loss_curves_plot": loss_curves_png,
        "validation_metrics_last_epoch": {
            "epoch": int(final_epoch.get("epoch", num_rows)),
            "precision": final_epoch.get("metrics/precision(B)"),
            "recall": final_epoch.get("metrics/recall(B)"),
            "map50": final_epoch.get("metrics/mAP50(B)"),
            "map50_95": final_epoch.get("metrics/mAP50-95(B)"),
        },
        "validation_metrics_best_epoch": {
            "epoch": int(best_epoch_row.get("epoch", num_rows)),
            "precision": best_epoch_row.get("metrics/precision(B)"),
            "recall": best_epoch_row.get("metrics/recall(B)"),
            "map50": best_epoch_row.get("metrics/mAP50(B)"),
            "map50_95": best_epoch_row.get("metrics/mAP50-95(B)"),
        },
        "training_history": history,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary_file = out / "training_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _plot_per_class_chart(per_class: List[Dict[str, Any]], output_dir: Path) -> str:
    codes = [c["class_code"] for c in per_class]
    prec = [c["precision"] for c in per_class]
    rec = [c["recall"] for c in per_class]
    f1 = [c["f1_score"] for c in per_class]
    ap50 = [c["map50"] for c in per_class]

    x = np.arange(len(codes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    fig.patch.set_facecolor("#0B101E")
    ax.set_facecolor("#10172A")

    ax.bar(x - 1.5 * width, prec, width, label="Precision", color="#3B82F6")
    ax.bar(x - 0.5 * width, rec, width, label="Recall", color="#06B6D4")
    ax.bar(x + 0.5 * width, f1, width, label="F1-Score", color="#10B981")
    ax.bar(x + 1.5 * width, ap50, width, label="mAP@50", color="#F59E0B")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c['class_code']}\n({c['class_name']})" for c in per_class], color="#F1F5F9")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", color="#CBD5E1")
    ax.set_title("Held-Out Test Set Performance by Road Damage Category", color="#F1F5F9", pad=12)
    ax.tick_params(colors="#CBD5E1")
    for spine in ax.spines.values():
        spine.set_color("#1E293B")
    ax.legend(facecolor="#151E33", edgecolor="#1E293B", labelcolor="#F1F5F9")

    plt.tight_layout()
    dest = output_dir / "per_class_performance.png"
    plt.savefig(dest, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return str(dest)


def _plot_confusion_matrix(matrix: List[List[float]], labels: List[str], output_dir: Path) -> str:
    arr = np.array(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=140)
    fig.patch.set_facecolor("#0B101E")
    ax.set_facecolor("#10172A")

    sns.heatmap(
        arr,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("True Class", color="#CBD5E1")
    ax.set_ylabel("Predicted Class", color="#CBD5E1")
    ax.set_title("Held-Out Test Set Confusion Matrix", color="#F1F5F9", pad=12)
    ax.tick_params(colors="#F1F5F9")

    plt.tight_layout()
    dest = output_dir / "confusion_matrix_heatmap.png"
    plt.savefig(dest, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return str(dest)


def evaluate_trained_model(
    checkpoint_path: Path | str,
    dataset_yaml: Path | str,
    split: str = "test",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.50,
    image_size: int = 640,
    device: str = "auto",
    output_dir: Path | str = "./outputs/experiments/evaluation",
    training_results_csv: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained YOLO checkpoint strictly on the held-out test set (or specified split),
    computing overall and per-class P, R, F1, mAP@50, mAP@50-95, confusion matrix, latency, and FPS.
    """
    ckpt = Path(checkpoint_path).resolve()
    yaml_p = Path(dataset_yaml).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Trained checkpoint not found: {ckpt}")
    if not yaml_p.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_p}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = (0 if torch.cuda.is_available() else "cpu") if device == "auto" else device

    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    t0 = time.perf_counter()
    metrics = model.val(
        data=str(yaml_p),
        split=split,
        imgsz=int(image_size),
        conf=float(conf_threshold),
        iou=float(iou_threshold),
        device=dev,
        plots=True,
        project=str(out_dir),
        name=f"eval_{split}",
        exist_ok=True,
        verbose=False,
    )
    wall_s = time.perf_counter() - t0

    box_m = metrics.box
    overall_p = round(float(box_m.mp), 4)
    overall_r = round(float(box_m.mr), 4)
    overall_f1 = _compute_f1(overall_p, overall_r)
    overall_map50 = round(float(box_m.map50), 4)
    overall_map50_95 = round(float(box_m.map), 4)

    speed_dict = getattr(metrics, "speed", {}) or {}
    infer_ms = float(speed_dict.get("inference", 0.0))
    total_ms = float(
        speed_dict.get("preprocess", 0.0)
        + infer_ms
        + speed_dict.get("postprocess", 0.0)
    )
    fps = round(1000.0 / max(total_ms, 0.01), 2) if total_ms > 0 else 0.0

    # Per-class breakdown
    per_class_list: List[Dict[str, Any]] = []
    ap_class_indices = [int(i) for i in getattr(box_m, "ap_class_index", [])]
    idx_lookup = {cls_i: pos for pos, cls_i in enumerate(ap_class_indices)}

    for cls_id in range(len(TARGET_CODES)):
        spec = ID_TO_SPEC[cls_id]
        if cls_id in idx_lookup:
            pos = idx_lookup[cls_id]
            cp = round(float(box_m.p[pos]), 4)
            cr = round(float(box_m.r[pos]), 4)
            cf1 = _compute_f1(cp, cr)
            cap50 = round(float(box_m.ap50[pos]), 4)
            cap = round(float(box_m.ap[pos]), 4)
        else:
            cp = cr = cf1 = cap50 = cap = 0.0

        per_class_list.append({
            "class_id": cls_id,
            "class_code": spec.code,
            "class_name": spec.name,
            "precision": cp,
            "recall": cr,
            "f1_score": cf1,
            "map50": cap50,
            "map50_95": cap,
        })

    cm_obj = getattr(metrics, "confusion_matrix", None)
    cm_matrix: List[List[float]] = []
    cm_labels = TARGET_CODES + ["background"]
    if cm_obj is not None and hasattr(cm_obj, "matrix"):
        cm_matrix = cm_obj.matrix.tolist()

    artifacts: Dict[str, Optional[str]] = {
        "per_class_chart": _plot_per_class_chart(per_class_list, out_dir),
        "confusion_matrix_plot": _plot_confusion_matrix(cm_matrix, cm_labels, out_dir) if cm_matrix else None,
        "loss_curves_plot": plot_training_history_curves(training_results_csv, out_dir)
        if training_results_csv
        else None,
    }

    eval_save_dir = Path(getattr(metrics, "save_dir", out_dir / f"eval_{split}"))
    for plot_name in ("BoxPR_curve.png", "PR_curve.png", "BoxF1_curve.png", "F1_curve.png", "confusion_matrix.png"):
        cand = eval_save_dir / plot_name
        if cand.is_file():
            artifacts[plot_name] = str(cand)

    report = {
        "checkpoint_path": str(ckpt),
        "dataset_yaml": str(yaml_p),
        "evaluated_split": split,
        "precision": overall_p,
        "recall": overall_r,
        "f1_score": overall_f1,
        "map50": overall_map50,
        "map50_95": overall_map50_95,
        "inference_ms": round(infer_ms, 3),
        "end_to_end_ms": round(total_ms, 3),
        "fps": fps,
        "evaluation_duration_s": round(wall_s, 2),
        "per_class": per_class_list,
        "confusion_matrix": {
            "labels": cm_labels,
            "matrix": cm_matrix,
        },
        "artifacts": artifacts,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_json = out_dir / "evaluation_summary.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
