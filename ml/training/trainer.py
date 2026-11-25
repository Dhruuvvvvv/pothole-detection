"""Configurable Ultralytics YOLO training pipeline with GPU/CPU support, resume, and experiment tracking."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ml.training.augmentations import get_yolo_augmentation_hyperparams


SUPPORTED_BASELINE_MODELS: List[str] = [
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolov8n.pt",
    "yolov8s.pt",
]


@dataclass
class TrainingConfig:
    dataset_yaml: str
    model_architecture: str = "yolo26n.pt"
    epochs: int = 50
    batch_size: int = 16
    image_size: int = 640
    patience: int = 12
    learning_rate: float = 0.01
    optimizer: str = "auto"
    device: str = "auto"
    workers: int = 4
    seed: int = 42
    project_dir: str = "./outputs/experiments"
    experiment_name: str = "rdd2022_baseline"
    resume: bool = False
    export_best_to: Optional[str] = "./models/best.pt"

    def resolve_device(self) -> str | int:
        if self.device != "auto":
            return self.device
        return 0 if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_ultralytics_model_with_fallback(requested_arch: str):
    """
    Initialize the requested Ultralytics YOLO model (e.g. yolo26n.pt).
    If a specific pretrained weight tag is unavailable in the installed Ultralytics version,
    transparently fall back through yolo11n.pt -> yolov8n.pt only when `requested_arch` is the default.
    """
    from ultralytics import YOLO

    try:
        return YOLO(requested_arch), requested_arch
    except Exception as primary_err:
        if Path(requested_arch).suffix == ".pt" and requested_arch == "yolo26n.pt":
            for candidate in ("yolo11n.pt", "yolov8n.pt"):
                try:
                    return YOLO(candidate), candidate
                except Exception:
                    continue
        raise RuntimeError(
            f"Unable to initialize Ultralytics model '{requested_arch}': {primary_err}"
        ) from primary_err


def _parse_results_csv(csv_path: Path) -> List[Dict[str, float]]:
    if not csv_path.is_file():
        return []
    rows: List[Dict[str, float]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            clean: Dict[str, float] = {}
            for k, v in raw.items():
                if k is None or v is None:
                    continue
                key = k.strip()
                try:
                    clean[key] = float(v.strip())
                except ValueError:
                    continue
            if clean:
                rows.append(clean)
    return rows


def train_yolo_detector(config: TrainingConfig) -> Dict[str, Any]:
    """
    Execute transfer-learning training on the prepared RDD2022 YOLO dataset,
    save checkpoints & training history, and export `best.pt` for application use.
    """
    yaml_path = Path(config.dataset_yaml).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    project_path = Path(config.project_dir).resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    device = config.resolve_device()
    model, resolved_arch = _load_ultralytics_model_with_fallback(config.model_architecture)

    aug_params = get_yolo_augmentation_hyperparams()
    train_kwargs: Dict[str, Any] = {
        "data": str(yaml_path),
        "epochs": int(config.epochs),
        "imgsz": int(config.image_size),
        "batch": int(config.batch_size),
        "patience": int(config.patience),
        "lr0": float(config.learning_rate),
        "optimizer": config.optimizer,
        "device": device,
        "workers": int(config.workers),
        "seed": int(config.seed),
        "project": str(project_path),
        "name": config.experiment_name,
        "exist_ok": True,
        "pretrained": True,
        "save": True,
        "plots": True,
        "resume": bool(config.resume),
        **aug_params,
    }

    train_results = model.train(**train_kwargs)
    save_dir = Path(getattr(train_results, "save_dir", project_path / config.experiment_name))

    best_ckpt = save_dir / "weights" / "best.pt"
    last_ckpt = save_dir / "weights" / "last.pt"
    results_csv = save_dir / "results.csv"
    results_png = save_dir / "results.png"

    exported_ckpt: Optional[str] = None
    if config.export_best_to and best_ckpt.is_file():
        dest = Path(config.export_best_to).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt, dest)
        exported_ckpt = str(dest)

    history = _parse_results_csv(results_csv)
    final_epoch = history[-1] if history else {}

    summary = {
        "experiment_name": config.experiment_name,
        "requested_architecture": config.model_architecture,
        "resolved_architecture": resolved_arch,
        "device_used": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset_yaml": str(yaml_path),
        "epochs_configured": config.epochs,
        "epochs_completed": len(history),
        "batch_size": config.batch_size,
        "image_size": config.image_size,
        "run_directory": str(save_dir),
        "best_checkpoint": str(best_ckpt) if best_ckpt.is_file() else None,
        "last_checkpoint": str(last_ckpt) if last_ckpt.is_file() else None,
        "results_csv": str(results_csv) if results_csv.is_file() else None,
        "results_png": str(results_png) if results_png.is_file() else None,
        "exported_application_checkpoint": exported_ckpt,
        "validation_metrics_last_epoch": {
            "precision": final_epoch.get("metrics/precision(B)"),
            "recall": final_epoch.get("metrics/recall(B)"),
            "map50": final_epoch.get("metrics/mAP50(B)"),
            "map50_95": final_epoch.get("metrics/mAP50-95(B)"),
        },
        "training_history": history,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary_file = save_dir / "training_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
