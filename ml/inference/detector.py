"""Singleton YOLO Road Damage Detector with strict class-schema verification and CPU/GPU support."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from ml.assessment.severity_engine import (
    AssessedDetection,
    PriorityHeuristicConfig,
    assess_detections_batch,
    summarize_category_assessments,
)
from ml.constants import (
    CANONICAL_LABEL_ALIASES,
    CODE_TO_ID,
    ID_TO_SPEC,
    TARGET_CODES,
    normalize_label_token,
)


class ModelNotConfiguredError(RuntimeError):
    """Raised when inference is requested before a trained 4-class RDD2022 checkpoint is placed in models/."""


def draw_assessed_detections(
    img_bgr: np.ndarray,
    detections: List[AssessedDetection],
) -> np.ndarray:
    """Render clean bounding boxes, class codes, confidence scores, priority bands, and track IDs."""
    canvas = img_bgr.copy()
    h, w = canvas.shape[:2]
    thickness = max(2, int(round(min(h, w) / 320)))
    font_scale = max(0.48, min(0.85, min(h, w) / 1100.0))

    for det in detections:
        spec = ID_TO_SPEC.get(det.class_id)
        color = spec.bgr_color if spec else (246, 130, 59)
        x1, y1 = int(round(det.x1)), int(round(det.y1))
        x2, y2 = int(round(det.x2)), int(round(det.y2))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        prefix = f"#{det.track_id} " if det.track_id is not None else ""
        caption = f"{prefix}{det.class_code} {det.confidence * 100:.1f}% [{det.priority_band}]"
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_y = max(th + baseline + 6, y1)
        bg_x2 = min(w - 1, x1 + tw + 10)
        cv2.rectangle(canvas, (x1, label_y - th - baseline - 6), (bg_x2, label_y), color, -1)
        cv2.putText(
            canvas,
            caption,
            (x1 + 5, label_y - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


class RoadDamageDetector:
    """
    Thread-safe wrapper around Ultralytics YOLO that loads trained weights once,
    verifies class names map cleanly to {D00, D10, D20, D40}, and runs CPU/GPU inference.
    """

    def __init__(
        self,
        checkpoint_path: Path | str = "./models/best.pt",
        device: str = "auto",
    ) -> None:
        self._lock = threading.Lock()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.requested_device = device
        self._model: Any = None
        self._model_class_map: Dict[int, int] = {}
        self._raw_model_names: Dict[int, str] = {}
        self._load_error: Optional[str] = None
        self._loaded_at: Optional[float] = None
        self.reload()

    def _resolve_device(self) -> str | int:
        if self.requested_device != "auto":
            return self.requested_device
        return 0 if torch.cuda.is_available() else "cpu"

    def reload(self, new_checkpoint_path: Optional[Path | str] = None) -> Dict[str, Any]:
        """Load or hot-reload the YOLO checkpoint without restarting the application."""
        with self._lock:
            if new_checkpoint_path is not None:
                self.checkpoint_path = Path(new_checkpoint_path).resolve()

            self._model = None
            self._model_class_map = {}
            self._raw_model_names = {}
            self._load_error = None

            if not self.checkpoint_path.is_file():
                self._load_error = (
                    f"Trained checkpoint not found at '{self.checkpoint_path}'. "
                    "Train on RDD2022 using notebooks/03_model_training.ipynb (or ml/training/trainer.py) "
                    "and place 'best.pt' in the models/ directory."
                )
                return self.get_status()

            try:
                from ultralytics import YOLO

                loaded = YOLO(str(self.checkpoint_path))
                names_attr = getattr(loaded, "names", {})
                if isinstance(names_attr, list):
                    raw_names = {i: str(n) for i, n in enumerate(names_attr)}
                elif isinstance(names_attr, dict):
                    raw_names = {int(k): str(v) for k, v in names_attr.items()}
                else:
                    raw_names = {}

                # Verify that checkpoint classes map to RDD2022 classes (reject generic 80-class COCO weights!)
                mapped: Dict[int, int] = {}
                for cid, cname in raw_names.items():
                    canonical = CANONICAL_LABEL_ALIASES.get(normalize_label_token(cname))
                    if canonical is not None:
                        mapped[cid] = CODE_TO_ID[canonical]

                if not mapped:
                    self._load_error = (
                        f"Checkpoint '{self.checkpoint_path.name}' has classes {list(raw_names.values())[:10]}, "
                        f"which do not match RoadGuard AI target categories {TARGET_CODES}. "
                        "Refusing to run mislabelled inference."
                    )
                    return self.get_status()

                self._model = loaded
                self._raw_model_names = raw_names
                self._model_class_map = mapped
                self._loaded_at = time.time()
            except Exception as exc:
                self._load_error = f"Failed to load checkpoint '{self.checkpoint_path}': {exc}"

            return self.get_status()

    @property
    def is_ready(self) -> bool:
        return self._model is not None and bool(self._model_class_map)

    def get_status(self) -> Dict[str, Any]:
        dev = self._resolve_device()
        return {
            "configured": self.is_ready,
            "status": "ready" if self.is_ready else "model_not_configured",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_exists": self.checkpoint_path.is_file(),
            "checkpoint_size_mb": round(self.checkpoint_path.stat().st_size / (1024 * 1024), 2)
            if self.checkpoint_path.is_file()
            else None,
            "device": str(dev),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "model_classes": self._raw_model_names,
            "target_classes": {spec_id: ID_TO_SPEC[spec_id].code for spec_id in sorted(ID_TO_SPEC)},
            "message": "Trained RDD2022 model loaded and ready." if self.is_ready else self._load_error,
        }

    def predict_raw_frame(
        self,
        img_bgr: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Run single-frame detection and return (raw_detection_dicts, inference_ms)."""
        if not self.is_ready:
            raise ModelNotConfiguredError(self._load_error or "Model checkpoint is not configured.")

        dev = self._resolve_device()
        t0 = time.perf_counter()
        results = self._model.predict(
            source=img_bgr,
            conf=float(conf_threshold),
            iou=float(iou_threshold),
            imgsz=int(image_size),
            device=dev,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Dict[str, Any]] = []
        if results:
            res = results[0]
            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy_arr = boxes.xyxy.detach().cpu().numpy()
                conf_arr = boxes.conf.detach().cpu().numpy()
                cls_arr = boxes.cls.detach().cpu().numpy().astype(int)
                for (x1, y1, x2, y2), conf, raw_cls in zip(xyxy_arr, conf_arr, cls_arr):
                    if int(raw_cls) not in self._model_class_map:
                        continue
                    target_cls = self._model_class_map[int(raw_cls)]
                    detections.append({
                        "class_id": target_cls,
                        "confidence": float(conf),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    })
        return detections, round(inference_ms, 3)

    def predict_image(
        self,
        img_bgr: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        priority_config: Optional[PriorityHeuristicConfig] = None,
    ) -> Tuple[np.ndarray, List[AssessedDetection], Dict[str, Any]]:
        """Run multi-instance detection, damage assessment, and bounding-box visualization."""
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Input image is empty or unreadable.")

        h, w = img_bgr.shape[:2]
        raw_dets, inference_ms = self.predict_raw_frame(
            img_bgr=img_bgr,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
        )
        assessed = assess_detections_batch(
            raw_detections=raw_dets,
            image_width=w,
            image_height=h,
            config=priority_config,
        )
        annotated = draw_assessed_detections(img_bgr, assessed)
        summary = summarize_category_assessments(assessed)
        summary.update({
            "image_width": w,
            "image_height": h,
            "inference_time_ms": inference_ms,
            "fps": round(1000.0 / max(inference_ms, 0.001), 2),
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "image_size": image_size,
        })
        return annotated, assessed, summary


_GLOBAL_DETECTOR: Optional[RoadDamageDetector] = None
_SINGLETON_LOCK = threading.Lock()


def get_detector_singleton(
    checkpoint_path: Path | str = "./models/best.pt",
    device: str = "auto",
) -> RoadDamageDetector:
    global _GLOBAL_DETECTOR
    with _SINGLETON_LOCK:
        if _GLOBAL_DETECTOR is None:
            _GLOBAL_DETECTOR = RoadDamageDetector(checkpoint_path=checkpoint_path, device=device)
        return _GLOBAL_DETECTOR
