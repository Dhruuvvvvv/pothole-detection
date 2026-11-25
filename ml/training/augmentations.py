"""Domain-valid data augmentation configurations for road surface damage detection."""

from __future__ import annotations

from typing import Any, Dict

import albumentations as A


def get_yolo_augmentation_hyperparams() -> Dict[str, Any]:
    """
    Return Ultralytics YOLO augmentation hyperparameters tailored to road imagery:
      - Horizontal flip (fliplr=0.5) is physically valid (symmetric road lanes).
      - Vertical flip (flipud=0.0) is DISABLED because upside-down perspective roads
        violate camera geometry and invert longitudinal/transverse context.
      - Mild rotation (±5 deg) models vehicle pitch/roll without turning D00 into D10!
        (A 90-degree rotation would turn a longitudinal crack into a transverse crack!)
      - Brightness/value & saturation jitter models shadows, overcast sky, and glare.
    """
    return {
        "hsv_h": 0.012,
        "hsv_s": 0.45,
        "hsv_v": 0.35,
        "degrees": 5.0,
        "translate": 0.10,
        "scale": 0.35,
        "shear": 1.5,
        "perspective": 0.0002,
        "flipud": 0.0,
        "fliplr": 0.50,
        "mosaic": 1.0,
        "mixup": 0.0,
        "close_mosaic": 10,
    }


def build_albumentations_pipeline(image_size: int = 640) -> A.Compose:
    """
    Build a realistic Albumentations bounding-box augmentation pipeline for offline
    dataset inspection or custom PyTorch training loops.
    Note: Rotation is strictly bounded to ±5 degrees so longitudinal (D00) and
    transverse (D10) crack orientations remain semantically valid.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                rotate=(-5, 5),
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.22, contrast_limit=0.22, p=0.5),
            A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.2),
            A.Resize(height=image_size, width=image_size),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.25,
            clip=True,
        ),
    )
