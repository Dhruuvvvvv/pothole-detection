"""Model training and domain-valid augmentation modules (Module 2)."""

from .augmentations import build_albumentations_pipeline, get_yolo_augmentation_hyperparams
from .trainer import TrainingConfig, train_yolo_detector

__all__ = [
    "TrainingConfig",
    "build_albumentations_pipeline",
    "get_yolo_augmentation_hyperparams",
    "train_yolo_detector",
]
