"""Tests for Phase D — Augmentation Pipeline and Training Configuration."""

from __future__ import annotations

import numpy as np
import pytest

from ml.training.augmentations import (
    build_albumentations_pipeline,
    get_yolo_augmentation_hyperparams,
)
from ml.training.trainer import TrainingConfig


def test_road_augmentation_hyperparams_preserve_road_geometry() -> None:
    hp = get_yolo_augmentation_hyperparams()
    # Vertical flip must be 0.0 so upside-down road geometry is never generated
    assert hp["flipud"] == 0.0
    assert hp["fliplr"] == 0.50
    # Rotation must stay small so D00 (longitudinal) does not become D10 (transverse)
    assert hp["degrees"] <= 10.0


def test_albumentations_bbox_pipeline_executes_cleanly() -> None:
    pipe = build_albumentations_pipeline(image_size=320)
    img = np.full((240, 360, 3), 120, dtype=np.uint8)
    bboxes = [[0.5, 0.5, 0.3, 0.25]]
    labels = [3]

    out = pipe(image=img, bboxes=bboxes, class_labels=labels)
    assert out["image"].shape == (320, 320, 3)
    assert len(out["bboxes"]) == 1
    assert out["class_labels"] == [3]


def test_training_config_device_resolution() -> None:
    cfg = TrainingConfig(dataset_yaml="./outputs/dataset/dataset_config.yaml", device="cpu")
    assert cfg.resolve_device() == "cpu"
    d = cfg.to_dict()
    assert d["epochs"] == 50
    assert d["image_size"] == 640
