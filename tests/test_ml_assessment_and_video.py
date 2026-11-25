"""Unit tests for Phases D, E, and F — Detector Safeguards, Damage Assessment, and Video Tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pytest

from ml.assessment.severity_engine import (
    PriorityHeuristicConfig,
    assess_detection,
    summarize_category_assessments,
)
from ml.inference.detector import ModelNotConfiguredError, RoadDamageDetector
from ml.tracking.video_processor import ByteSortTracker, process_road_video


def test_priority_heuristic_ordering_and_custom_thresholds() -> None:
    cfg = PriorityHeuristicConfig(low_max_threshold=35.0, high_min_threshold=65.0)
    # Small low-confidence longitudinal crack -> Low priority
    d_low = assess_detection(
        class_id=0,  # D00
        confidence=0.30,
        x1=100,
        y1=100,
        x2=120,
        y2=120,
        image_width=1000,
        image_height=1000,
        config=cfg,
    )
    # Large high-confidence pothole -> High priority
    d_high = assess_detection(
        class_id=3,  # D40 Pothole
        confidence=0.92,
        x1=200,
        y1=300,
        x2=600,
        y2=700,
        image_width=1000,
        image_height=1000,
        config=cfg,
    )

    assert d_low.priority_band == "Low"
    assert d_high.priority_band == "High"
    assert d_high.priority_score > d_low.priority_score

    summary = summarize_category_assessments([d_low, d_high])
    assert summary["total_detections"] == 2
    assert summary["priority_bands"]["High"] == 1
    assert summary["priority_bands"]["Low"] == 1

    # Invalid threshold ordering must fail
    with pytest.raises(ValueError):
        PriorityHeuristicConfig(low_max_threshold=70.0, high_min_threshold=40.0).validate()


def test_detector_missing_weights_graceful_state(tmp_path: Path) -> None:
    missing_pt = tmp_path / "nonexistent_best.pt"
    det = RoadDamageDetector(checkpoint_path=missing_pt)
    status = det.get_status()
    assert status["configured"] is False
    assert status["status"] == "model_not_configured"

    dummy_img = np.zeros((120, 160, 3), dtype=np.uint8)
    with pytest.raises(ModelNotConfiguredError):
        det.predict_image(dummy_img)


def test_bytesort_two_stage_low_confidence_recovery() -> None:
    tracker = ByteSortTracker(mode="bytetrack", high_thresh=0.5, low_thresh=0.15, iou_threshold=0.25)

    # Frame 0: High confidence pothole initializes Track #1
    f0 = tracker.update([{"class_id": 3, "confidence": 0.85, "x1": 100, "y1": 100, "x2": 180, "y2": 180}])
    assert len(f0) == 1
    tid = f0[0]["track_id"]

    # Frame 1: Partial blur drops confidence to 0.22 (between low_thresh and high_thresh) -> ByteTrack Stage 2 keeps Track #1!
    f1 = tracker.update([{"class_id": 3, "confidence": 0.22, "x1": 104, "y1": 103, "x2": 184, "y2": 183}])
    assert len(f1) == 1
    assert f1[0]["track_id"] == tid


class _StubRoadDetector(RoadDamageDetector):
    """Deterministic 4-class detector stub for testing video pipeline end-to-end without fabricated weights."""

    def __init__(self) -> None:
        self.checkpoint_path = Path("stub_best.pt")
        self.requested_device = "cpu"
        self._model = True
        self._model_class_map = {0: 0, 1: 1, 2: 2, 3: 3}
        self._raw_model_names = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
        self._load_error = None
        self._frame_counter = 0

    def predict_raw_frame(
        self,
        img_bgr: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
    ) -> Tuple[List[Dict[str, Any]], float]:
        i = self._frame_counter
        self._frame_counter += 1
        # Simulate a pothole moving gradually downward in dashcam perspective
        return (
            [
                {
                    "class_id": 3,
                    "confidence": 0.82,
                    "x1": 100.0 + i * 3.0,
                    "y1": 60.0 + i * 4.0,
                    "x2": 170.0 + i * 3.0,
                    "y2": 120.0 + i * 4.0,
                }
            ],
            4.5,
        )


def test_video_analytics_pipeline_end_to_end(tmp_path: Path) -> None:
    in_video = tmp_path / "dashcam_test.avi"
    writer = cv2.VideoWriter(str(in_video), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (320, 240))
    assert writer.isOpened()
    for f in range(10):
        frame = np.full((240, 320, 3), 110, dtype=np.uint8)
        cv2.rectangle(frame, (100 + f * 3, 60 + f * 4), (170 + f * 3, 120 + f * 4), (30, 30, 30), -1)
        writer.write(frame)
    writer.release()

    stub_detector = _StubRoadDetector()
    out_video = tmp_path / "annotated_out.mp4"
    res = process_road_video(
        video_path=in_video,
        output_video_path=out_video,
        detector=stub_detector,
        tracker_type="bytetrack",
        conf_threshold=0.30,
    )

    assert res.processed_frames == 10
    assert res.total_frame_level_detections == 10
    assert res.unique_track_ids_observed == 1  # Continuous trajectory maintained as single Track ID!
    assert res.class_frame_detection_distribution["D40"] == 10
    assert Path(res.output_video_path).is_file()
    assert Path(res.output_video_path).stat().st_size > 1000
