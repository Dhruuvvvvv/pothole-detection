"""
Video Road Damage Detection & Multi-Object Tracking Pipeline (Module 4).

Supports MP4, AVI, and MOV files using OpenCV frame decoding, YOLO frame detection,
and configurable **ByteTrack** (two-stage high/low confidence IoU + velocity association)
or **SORT** (single-stage constant-velocity IoU association) multi-object tracking.

IMPORTANT METHODOLOGY NOTE:
Frame-level detection counts are strictly separated from local trajectory Track IDs.
Because dashcam perspective foreshortening, brief occlusions, and crack fragmentation can
split or reset trajectories, Track IDs represent local frame-to-frame continuity rather
than globally unique physical potholes or cracks along a highway corridor.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from ml.assessment.severity_engine import (
    AssessedDetection,
    PriorityHeuristicConfig,
    assess_detections_batch,
)
from ml.constants import TARGET_CODES, VIDEO_EXTENSIONS
from ml.inference.detector import RoadDamageDetector, draw_assessed_detections


VIDEO_TRACKING_DISCLAIMER = (
    "Frame-level detections count every bounding box observed across processed video frames. "
    "Unique Track IDs reflect local multi-frame trajectory continuity (ByteTrack/SORT) and must not "
    "be interpreted as exact counts of globally unique physical road defects without GPS/odometry fusion."
)


def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix [N, M] between two arrays of [x1, y1, x2, y2] boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 1e-6, inter / union, 0.0).astype(np.float32)


@dataclass
class _TrackState:
    track_id: int
    class_id: int
    bbox: np.ndarray          # [x1, y1, x2, y2]
    velocity: np.ndarray      # [vx1, vy1, vx2, vy2] damped linear motion model
    confidence: float
    hits: int = 1
    age: int = 1
    time_since_update: int = 0

    def predict(self) -> np.ndarray:
        self.age += 1
        self.time_since_update += 1
        self.bbox = self.bbox + self.velocity
        self.velocity *= 0.85
        return self.bbox

    def update(self, new_bbox: np.ndarray, conf: float, class_id: int) -> None:
        measured_vel = new_bbox - self.bbox
        self.velocity = 0.6 * self.velocity + 0.4 * measured_vel
        self.bbox = new_bbox.copy()
        self.confidence = conf
        self.class_id = class_id
        self.hits += 1
        self.time_since_update = 0


class ByteSortTracker:
    """
    Deterministic multi-object tracker supporting both:
      - `mode='bytetrack'`: Two-stage Hungarian IoU association matching high-confidence detections
        first, followed by low-confidence recovery against unmatched active tracks (Zhang et al., 2022).
      - `mode='sort'`: Single-stage constant-velocity Hungarian IoU association (Bewley et al., 2016).
    """

    def __init__(
        self,
        mode: str = "bytetrack",
        high_thresh: float = 0.35,
        low_thresh: float = 0.15,
        iou_threshold: float = 0.25,
        max_age: int = 12,
        min_hits: int = 1,
    ) -> None:
        self.mode = mode.lower()
        if self.mode not in {"bytetrack", "sort"}:
            raise ValueError("Tracker mode must be 'bytetrack' or 'sort'")
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._next_id = 1
        self._tracks: List[_TrackState] = []

    def _associate(
        self,
        tracks: List[_TrackState],
        dets: List[Dict[str, Any]],
        iou_thresh: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        trk_boxes = np.array([t.bbox for t in tracks], dtype=np.float32)
        det_boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets], dtype=np.float32)
        iou_mat = _compute_iou_matrix(trk_boxes, det_boxes)

        # Penalize cross-class switches mildly while allowing brief class flicker
        for ti, trk in enumerate(tracks):
            for di, det in enumerate(dets):
                if int(det["class_id"]) != trk.class_id:
                    iou_mat[ti, di] *= 0.5

        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        matches: List[Tuple[int, int]] = []
        unmatched_trks = set(range(len(tracks)))
        unmatched_dets = set(range(len(dets)))

        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= iou_thresh:
                matches.append((int(r), int(c)))
                unmatched_trks.discard(int(r))
                unmatched_dets.discard(int(c))

        return matches, sorted(unmatched_trks), sorted(unmatched_dets)

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Advance track states by one frame and return detections enriched with `track_id`."""
        for trk in self._tracks:
            trk.predict()

        if self.mode == "bytetrack":
            high_dets = [d for d in detections if float(d["confidence"]) >= self.high_thresh]
            low_dets = [
                d for d in detections
                if self.low_thresh <= float(d["confidence"]) < self.high_thresh
            ]
        else:
            high_dets = [d for d in detections if float(d["confidence"]) >= self.low_thresh]
            low_dets = []

        # Stage 1: Match high-confidence detections against all active tracks
        matches_1, unmatched_trk_idx, unmatched_high_idx = self._associate(
            self._tracks, high_dets, self.iou_threshold
        )

        output_dets: List[Dict[str, Any]] = []
        for ti, di in matches_1:
            det = high_dets[di]
            box = np.array([det["x1"], det["y1"], det["x2"], det["y2"]], dtype=np.float32)
            self._tracks[ti].update(box, float(det["confidence"]), int(det["class_id"]))
            if self._tracks[ti].hits >= self.min_hits:
                output_dets.append({**det, "track_id": self._tracks[ti].track_id})

        # Stage 2 (ByteTrack): Match low-confidence detections against remaining unmatched tracks
        if self.mode == "bytetrack" and unmatched_trk_idx and low_dets:
            rem_tracks = [self._tracks[i] for i in unmatched_trk_idx]
            matches_2, _, _ = self._associate(rem_tracks, low_dets, max(0.15, self.iou_threshold - 0.05))
            for rel_ti, di in matches_2:
                orig_ti = unmatched_trk_idx[rel_ti]
                det = low_dets[di]
                box = np.array([det["x1"], det["y1"], det["x2"], det["y2"]], dtype=np.float32)
                self._tracks[orig_ti].update(box, float(det["confidence"]), int(det["class_id"]))
                if self._tracks[orig_ti].hits >= self.min_hits:
                    output_dets.append({**det, "track_id": self._tracks[orig_ti].track_id})

        # Initialize new tracks for unmatched high-confidence detections
        for di in unmatched_high_idx:
            det = high_dets[di]
            box = np.array([det["x1"], det["y1"], det["x2"], det["y2"]], dtype=np.float32)
            new_trk = _TrackState(
                track_id=self._next_id,
                class_id=int(det["class_id"]),
                bbox=box,
                velocity=np.zeros(4, dtype=np.float32),
                confidence=float(det["confidence"]),
            )
            self._next_id += 1
            self._tracks.append(new_trk)
            if new_trk.hits >= self.min_hits:
                output_dets.append({**det, "track_id": new_trk.track_id})

        # Prune expired tracks
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]
        return output_dets


@dataclass
class FrameStatEntry:
    frame_index: int
    timestamp_sec: float
    detection_count: int
    high_priority_count: int
    active_track_ids: List[int]
    class_counts: Dict[str, int]


@dataclass
class VideoAnalysisResult:
    source_video_path: str
    output_video_path: str
    tracker_used: str
    video_width: int
    video_height: int
    source_fps: float
    total_video_frames: int
    processed_frames: int
    frame_stride: int
    total_frame_level_detections: int
    unique_track_ids_observed: int
    class_frame_detection_distribution: Dict[str, int]
    priority_band_distribution: Dict[str, int]
    average_processing_fps: float
    processing_duration_sec: float
    frame_timeline: List[FrameStatEntry] = field(default_factory=list)
    sampled_detections: List[Dict[str, Any]] = field(default_factory=list)
    methodology_disclaimer: str = VIDEO_TRACKING_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _open_video_writer_with_fallback(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> Tuple[cv2.VideoWriter, Path]:
    """Open a VideoWriter cleanly across Windows and Linux codecs without missing-DLL errors."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_fps = max(1.0, min(120.0, float(fps or 25.0)))

    for codec, ext in (("mp4v", ".mp4"), ("VP80", ".webm"), ("XVID", ".avi")):
        candidate = output_path.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(candidate), fourcc, safe_fps, (width, height))
        if writer.isOpened():
            return writer, candidate
        writer.release()

    raise RuntimeError(f"OpenCV VideoWriter could not initialize any supported codec for {output_path}")


def process_road_video(
    video_path: Path | str,
    output_video_path: Path | str,
    detector: RoadDamageDetector,
    tracker_type: str = "bytetrack",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    image_size: int = 640,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    priority_config: Optional[PriorityHeuristicConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> VideoAnalysisResult:
    """
    Decode road video, run YOLO object detection + ByteTrack/SORT tracking across frames,
    render annotated output video with HUD telemetry, and compile frame-level analytics.
    """
    in_path = Path(video_path).resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Video file not found: {in_path}")
    if in_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format '{in_path.suffix}'. Supported: {sorted(VIDEO_EXTENSIONS)}")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise ValueError(f"Could not decode video stream: {in_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if src_fps <= 0 or src_fps > 240:
        src_fps = 25.0
    total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions reported by decoder.")

    stride = max(1, int(frame_stride))
    out_fps = max(1.0, src_fps / stride)
    writer, final_out_path = _open_video_writer_with_fallback(Path(output_video_path), out_fps, width, height)

    # For ByteTrack two-stage association, query detector with slightly lower floor so low-conf boxes can recover tracks
    query_conf = max(0.10, conf_threshold * 0.6) if tracker_type.lower() == "bytetrack" else conf_threshold
    tracker = ByteSortTracker(
        mode=tracker_type.lower(),
        high_thresh=conf_threshold,
        low_thresh=query_conf,
        iou_threshold=0.25,
    )

    class_totals = Counter({code: 0 for code in TARGET_CODES})
    band_totals = Counter({"High": 0, "Medium": 0, "Low": 0})
    observed_track_ids: set[int] = set()
    timeline: List[FrameStatEntry] = []
    sampled_dets: List[Dict[str, Any]] = []

    frame_idx = 0
    processed_count = 0
    total_dets = 0
    t_start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            if max_frames is not None and processed_count >= max_frames:
                break

            raw_dets, _ = detector.predict_raw_frame(
                img_bgr=frame,
                conf_threshold=query_conf,
                iou_threshold=iou_threshold,
                image_size=image_size,
            )
            tracked_dets = tracker.update(raw_dets)
            for td in tracked_dets:
                td["frame_index"] = frame_idx

            assessed: List[AssessedDetection] = assess_detections_batch(
                raw_detections=tracked_dets,
                image_width=width,
                image_height=height,
                config=priority_config,
            )

            frame_cls = Counter({code: 0 for code in TARGET_CODES})
            active_ids: List[int] = []
            high_in_frame = 0

            for d in assessed:
                total_dets += 1
                class_totals[d.class_code] += 1
                frame_cls[d.class_code] += 1
                band_totals[d.priority_band] += 1
                if d.priority_band == "High":
                    high_in_frame += 1
                if d.track_id is not None:
                    observed_track_ids.add(d.track_id)
                    active_ids.append(d.track_id)
                if len(sampled_dets) < 500:
                    sampled_dets.append(d.to_dict())

            annotated = draw_assessed_detections(frame, assessed)

            # Draw subtle telemetry bar at top-left
            elapsed_now = max(time.perf_counter() - t_start, 1e-4)
            live_fps = (processed_count + 1) / elapsed_now
            hud = (
                f"RoadGuard AI | Frame {frame_idx} | Dets: {len(assessed)} | "
                f"Tracker: {tracker_type.upper()} | {live_fps:.1f} FPS"
            )
            cv2.rectangle(annotated, (10, 10), (min(width - 10, 560), 42), (11, 16, 30), -1)
            cv2.putText(
                annotated,
                hud,
                (18, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (241, 245, 249),
                1,
                cv2.LINE_AA,
            )

            writer.write(annotated)
            processed_count += 1

            timeline.append(
                FrameStatEntry(
                    frame_index=frame_idx,
                    timestamp_sec=round(frame_idx / src_fps, 3),
                    detection_count=len(assessed),
                    high_priority_count=high_in_frame,
                    active_track_ids=sorted(set(active_ids)),
                    class_counts=dict(frame_cls),
                )
            )

            if progress_callback is not None:
                est_total = max(1, (total_frames // stride) if total_frames > 0 else processed_count)
                progress_callback(processed_count, est_total)

            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    duration_s = max(time.perf_counter() - t_start, 1e-4)
    avg_fps = round(processed_count / duration_s, 2)

    return VideoAnalysisResult(
        source_video_path=str(in_path),
        output_video_path=str(final_out_path),
        tracker_used=tracker_type.lower(),
        video_width=width,
        video_height=height,
        source_fps=round(src_fps, 2),
        total_video_frames=total_frames if total_frames > 0 else frame_idx,
        processed_frames=processed_count,
        frame_stride=stride,
        total_frame_level_detections=total_dets,
        unique_track_ids_observed=len(observed_track_ids),
        class_frame_detection_distribution=dict(class_totals),
        priority_band_distribution=dict(band_totals),
        average_processing_fps=avg_fps,
        processing_duration_sec=round(duration_s, 3),
        frame_timeline=timeline,
        sampled_detections=sampled_dets,
    )
