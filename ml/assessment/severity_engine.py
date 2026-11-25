"""
Damage Assessment Engine (Module 3).

IMPORTANT ENGINEERING DISCLAIMER:
RDD2022 provides 2D bounding-box object annotations without physical depth maps,
crack width gauges, or camera intrinsics/extrinsics calibration.
Accordingly, this module computes an **Estimated Visual Priority Score** (0 - 100)
from transparent 2D image-space indicators:
  1. Normalized bounding-box area fraction (saturation-scaled relative to frame area)
  2. Model detection confidence score
  3. Configurable damage category structural weight (D40 Pothole > D20 Alligator > D10 Transverse > D00 Longitudinal)

It never claims physical dimensions (mm/cm) or civil-engineering structural pavement severity.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ml.constants import CODE_TO_SPEC, ID_TO_SPEC, TARGET_CODES


METHODOLOGY_DISCLAIMER = (
    "Visual Priority Scores are heuristic image-space indicators derived from 2D bounding-box occupancy, "
    "detection confidence, and damage category weighting. Without camera calibration or depth sensing, "
    "these estimates do not represent physical crack width/depth or structural engineering severity."
)


@dataclass
class PriorityHeuristicConfig:
    low_max_threshold: float = 35.0
    high_min_threshold: float = 65.0
    area_weight: float = 0.40
    category_weight: float = 0.35
    confidence_weight: float = 0.25
    # Area fraction at which the visual occupancy term saturates to 1.0 (e.g. 12% of image frame)
    reference_saturation_area_pct: float = 12.0
    class_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "D00": 0.55,
            "D10": 0.62,
            "D20": 0.82,
            "D40": 1.00,
        }
    )

    def validate(self) -> None:
        if not (0.0 < self.low_max_threshold < self.high_min_threshold < 100.0):
            raise ValueError(
                f"Thresholds must satisfy 0 < low_max ({self.low_max_threshold}) < high_min ({self.high_min_threshold}) < 100"
            )
        w_sum = self.area_weight + self.category_weight + self.confidence_weight
        if w_sum <= 0:
            raise ValueError("Sum of heuristic component weights must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PriorityHeuristicConfig":
        if not data:
            return cls()
        default = cls()
        cw = dict(default.class_weights)
        if isinstance(data.get("class_weights"), dict):
            for k, v in data["class_weights"].items():
                if k in TARGET_CODES:
                    cw[k] = max(0.0, min(1.0, float(v)))
        cfg = cls(
            low_max_threshold=float(data.get("low_max_threshold", default.low_max_threshold)),
            high_min_threshold=float(data.get("high_min_threshold", default.high_min_threshold)),
            area_weight=float(data.get("area_weight", default.area_weight)),
            category_weight=float(data.get("category_weight", default.category_weight)),
            confidence_weight=float(data.get("confidence_weight", default.confidence_weight)),
            reference_saturation_area_pct=max(1.0, float(data.get("reference_saturation_area_pct", default.reference_saturation_area_pct))),
            class_weights=cw,
        )
        cfg.validate()
        return cfg


@dataclass
class AssessedDetection:
    class_id: int
    class_code: str
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    box_width: float
    box_height: float
    box_area_px: float
    normalized_box_area: float
    image_area_pct: float
    priority_score: float
    priority_band: str
    score_breakdown: Dict[str, float]
    track_id: Optional[int] = None
    frame_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def assess_detection(
    class_id: int,
    confidence: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
    config: Optional[PriorityHeuristicConfig] = None,
    track_id: Optional[int] = None,
    frame_index: Optional[int] = None,
) -> AssessedDetection:
    """Compute bounding-box visual measurements and transparent Estimated Visual Priority Score."""
    cfg = config or PriorityHeuristicConfig()
    cfg.validate()

    img_w = max(1, int(image_width))
    img_h = max(1, int(image_height))
    frame_area = float(img_w * img_h)

    cx1 = max(0.0, min(float(img_w), float(x1)))
    cy1 = max(0.0, min(float(img_h), float(y1)))
    cx2 = max(cx1, min(float(img_w), float(x2)))
    cy2 = max(cy1, min(float(img_h), float(y2)))

    bw = cx2 - cx1
    bh = cy2 - cy1
    box_area = bw * bh
    norm_area = box_area / frame_area
    area_pct = norm_area * 100.0

    spec = ID_TO_SPEC.get(int(class_id))
    if spec is not None:
        code = spec.code
        name = spec.name
    else:
        code = f"CLS_{class_id}"
        name = f"Class {class_id}"

    conf = max(0.0, min(1.0, float(confidence)))

    # Normalize weights cleanly
    w_total = cfg.area_weight + cfg.category_weight + cfg.confidence_weight
    wa = cfg.area_weight / w_total
    wc = cfg.category_weight / w_total
    wp = cfg.confidence_weight / w_total

    # Smooth square-root saturation curve so small-to-medium cracks still differentiate meaningfully
    sat_pct = max(0.5, cfg.reference_saturation_area_pct)
    area_factor = min(1.0, math.sqrt(max(0.0, area_pct) / sat_pct))
    cat_factor = float(cfg.class_weights.get(code, CODE_TO_SPEC["D00"].default_category_weight))
    conf_factor = conf

    raw_score = 100.0 * (wa * area_factor + wc * cat_factor + wp * conf_factor)
    priority_score = round(max(0.0, min(100.0, raw_score)), 2)

    if priority_score >= cfg.high_min_threshold:
        band = "High"
    elif priority_score >= cfg.low_max_threshold:
        band = "Medium"
    else:
        band = "Low"

    breakdown = {
        "area_component": round(100.0 * wa * area_factor, 2),
        "category_component": round(100.0 * wc * cat_factor, 2),
        "confidence_component": round(100.0 * wp * conf_factor, 2),
    }

    return AssessedDetection(
        class_id=int(class_id),
        class_code=code,
        class_name=name,
        confidence=round(conf, 4),
        x1=round(cx1, 2),
        y1=round(cy1, 2),
        x2=round(cx2, 2),
        y2=round(cy2, 2),
        box_width=round(bw, 2),
        box_height=round(bh, 2),
        box_area_px=round(box_area, 2),
        normalized_box_area=round(norm_area, 6),
        image_area_pct=round(area_pct, 3),
        priority_score=priority_score,
        priority_band=band,
        score_breakdown=breakdown,
        track_id=track_id,
        frame_index=frame_index,
    )


def assess_detections_batch(
    raw_detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    config: Optional[PriorityHeuristicConfig] = None,
) -> List[AssessedDetection]:
    """Assess a list of raw detector outputs into sorted AssessedDetection instances."""
    assessed = [
        assess_detection(
            class_id=int(d["class_id"]),
            confidence=float(d["confidence"]),
            x1=float(d["x1"]),
            y1=float(d["y1"]),
            x2=float(d["x2"]),
            y2=float(d["y2"]),
            image_width=image_width,
            image_height=image_height,
            config=config,
            track_id=d.get("track_id"),
            frame_index=d.get("frame_index"),
        )
        for d in raw_detections
    ]
    assessed.sort(key=lambda item: item.priority_score, reverse=True)
    return assessed


def summarize_category_assessments(detections: List[AssessedDetection]) -> Dict[str, Any]:
    """Aggregate detection counts, priority bands, and mean scores across categories."""
    band_counts = Counter(d.priority_band for d in detections)
    cat_counts = Counter(d.class_code for d in detections)
    cat_scores: Dict[str, List[float]] = defaultdict(list)

    for d in detections:
        cat_scores[d.class_code].append(d.priority_score)

    per_category = {}
    for code in TARGET_CODES:
        scores = cat_scores.get(code, [])
        per_category[code] = {
            "count": cat_counts[code],
            "mean_priority_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "max_priority_score": round(max(scores), 2) if scores else 0.0,
        }

    return {
        "total_detections": len(detections),
        "priority_bands": {
            "High": band_counts["High"],
            "Medium": band_counts["Medium"],
            "Low": band_counts["Low"],
        },
        "max_priority_score": round(max((d.priority_score for d in detections), default=0.0), 2),
        "mean_priority_score": round(
            sum(d.priority_score for d in detections) / len(detections), 2
        ) if detections else 0.0,
        "per_category": per_category,
        "methodology_disclaimer": METHODOLOGY_DISCLAIMER,
    }
