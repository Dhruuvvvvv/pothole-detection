"""Pydantic v2 API request and response schemas for RoadGuard AI."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
    database_ok: bool
    model_configured: bool
    model_status: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    timestamp_utc: str


class DetectionItemSchema(BaseModel):
    id: Optional[int] = None
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
    image_area_pct: float
    priority_score: float
    priority_band: str
    track_id: Optional[int] = None
    frame_index: Optional[int] = None
    score_breakdown: Optional[Dict[str, float]] = None


class AnalysisSummaryResponse(BaseModel):
    id: int
    job_uid: str
    filename: str
    analysis_type: str
    status: str
    progress_pct: float
    timestamp: datetime
    processing_duration: float
    total_detections: int
    high_priority_count: int
    medium_priority_count: int
    low_priority_count: int
    input_url: Optional[str] = None
    output_url: Optional[str] = None
    summary: Dict[str, Any] = Field(default_factory=dict)
    detections: List[DetectionItemSchema] = Field(default_factory=list)


class DashboardSummaryResponse(BaseModel):
    total_analyses: int
    total_images_analysed: int
    total_videos_analysed: int
    total_damage_instances: int
    d00_count: int
    d10_count: int
    d20_count: int
    d40_count: int
    high_priority_detections: int
    medium_priority_detections: int
    low_priority_detections: int
    recent_analyses: List[AnalysisSummaryResponse]
    model_info: Dict[str, Any]


class PrioritySettingsPayload(BaseModel):
    low_max_threshold: float = Field(default=35.0, gt=0.0, lt=100.0)
    high_min_threshold: float = Field(default=65.0, gt=0.0, lt=100.0)
    area_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    category_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    confidence_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    reference_saturation_area_pct: float = Field(default=12.0, ge=1.0, le=100.0)
    class_weights: Dict[str, float] = Field(
        default_factory=lambda: {"D00": 0.55, "D10": 0.62, "D20": 0.82, "D40": 1.00}
    )
    default_conf_threshold: float = Field(default=0.25, ge=0.05, le=0.95)
    default_iou_threshold: float = Field(default=0.45, ge=0.10, le=0.95)
    checkpoint_path: Optional[str] = None


class ModelReloadPayload(BaseModel):
    checkpoint_path: Optional[str] = None
