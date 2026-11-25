"""SQLAlchemy ORM schema for analyses, detections, model_experiments, and app_settings."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.session import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_uid: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    analysis_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)  # 'image' | 'video'
    status: Mapped[str] = mapped_column(String(32), index=True, default="completed", nullable=False)
    progress_pct: Mapped[float] = mapped_column(Float, default=100.0, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True, nullable=False)
    processing_duration: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    total_detections: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    high_priority_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    medium_priority_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    low_priority_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    input_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    output_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    summary_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    detections: Mapped[List["DetectionRecord"]] = relationship(
        "DetectionRecord",
        back_populates="analysis",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class DetectionRecord(Base):
    __tablename__ = "detections"
    __table_args__ = (
        Index("ix_detections_analysis_class", "analysis_id", "class_code"),
        Index("ix_detections_priority_band", "priority_band"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("analyses.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    frame_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    class_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    class_code: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)
    box_width: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    box_height: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    box_area: Mapped[float] = mapped_column(Float, nullable=False)
    image_area_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    priority_score: Mapped[float] = mapped_column(Float, index=True, nullable=False)
    priority_band: Mapped[str] = mapped_column(String(16), nullable=False)

    analysis: Mapped["AnalysisRecord"] = relationship("AnalysisRecord", back_populates="detections")


class ModelExperimentRecord(Base):
    __tablename__ = "model_experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_name: Mapped[str] = mapped_column(String(128), default="rdd2022_run", nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(64), default="RDD2022-4Class", nullable=False)
    epochs: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, default=16, nullable=False)
    image_size: Mapped[int] = mapped_column(Integer, default=640, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    f1_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    map50: Mapped[float] = mapped_column(Float, nullable=False)
    map50_95: Mapped[float] = mapped_column(Float, nullable=False)
    inference_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    fps: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    per_class_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    artifacts_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    training_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True, nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(String(512), nullable=False)


class AppSettingRecord(Base):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
