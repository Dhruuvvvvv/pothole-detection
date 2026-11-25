"""Centralized application configuration loaded from environment variables or `.env`."""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "RoadGuard AI"
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173"

    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'outputs' / 'roadguard.db').as_posix()}"

    model_checkpoint_path: str = str(PROJECT_ROOT / "models" / "best.pt")
    fallback_architecture: str = "yolo26n.pt"
    default_conf_threshold: float = 0.25
    default_iou_threshold: float = 0.45
    default_image_size: int = 640
    inference_device: str = "auto"

    max_image_upload_mb: int = 25
    max_video_upload_mb: int = 250
    temp_retention_hours: int = 24

    kaggle_dataset_slug: str = "aliabdelmenam/rdd-2022"
    dataset_output_dir: str = str(PROJECT_ROOT / "outputs" / "dataset")
    outputs_dir: str = str(PROJECT_ROOT / "outputs")
    reports_dir: str = str(PROJECT_ROOT / "reports")

    priority_low_max: float = 35.0
    priority_high_min: float = 65.0

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
