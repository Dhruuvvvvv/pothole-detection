"""Persistent runtime settings service backed by the database."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.models.db_models import AppSettingRecord
from backend.app.schemas.api_schemas import PrioritySettingsPayload
from ml.assessment.severity_engine import PriorityHeuristicConfig


SETTINGS_KEY = "global_runtime_settings"


def get_runtime_settings(db: Session) -> Dict[str, Any]:
    row = db.get(AppSettingRecord, SETTINGS_KEY)
    default_payload = PrioritySettingsPayload(
        low_max_threshold=settings.priority_low_max,
        high_min_threshold=settings.priority_high_min,
        default_conf_threshold=settings.default_conf_threshold,
        default_iou_threshold=settings.default_iou_threshold,
        checkpoint_path=settings.model_checkpoint_path,
    ).model_dump()

    if row is None:
        return default_payload

    try:
        stored = json.loads(row.value_json)
        merged = {**default_payload, **stored}
        return merged
    except Exception:
        return default_payload


def update_runtime_settings(db: Session, payload: PrioritySettingsPayload) -> Dict[str, Any]:
    # Validate priority heuristic thresholds & weights before persisting
    cfg = PriorityHeuristicConfig.from_dict(payload.model_dump())
    data = payload.model_dump()
    data["class_weights"] = cfg.class_weights
    if not data.get("checkpoint_path"):
        data["checkpoint_path"] = settings.model_checkpoint_path

    row = db.get(AppSettingRecord, SETTINGS_KEY)
    serialized = json.dumps(data)
    if row is None:
        row = AppSettingRecord(
            key=SETTINGS_KEY,
            value_json=serialized,
            updated_at=datetime.now(timezone.utc),
        )
        db.add(row)
    else:
        row.value_json = serialized
        row.updated_at = datetime.now(timezone.utc)
    db.commit()
    return data


def build_priority_config_from_db(db: Session) -> PriorityHeuristicConfig:
    rt = get_runtime_settings(db)
    return PriorityHeuristicConfig.from_dict(rt)
