"""Integration tests for Phase G — FastAPI Backend, Upload Validation, Database FKs, and Report Exports."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

import backend.app.api.routes as routes_module
from backend.app.database.session import Base, build_engine, get_db
from backend.app.main import create_app
from backend.app.models.db_models import AnalysisRecord, DetectionRecord
from ml.inference.detector import RoadDamageDetector


def _encode_png_bytes(width: int = 240, height: int = 180) -> bytes:
    img = np.full((height, width, 3), 120, dtype=np.uint8)
    cv2.circle(img, (120, 90), 25, (30, 30, 35), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class _ConfiguredTestDetector(RoadDamageDetector):
    def __init__(self) -> None:
        self.checkpoint_path = Path("verified_test_rdd2022.pt")
        self.requested_device = "cpu"
        self._model = True
        self._model_class_map = {0: 0, 1: 1, 2: 2, 3: 3}
        self._raw_model_names = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
        self._load_error = None

    def predict_raw_frame(
        self,
        img_bgr: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
    ) -> Tuple[List[Dict[str, Any]], float]:
        return (
            [
                {"class_id": 3, "confidence": 0.88, "x1": 60.0, "y1": 50.0, "x2": 170.0, "y2": 140.0},
                {"class_id": 0, "confidence": 0.61, "x1": 15.0, "y1": 20.0, "x2": 35.0, "y2": 160.0},
            ],
            5.2,
        )


@pytest.fixture
def client_and_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Tuple[TestClient, sessionmaker], None, None]:
    db_file = tmp_path / "test_roadguard.db"
    test_engine = build_engine(f"sqlite:///{db_file.as_posix()}")
    Base.metadata.create_all(bind=test_engine)
    TestSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, expire_on_commit=False)

    out_dir = tmp_path / "outputs"
    rep_dir = tmp_path / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(routes_module.settings, "outputs_dir", str(out_dir))
    monkeypatch.setattr(routes_module.settings, "reports_dir", str(rep_dir))
    monkeypatch.setattr(routes_module, "SessionLocal", TestSessionLocal)

    app = create_app()

    def _override_get_db() -> Generator[Session, None, None]:
        s = TestSessionLocal()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _override_get_db

    with TestClient(app) as tc:
        yield tc, TestSessionLocal


def test_health_empty_dashboard_and_unconfigured_model(
    client_and_session: Tuple[TestClient, sessionmaker],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = client_and_session
    unconfigured = RoadDamageDetector(checkpoint_path=tmp_path / "missing.pt")
    monkeypatch.setattr(routes_module, "get_detector_singleton", lambda *_a, **_kw: unconfigured)

    h = client.get("/api/health")
    assert h.status_code == 200
    assert h.json()["database_ok"] is True
    assert h.json()["model_configured"] is False

    dash = client.get("/api/dashboard/summary")
    assert dash.status_code == 200
    body = dash.json()
    assert body["total_analyses"] == 0
    assert body["total_damage_instances"] == 0
    assert body["recent_analyses"] == []

    # Analyzing an image before training must return 503 MODEL_NOT_CONFIGURED (not fake results)
    png_bytes = _encode_png_bytes()
    res = client.post(
        "/api/analyze/image",
        files={"file": ("road.png", png_bytes, "image/png")},
    )
    assert res.status_code == 503
    assert res.json()["detail"]["code"] == "MODEL_NOT_CONFIGURED"


def test_classical_image_processing_and_upload_validation(
    client_and_session: Tuple[TestClient, sessionmaker],
) -> None:
    client, _ = client_and_session

    cat = client.get("/api/process/catalog")
    assert cat.status_code == 200
    assert len(cat.json()["operations"]) >= 20

    # Valid CLAHE operation
    png_bytes = _encode_png_bytes()
    proc = client.post(
        "/api/process/image",
        files={"file": ("crack.png", png_bytes, "image/png")},
        data={"operation": "clahe", "parameters_json": json.dumps({"clip_limit": 2.5, "tile_grid_size": 8})},
    )
    assert proc.status_code == 200
    pdata = proc.json()
    assert pdata["processed_url"].startswith("/static/")
    assert pdata["processed_metrics"]["ssim_score"] is not None

    # Corrupt bytes masquerading as .jpg must return HTTP 400
    bad = client.post(
        "/api/process/image",
        files={"file": ("corrupt.jpg", b"not_an_image_payload", "image/jpeg")},
        data={"operation": "clahe"},
    )
    assert bad.status_code == 400

    # Unsupported extension must return HTTP 400
    bad_ext = client.post(
        "/api/process/image",
        files={"file": ("malware.exe", png_bytes, "application/octet-stream")},
        data={"operation": "clahe"},
    )
    assert bad_ext.status_code == 400


def test_full_inference_persistence_reports_exports_and_cascade_delete(
    client_and_session: Tuple[TestClient, sessionmaker],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, TestSessionLocal = client_and_session
    detector = _ConfiguredTestDetector()
    monkeypatch.setattr(routes_module, "get_detector_singleton", lambda *_a, **_kw: detector)

    # 1. Image analysis
    png_bytes = _encode_png_bytes()
    resp = client.post(
        "/api/analyze/image",
        files={"file": ("highway_segment.png", png_bytes, "image/png")},
        data={"conf_threshold": "0.25", "iou_threshold": "0.45"},
    )
    assert resp.status_code == 200
    analysis = resp.json()
    analysis_id = analysis["id"]
    assert analysis["total_detections"] == 2
    assert len(analysis["detections"]) == 2

    # 2. Dashboard reflects actual stored data
    dash = client.get("/api/dashboard/summary").json()
    assert dash["total_images_analysed"] == 1
    assert dash["total_damage_instances"] == 2
    assert dash["d40_count"] == 1
    assert dash["d00_count"] == 1

    # 3. Export CSV & PDF
    csv_res = client.get(f"/api/reports/{analysis_id}/export/csv")
    assert csv_res.status_code == 200
    assert "D40" in csv_res.text and "D00" in csv_res.text

    pdf_res = client.get(f"/api/reports/{analysis_id}/export/pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.content.startswith(b"%PDF-")

    # 4. Synchronous video analysis endpoint test
    vid_file = tmp_path / "clip.avi"
    vw = cv2.VideoWriter(str(vid_file), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (240, 180))
    for _ in range(4):
        vw.write(np.full((180, 240, 3), 100, dtype=np.uint8))
    vw.release()

    vid_resp = client.post(
        "/api/analyze/video?wait=true",
        files={"file": ("clip.avi", vid_file.read_bytes(), "video/x-msvideo")},
        data={"tracker_type": "bytetrack", "conf_threshold": "0.25"},
    )
    assert vid_resp.status_code == 200
    assert vid_resp.json()["status"] == "completed"
    assert vid_resp.json()["total_detections"] == 8

    # 5. Cascade delete removes child DetectionRecords via SQLite FK
    del_res = client.delete(f"/api/reports/{analysis_id}")
    assert del_res.status_code == 200
    with TestSessionLocal() as db:
        rem_dets = db.scalar(select(func.count(DetectionRecord.id)).where(DetectionRecord.analysis_id == analysis_id))
        assert rem_dets == 0


def test_settings_validation_and_evaluation_import(
    client_and_session: Tuple[TestClient, sessionmaker],
) -> None:
    client, _ = client_and_session

    # Invalid threshold order (low_max > high_min) must be rejected with 422
    bad_s = client.put(
        "/api/settings",
        json={"low_max_threshold": 75.0, "high_min_threshold": 40.0},
    )
    assert bad_s.status_code == 422

    ok_s = client.put(
        "/api/settings",
        json={"low_max_threshold": 30.0, "high_min_threshold": 70.0},
    )
    assert ok_s.status_code == 200
    assert ok_s.json()["low_max_threshold"] == 30.0

    eval_doc = {
        "precision": 0.712,
        "recall": 0.648,
        "f1_score": 0.6785,
        "map50": 0.691,
        "map50_95": 0.412,
        "inference_ms": 6.4,
        "fps": 156.2,
        "per_class": [],
    }
    imp = client.post(
        "/api/model/import-evaluation",
        files={"file": ("evaluation_summary.json", json.dumps(eval_doc).encode("utf-8"), "application/json")},
        data={"experiment_name": "pytest_imported_eval", "model_name": "yolo26n", "epochs": "50"},
    )
    assert imp.status_code == 200
    minfo = client.get("/api/model/info").json()
    assert len(minfo["experiments"]) == 1
    assert minfo["experiments"][0]["map50"] == pytest.approx(0.691)

    csv_content = (
        "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
        "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),"
        "val/box_loss,val/cls_loss,val/dfl_loss\n"
        "1,10.0,2.1,2.9,1.8,0.35,0.28,0.22,0.09,2.2,2.8,1.7\n"
        "2,20.0,1.9,2.4,1.6,0.45,0.36,0.31,0.14,2.0,2.4,1.6\n"
    )
    csv_imp = client.post(
        "/api/model/import-training-csv",
        files={"file": ("results.csv", csv_content.encode("utf-8"), "text/csv")},
        data={"experiment_name": "pytest_csv_run"},
    )
    assert csv_imp.status_code == 200
    assert csv_imp.json()["epochs"] == 2
    assert csv_imp.json()["results_csv_url"].endswith("results.csv")
    assert csv_imp.json()["results_png_url"].endswith("results.png")
