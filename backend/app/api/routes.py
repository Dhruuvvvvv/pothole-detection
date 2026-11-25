"""Modular FastAPI API routers for RoadGuard AI (Module 7)."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import Response
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.database.session import SessionLocal, get_db
from backend.app.models.db_models import (
    AnalysisRecord,
    DetectionRecord,
    ModelExperimentRecord,
)
from backend.app.schemas.api_schemas import (
    AnalysisSummaryResponse,
    DashboardSummaryResponse,
    DetectionItemSchema,
    HealthResponse,
    ModelReloadPayload,
    PrioritySettingsPayload,
)
from backend.app.services.dataset_service import get_dataset_explorer_summary
from backend.app.services.settings_service import (
    build_priority_config_from_db,
    get_runtime_settings,
    update_runtime_settings,
)
from backend.app.utils.exporters import (
    generate_analysis_csv_bytes,
    generate_analysis_pdf_bytes,
)
from backend.app.utils.file_validation import (
    path_to_static_url,
    read_and_validate_image_upload,
    save_and_validate_video_upload,
)
from ml.constants import TARGET_CLASS_SPECS, TARGET_CODES
from ml.experiments.ip_vs_detection import COMMON_ERROR_SOURCES, compute_yolo_eigencam
from ml.inference.detector import ModelNotConfiguredError, get_detector_singleton
from ml.preprocessing.classical_ops import (
    apply_classical_operation,
    compare_filtering_techniques,
    get_operation_catalog,
)
from ml.tracking.video_processor import process_road_video

router = APIRouter(prefix="/api")


def _serialize_analysis(
    rec: AnalysisRecord,
    include_detections: bool = False,
) -> AnalysisSummaryResponse:
    summary_dict: Dict[str, Any] = {}
    if rec.summary_json:
        try:
            summary_dict = json.loads(rec.summary_json)
        except Exception:
            summary_dict = {}

    det_items: List[DetectionItemSchema] = []
    if include_detections:
        for d in rec.detections:
            det_items.append(
                DetectionItemSchema(
                    id=d.id,
                    class_id=d.class_id,
                    class_code=d.class_code,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    x1=d.x1,
                    y1=d.y1,
                    x2=d.x2,
                    y2=d.y2,
                    box_width=d.box_width,
                    box_height=d.box_height,
                    box_area_px=d.box_area,
                    image_area_pct=d.image_area_pct,
                    priority_score=d.priority_score,
                    priority_band=d.priority_band,
                    track_id=d.track_id,
                    frame_index=d.frame_index,
                )
            )

    return AnalysisSummaryResponse(
        id=rec.id,
        job_uid=rec.job_uid,
        filename=rec.filename,
        analysis_type=rec.analysis_type,
        status=rec.status,
        progress_pct=rec.progress_pct,
        timestamp=rec.timestamp,
        processing_duration=rec.processing_duration,
        total_detections=rec.total_detections,
        high_priority_count=rec.high_priority_count,
        medium_priority_count=rec.medium_priority_count,
        low_priority_count=rec.low_priority_count,
        input_url=path_to_static_url(rec.input_path),
        output_url=path_to_static_url(rec.output_path),
        summary=summary_dict,
        detections=det_items,
    )


# ---------------------------------------------------------------------------
# 1. Health & Diagnostics
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse, tags=["System"])
def get_health(db: Session = Depends(get_db)) -> HealthResponse:
    db_ok = True
    try:
        db.execute(select(func.count(AnalysisRecord.id)))
    except Exception:
        db_ok = False

    detector = get_detector_singleton(settings.model_checkpoint_path, settings.inference_device)
    st = detector.get_status()

    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        app_name=settings.app_name,
        environment=settings.app_env,
        database_ok=db_ok,
        model_configured=bool(st["configured"]),
        model_status=str(st["status"]),
        cuda_available=bool(st["cuda_available"]),
        gpu_name=st.get("gpu_name"),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# 2. Image Analysis & Explainability
# ---------------------------------------------------------------------------
@router.post("/analyze/image", response_model=AnalysisSummaryResponse, tags=["Inference"])
async def analyze_image_endpoint(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    image_size: int = Form(640),
    include_eigencam: bool = Form(False),
    db: Session = Depends(get_db),
) -> AnalysisSummaryResponse:
    if not (0.05 <= conf_threshold <= 0.95):
        raise HTTPException(status_code=422, detail="conf_threshold must be between 0.05 and 0.95")
    if not (0.10 <= iou_threshold <= 0.95):
        raise HTTPException(status_code=422, detail="iou_threshold must be between 0.10 and 0.95")

    bgr, orig_name, _ = await read_and_validate_image_upload(file)
    detector = get_detector_singleton(settings.model_checkpoint_path, settings.inference_device)
    if not detector.is_ready:
        st = detector.get_status()
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_NOT_CONFIGURED",
                "message": st["message"],
                "checkpoint_path": st["checkpoint_path"],
            },
        )

    priority_cfg = build_priority_config_from_db(db)
    t0 = time.perf_counter()
    try:
        annotated_bgr, assessed, summary = detector.predict_image(
            img_bgr=bgr,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
            priority_config=priority_cfg,
        )
    except ModelNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail={"code": "MODEL_NOT_CONFIGURED", "message": str(exc)}) from exc

    job_uid = uuid.uuid4().hex[:12]
    uploads_dir = Path(settings.outputs_dir) / "uploads"
    annotated_dir = Path(settings.outputs_dir) / "annotated"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    in_path = uploads_dir / f"{job_uid}_{orig_name}"
    out_path = annotated_dir / f"{job_uid}_annotated.png"
    cv2.imwrite(str(in_path), bgr)
    cv2.imwrite(str(out_path), annotated_bgr)

    if include_eigencam:
        try:
            cam_bgr, cam_note = compute_yolo_eigencam(detector, bgr, image_size=image_size)
            cam_path = annotated_dir / f"{job_uid}_eigencam.jpg"
            cv2.imwrite(str(cam_path), cam_bgr)
            summary["eigencam_url"] = path_to_static_url(cam_path)
            summary["eigencam_note"] = cam_note
        except Exception as cam_err:
            logger.warning("EigenCAM computation skipped: %s", cam_err)

    duration_s = round(time.perf_counter() - t0, 3)

    analysis_rec = AnalysisRecord(
        job_uid=job_uid,
        filename=orig_name,
        analysis_type="image",
        status="completed",
        progress_pct=100.0,
        processing_duration=duration_s,
        total_detections=len(assessed),
        high_priority_count=summary["priority_bands"]["High"],
        medium_priority_count=summary["priority_bands"]["Medium"],
        low_priority_count=summary["priority_bands"]["Low"],
        input_path=str(in_path),
        output_path=str(out_path),
        summary_json=json.dumps(summary),
    )
    db.add(analysis_rec)
    db.flush()

    for d in assessed:
        db.add(
            DetectionRecord(
                analysis_id=analysis_rec.id,
                class_id=d.class_id,
                class_code=d.class_code,
                class_name=d.class_name,
                confidence=d.confidence,
                x1=d.x1,
                y1=d.y1,
                x2=d.x2,
                y2=d.y2,
                box_width=d.box_width,
                box_height=d.box_height,
                box_area=d.box_area_px,
                image_area_pct=d.image_area_pct,
                priority_score=d.priority_score,
                priority_band=d.priority_band,
            )
        )

    db.commit()
    db.refresh(analysis_rec)
    resp = _serialize_analysis(analysis_rec, include_detections=True)
    # Enrich score_breakdown on immediate response
    for resp_det, assessed_det in zip(resp.detections, assessed):
        resp_det.score_breakdown = assessed_det.score_breakdown
    return resp


# ---------------------------------------------------------------------------
# 3. Video Analytics (Background Job + Synchronous Option)
# ---------------------------------------------------------------------------
def _execute_video_job(
    analysis_id: int,
    in_video_path: str,
    out_video_path: str,
    tracker_type: str,
    conf_threshold: float,
    iou_threshold: float,
    image_size: int,
    frame_stride: int,
) -> None:
    db = SessionLocal()
    try:
        rec = db.get(AnalysisRecord, analysis_id)
        if rec is None:
            return
        rec.status = "processing"
        rec.progress_pct = 2.0
        db.commit()

        detector = get_detector_singleton(settings.model_checkpoint_path, settings.inference_device)
        priority_cfg = build_priority_config_from_db(db)

        last_reported = [0.0]

        def _on_progress(done_frames: int, est_total: int) -> None:
            pct = min(98.0, round(100.0 * done_frames / max(1, est_total), 1))
            if pct - last_reported[0] >= 5.0:
                last_reported[0] = pct
                r_inner = db.get(AnalysisRecord, analysis_id)
                if r_inner is not None:
                    r_inner.progress_pct = pct
                    db.commit()

        res = process_road_video(
            video_path=in_video_path,
            output_video_path=out_video_path,
            detector=detector,
            tracker_type=tracker_type,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
            frame_stride=frame_stride,
            priority_config=priority_cfg,
            progress_callback=_on_progress,
        )

        rec = db.get(AnalysisRecord, analysis_id)
        if rec is None:
            return

        summary_payload = res.to_dict()
        sampled = summary_payload.pop("sampled_detections", [])

        rec.status = "completed"
        rec.progress_pct = 100.0
        rec.processing_duration = res.processing_duration_sec
        rec.total_detections = res.total_frame_level_detections
        rec.high_priority_count = res.priority_band_distribution.get("High", 0)
        rec.medium_priority_count = res.priority_band_distribution.get("Medium", 0)
        rec.low_priority_count = res.priority_band_distribution.get("Low", 0)
        rec.output_path = res.output_video_path
        rec.summary_json = json.dumps(summary_payload)

        for d in sampled:
            db.add(
                DetectionRecord(
                    analysis_id=rec.id,
                    frame_index=d.get("frame_index"),
                    track_id=d.get("track_id"),
                    class_id=int(d["class_id"]),
                    class_code=str(d["class_code"]),
                    class_name=str(d["class_name"]),
                    confidence=float(d["confidence"]),
                    x1=float(d["x1"]),
                    y1=float(d["y1"]),
                    x2=float(d["x2"]),
                    y2=float(d["y2"]),
                    box_width=float(d["box_width"]),
                    box_height=float(d["box_height"]),
                    box_area=float(d["box_area_px"]),
                    image_area_pct=float(d["image_area_pct"]),
                    priority_score=float(d["priority_score"]),
                    priority_band=str(d["priority_band"]),
                )
            )
        db.commit()
    except Exception as exc:
        logger.exception("Video processing job #%s failed: %s", analysis_id, exc)
        rec = db.get(AnalysisRecord, analysis_id)
        if rec is not None:
            rec.status = "failed"
            rec.summary_json = json.dumps({"error": str(exc)})
            db.commit()
    finally:
        db.close()


@router.post("/analyze/video", response_model=AnalysisSummaryResponse, tags=["Inference"])
async def analyze_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tracker_type: str = Form("bytetrack"),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    image_size: int = Form(640),
    frame_stride: int = Form(1),
    wait: bool = Query(False, description="Wait synchronously for completion (useful for short clips or CLI/tests)"),
    db: Session = Depends(get_db),
) -> AnalysisSummaryResponse:
    if tracker_type.lower() not in {"bytetrack", "sort"}:
        raise HTTPException(status_code=422, detail="tracker_type must be 'bytetrack' or 'sort'")

    detector = get_detector_singleton(settings.model_checkpoint_path, settings.inference_device)
    if not detector.is_ready:
        st = detector.get_status()
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_NOT_CONFIGURED",
                "message": st["message"],
                "checkpoint_path": st["checkpoint_path"],
            },
        )

    uploads_dir = Path(settings.outputs_dir) / "uploads"
    videos_dir = Path(settings.outputs_dir) / "videos"
    saved_in, orig_name = await save_and_validate_video_upload(file, uploads_dir)

    job_uid = uuid.uuid4().hex[:12]
    out_video = videos_dir / f"{job_uid}_tracked.mp4"

    rec = AnalysisRecord(
        job_uid=job_uid,
        filename=orig_name,
        analysis_type="video",
        status="queued",
        progress_pct=0.0,
        input_path=str(saved_in),
        summary_json=json.dumps({"tracker_used": tracker_type.lower()}),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    if wait:
        _execute_video_job(
            analysis_id=rec.id,
            in_video_path=str(saved_in),
            out_video_path=str(out_video),
            tracker_type=tracker_type,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
            frame_stride=max(1, frame_stride),
        )
        db.expire_all()
        updated = db.get(AnalysisRecord, rec.id)
        return _serialize_analysis(updated or rec, include_detections=True)

    background_tasks.add_task(
        _execute_video_job,
        rec.id,
        str(saved_in),
        str(out_video),
        tracker_type,
        conf_threshold,
        iou_threshold,
        image_size,
        max(1, frame_stride),
    )
    return _serialize_analysis(rec, include_detections=False)


@router.get("/analyze/video/{analysis_id}/status", response_model=AnalysisSummaryResponse, tags=["Inference"])
def get_video_job_status(analysis_id: int, db: Session = Depends(get_db)) -> AnalysisSummaryResponse:
    rec = db.get(AnalysisRecord, analysis_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Analysis #{analysis_id} not found")
    return _serialize_analysis(rec, include_detections=(rec.status == "completed"))


# ---------------------------------------------------------------------------
# 4. Classical Image Processing Laboratory (Module 1)
# ---------------------------------------------------------------------------
@router.get("/process/catalog", tags=["Image Processing Lab"])
def get_classical_ip_catalog() -> Dict[str, Any]:
    return {"operations": get_operation_catalog()}


@router.post("/process/image", tags=["Image Processing Lab"])
async def process_classical_image_endpoint(
    file: UploadFile = File(...),
    operation: str = Form(...),
    parameters_json: str = Form("{}"),
) -> Dict[str, Any]:
    bgr, orig_name, _ = await read_and_validate_image_upload(file)
    try:
        params = json.loads(parameters_json or "{}")
        if not isinstance(params, dict):
            raise ValueError("parameters_json must decode to an object")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid parameters_json: {exc}") from exc

    try:
        proc_bgr, diff_bgr, telemetry = apply_classical_operation(bgr, operation=operation, params=params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    uid = uuid.uuid4().hex[:10]
    proc_dir = Path(settings.outputs_dir) / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    orig_out = proc_dir / f"{uid}_orig_{Path(orig_name).stem}.png"
    proc_out = proc_dir / f"{uid}_{operation}.png"
    diff_out = proc_dir / f"{uid}_{operation}_diff.png"

    cv2.imwrite(str(orig_out), bgr)
    cv2.imwrite(str(proc_out), proc_bgr)
    cv2.imwrite(str(diff_out), diff_bgr)

    return {
        "filename": orig_name,
        "original_url": path_to_static_url(orig_out),
        "processed_url": path_to_static_url(proc_out),
        "difference_url": path_to_static_url(diff_out),
        **telemetry,
    }


@router.post("/process/compare-filters", tags=["Image Processing Lab"])
async def compare_filters_endpoint(
    file: UploadFile = File(...),
    kernel_size: int = Form(5),
) -> Dict[str, Any]:
    bgr, orig_name, _ = await read_and_validate_image_upload(file)
    results = compare_filtering_techniques(bgr, kernel_size=kernel_size)

    uid = uuid.uuid4().hex[:10]
    proc_dir = Path(settings.outputs_dir) / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    orig_out = proc_dir / f"{uid}_cmp_orig.png"
    cv2.imwrite(str(orig_out), bgr)

    serialized = []
    for item in results:
        op_name = item["operation"]
        p_path = proc_dir / f"{uid}_cmp_{op_name}.png"
        d_path = proc_dir / f"{uid}_cmp_{op_name}_diff.png"
        cv2.imwrite(str(p_path), item["processed_bgr"])
        cv2.imwrite(str(d_path), item["diff_bgr"])
        serialized.append({
            "operation": op_name,
            "label": item["label"],
            "processed_url": path_to_static_url(p_path),
            "difference_url": path_to_static_url(d_path),
            "processing_time_ms": item["processing_time_ms"],
            "metrics": item["metrics"],
        })

    return {
        "filename": orig_name,
        "kernel_size": kernel_size,
        "original_url": path_to_static_url(orig_out),
        "comparisons": serialized,
    }


# ---------------------------------------------------------------------------
# 5. Dashboard & Analytics (Actual Stored Database Telemetry Only)
# ---------------------------------------------------------------------------
@router.get("/dashboard/summary", response_model=DashboardSummaryResponse, tags=["Analytics"])
def get_dashboard_summary(db: Session = Depends(get_db)) -> DashboardSummaryResponse:
    completed = db.scalars(
        select(AnalysisRecord).where(AnalysisRecord.status == "completed")
    ).all()

    total_images = sum(1 for r in completed if r.analysis_type == "image")
    total_videos = sum(1 for r in completed if r.analysis_type == "video")
    total_dets = sum(r.total_detections for r in completed)
    high_cnt = sum(r.high_priority_count for r in completed)
    med_cnt = sum(r.medium_priority_count for r in completed)
    low_cnt = sum(r.low_priority_count for r in completed)

    # Aggregate class counts from DetectionRecord (for images) + video summary_json (for full video counts)
    cls_counts = {code: 0 for code in TARGET_CODES}
    img_det_rows = db.execute(
        select(DetectionRecord.class_code, func.count(DetectionRecord.id))
        .join(AnalysisRecord, DetectionRecord.analysis_id == AnalysisRecord.id)
        .where(AnalysisRecord.analysis_type == "image", AnalysisRecord.status == "completed")
        .group_by(DetectionRecord.class_code)
    ).all()
    for code, cnt in img_det_rows:
        if code in cls_counts:
            cls_counts[code] += int(cnt)

    for r in completed:
        if r.analysis_type == "video" and r.summary_json:
            try:
                s_obj = json.loads(r.summary_json)
                dist = s_obj.get("class_frame_detection_distribution", {})
                for code in TARGET_CODES:
                    cls_counts[code] += int(dist.get(code, 0))
            except Exception:
                continue

    recent_rows = db.scalars(
        select(AnalysisRecord).order_by(desc(AnalysisRecord.timestamp)).limit(8)
    ).all()

    detector = get_detector_singleton(settings.model_checkpoint_path, settings.inference_device)

    return DashboardSummaryResponse(
        total_analyses=len(completed),
        total_images_analysed=total_images,
        total_videos_analysed=total_videos,
        total_damage_instances=total_dets,
        d00_count=cls_counts["D00"],
        d10_count=cls_counts["D10"],
        d20_count=cls_counts["D20"],
        d40_count=cls_counts["D40"],
        high_priority_detections=high_cnt,
        medium_priority_detections=med_cnt,
        low_priority_detections=low_cnt,
        recent_analyses=[_serialize_analysis(r, include_detections=False) for r in recent_rows],
        model_info=detector.get_status(),
    )


@router.get("/analytics/distribution", tags=["Analytics"])
def get_analytics_distribution(db: Session = Depends(get_db)) -> Dict[str, Any]:
    dash = get_dashboard_summary(db)
    counts = {
        "D00": dash.d00_count,
        "D10": dash.d10_count,
        "D20": dash.d20_count,
        "D40": dash.d40_count,
    }
    categories = [
        {
            "code": spec.code,
            "name": spec.name,
            "count": counts[spec.code],
            "color": spec.hex_color,
        }
        for spec in TARGET_CLASS_SPECS
    ]
    priorities = [
        {"band": "High", "count": dash.high_priority_detections, "color": "#EF4444"},
        {"band": "Medium", "count": dash.medium_priority_detections, "color": "#F59E0B"},
        {"band": "Low", "count": dash.low_priority_detections, "color": "#10B981"},
    ]
    return {"categories": categories, "priorities": priorities}


@router.get("/analytics/history", tags=["Analytics"])
def get_analytics_history(
    limit: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    rows = db.scalars(
        select(AnalysisRecord)
        .where(AnalysisRecord.status == "completed")
        .order_by(AnalysisRecord.timestamp.asc())
    ).all()

    by_day: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        day = r.timestamp.strftime("%Y-%m-%d") if r.timestamp else "Unknown"
        bucket = by_day.setdefault(
            day,
            {"date": day, "analyses": 0, "detections": 0, "high_priority": 0},
        )
        bucket["analyses"] += 1
        bucket["detections"] += r.total_detections
        bucket["high_priority"] += r.high_priority_count

    series = list(by_day.values())[-limit:]
    return {"timeline": series}


# ---------------------------------------------------------------------------
# 6. Damage Reports & Exports (CSV & PDF)
# ---------------------------------------------------------------------------
@router.get("/reports", tags=["Reports"])
def list_reports(
    search: Optional[str] = Query(None),
    class_code: Optional[str] = Query(None),
    priority_band: Optional[str] = Query(None),
    analysis_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    stmt = select(AnalysisRecord).order_by(desc(AnalysisRecord.timestamp))

    if search:
        stmt = stmt.where(AnalysisRecord.filename.ilike(f"%{search.strip()}%"))
    if analysis_type in {"image", "video"}:
        stmt = stmt.where(AnalysisRecord.analysis_type == analysis_type)
    if priority_band == "High":
        stmt = stmt.where(AnalysisRecord.high_priority_count > 0)
    elif priority_band == "Medium":
        stmt = stmt.where(AnalysisRecord.medium_priority_count > 0)
    elif priority_band == "Low":
        stmt = stmt.where(AnalysisRecord.low_priority_count > 0)

    if class_code in TARGET_CODES:
        subq = select(DetectionRecord.analysis_id).where(DetectionRecord.class_code == class_code)
        stmt = stmt.where(AnalysisRecord.id.in_(subq))

    all_matching = db.scalars(stmt).all()
    total = len(all_matching)
    sliced = all_matching[offset : offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [_serialize_analysis(r, include_detections=False) for r in sliced],
    }


@router.get("/reports/{analysis_id}", response_model=AnalysisSummaryResponse, tags=["Reports"])
def get_report_detail(analysis_id: int, db: Session = Depends(get_db)) -> AnalysisSummaryResponse:
    rec = db.get(AnalysisRecord, analysis_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Report #{analysis_id} not found")
    return _serialize_analysis(rec, include_detections=True)


@router.get("/reports/{analysis_id}/export/csv", tags=["Reports"])
def export_report_csv(analysis_id: int, db: Session = Depends(get_db)) -> Response:
    rec = db.get(AnalysisRecord, analysis_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Report #{analysis_id} not found")
    csv_bytes = generate_analysis_csv_bytes(rec, rec.detections)
    fname = f"roadguard_report_{rec.id}_{Path(rec.filename).stem}.csv"
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/reports/{analysis_id}/export/pdf", tags=["Reports"])
def export_report_pdf(analysis_id: int, db: Session = Depends(get_db)) -> Response:
    rec = db.get(AnalysisRecord, analysis_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Report #{analysis_id} not found")
    pdf_bytes = generate_analysis_pdf_bytes(rec, rec.detections)
    fname = f"roadguard_report_{rec.id}_{Path(rec.filename).stem}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.delete("/reports/{analysis_id}", tags=["Reports"])
def delete_report(analysis_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    rec = db.get(AnalysisRecord, analysis_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Report #{analysis_id} not found")
    for fp in (rec.input_path, rec.output_path):
        if fp:
            Path(fp).unlink(missing_ok=True)
    db.delete(rec)
    db.commit()
    return {"deleted": True, "id": analysis_id}


# ---------------------------------------------------------------------------
# 7. Model Info, Performance Experiments & Hot-Reload
# ---------------------------------------------------------------------------
@router.get("/model/info", tags=["Model & Experiments"])
def get_model_info(db: Session = Depends(get_db)) -> Dict[str, Any]:
    detector = get_detector_singleton(settings.model_checkpoint_path, settings.inference_device)
    exp_rows = db.scalars(
        select(ModelExperimentRecord).order_by(desc(ModelExperimentRecord.training_date))
    ).all()

    experiments = []
    for e in exp_rows:
        experiments.append({
            "id": e.id,
            "experiment_name": e.experiment_name,
            "model_name": e.model_name,
            "dataset_version": e.dataset_version,
            "epochs": e.epochs,
            "batch_size": e.batch_size,
            "image_size": e.image_size,
            "precision": e.precision,
            "recall": e.recall,
            "f1_score": e.f1_score,
            "map50": e.map50,
            "map50_95": e.map50_95,
            "inference_ms": e.inference_ms,
            "fps": e.fps,
            "per_class": json.loads(e.per_class_json) if e.per_class_json else [],
            "artifacts": json.loads(e.artifacts_json) if e.artifacts_json else {},
            "training_date": e.training_date.isoformat() if e.training_date else None,
            "checkpoint_path": e.checkpoint_path,
        })

    # Also inspect outputs/experiments/evaluation/evaluation_summary.json if present on disk
    disk_eval_path = Path(settings.outputs_dir) / "experiments" / "evaluation" / "evaluation_summary.json"
    disk_evaluation = None
    if disk_eval_path.is_file():
        try:
            disk_evaluation = json.loads(disk_eval_path.read_text(encoding="utf-8"))
        except Exception:
            disk_evaluation = None

    disk_train_dir = Path(settings.outputs_dir) / "experiments" / "rdd2022_baseline"
    disk_train_path = disk_train_dir / "training_summary.json"
    disk_training = None
    if disk_train_path.is_file():
        try:
            disk_training = json.loads(disk_train_path.read_text(encoding="utf-8"))
        except Exception:
            disk_training = None

    results_csv_path = disk_train_dir / "results.csv"
    results_png_path = disk_train_dir / "results.png"
    loss_curves_path = disk_train_dir / "training_loss_and_map_curves.png"
    per_class_chart_path = Path(settings.outputs_dir) / "experiments" / "evaluation" / "per_class_performance.png"

    training_artifacts = {
        "results_csv_url": path_to_static_url(results_csv_path) if results_csv_path.is_file() else None,
        "results_png_url": path_to_static_url(results_png_path) if results_png_path.is_file() else None,
        "loss_curves_url": path_to_static_url(loss_curves_path) if loss_curves_path.is_file() else None,
        "per_class_chart_url": path_to_static_url(per_class_chart_path) if per_class_chart_path.is_file() else None,
    }

    return {
        "runtime": detector.get_status(),
        "experiments": experiments,
        "latest_evaluation": disk_evaluation,
        "latest_training": disk_training,
        "training_artifacts": training_artifacts,
        "error_taxonomy": COMMON_ERROR_SOURCES,
    }


@router.post("/model/import-training-csv", tags=["Model & Experiments"])
async def import_training_csv_endpoint(
    file: UploadFile = File(...),
    experiment_name: str = Form("yolo_training_run"),
) -> Dict[str, Any]:
    """Import an Ultralytics YOLO `results.csv` file, plot `results.png`, and persist `training_summary.json`."""
    import csv as _csv
    import io as _io
    from ml.evaluation.evaluator import (
        plot_training_history_curves,
        plot_ultralytics_results_png,
    )

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid UTF-8 CSV file: {exc}") from exc

    rows: List[Dict[str, float]] = []
    reader = _csv.DictReader(_io.StringIO(text))
    headers = [h.strip() for h in (reader.fieldnames or []) if h]
    if "epoch" not in headers or not any("loss" in h for h in headers):
        raise HTTPException(
            status_code=422,
            detail="CSV does not match Ultralytics YOLO results.csv format (requires 'epoch' and loss columns).",
        )

    for raw_row in reader:
        clean: Dict[str, float] = {}
        for k, v in raw_row.items():
            if not k or v is None:
                continue
            try:
                clean[k.strip()] = float(v.strip())
            except ValueError:
                continue
        if clean:
            rows.append(clean)

    if not rows:
        raise HTTPException(status_code=422, detail="Uploaded results.csv contains no valid epoch rows.")

    run_dir = Path(settings.outputs_dir) / "experiments" / "rdd2022_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_dest = run_dir / "results.csv"
    csv_dest.write_text(text, encoding="utf-8")

    # Generate official Ultralytics results.png + dark-themed training curve
    results_png_str = plot_ultralytics_results_png(csv_dest)
    plot_training_history_curves(csv_dest, run_dir)

    summary = {
        "experiment_name": experiment_name,
        "epochs_completed": len(rows),
        "results_csv": str(csv_dest),
        "results_png": str(run_dir / "results.png") if (run_dir / "results.png").is_file() else None,
        "training_history": rows,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "imported": True,
        "epochs": len(rows),
        "results_csv_url": path_to_static_url(csv_dest),
        "results_png_url": path_to_static_url(run_dir / "results.png") if (run_dir / "results.png").is_file() else None,
    }


@router.post("/model/reload", tags=["Model & Experiments"])
def reload_model_endpoint(payload: ModelReloadPayload) -> Dict[str, Any]:
    detector = get_detector_singleton(settings.model_checkpoint_path, settings.inference_device)
    new_status = detector.reload(payload.checkpoint_path)
    return new_status


@router.post("/model/import-evaluation", tags=["Model & Experiments"])
async def import_evaluation_json_endpoint(
    file: UploadFile = File(...),
    experiment_name: str = Form("colab_rdd2022_run"),
    model_name: str = Form("yolo26n"),
    epochs: int = Form(50),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Import an actual `evaluation_summary.json` produced by Colab (`04_model_evaluation.ipynb`) into SQLite."""
    raw = await file.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {exc}") from exc

    required = {"precision", "recall", "map50", "map50_95"}
    if not required.issubset(data.keys()):
        raise HTTPException(
            status_code=422,
            detail=f"Evaluation JSON missing required metric keys: {sorted(required - set(data.keys()))}",
        )

    rec = ModelExperimentRecord(
        experiment_name=experiment_name,
        model_name=model_name,
        dataset_version="RDD2022-4Class",
        epochs=int(data.get("epochs", epochs)),
        batch_size=int(data.get("batch_size", 16)),
        image_size=int(data.get("image_size", 640)),
        precision=float(data["precision"]),
        recall=float(data["recall"]),
        f1_score=float(data.get("f1_score", 0.0)),
        map50=float(data["map50"]),
        map50_95=float(data["map50_95"]),
        inference_ms=float(data.get("inference_ms", 0.0)),
        fps=float(data.get("fps", 0.0)),
        per_class_json=json.dumps(data.get("per_class", [])),
        artifacts_json=json.dumps(data.get("artifacts", {})),
        checkpoint_path=str(data.get("checkpoint_path", settings.model_checkpoint_path)),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)

    # Persist copy to outputs/experiments/evaluation/evaluation_summary.json
    eval_dir = Path(settings.outputs_dir) / "experiments" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "evaluation_summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    return {"imported": True, "experiment_id": rec.id}


# ---------------------------------------------------------------------------
# 8. Dataset Summary & Runtime Settings
# ---------------------------------------------------------------------------
@router.get("/dataset/summary", tags=["Dataset"])
def get_dataset_summary_endpoint() -> Dict[str, Any]:
    return get_dataset_explorer_summary()


@router.get("/settings", tags=["Settings"])
def get_settings_endpoint(db: Session = Depends(get_db)) -> Dict[str, Any]:
    return get_runtime_settings(db)


@router.put("/settings", tags=["Settings"])
def put_settings_endpoint(
    payload: PrioritySettingsPayload,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    try:
        updated = update_runtime_settings(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if payload.checkpoint_path:
        detector = get_detector_singleton(settings.model_checkpoint_path, settings.inference_device)
        detector.reload(payload.checkpoint_path)

    return updated
