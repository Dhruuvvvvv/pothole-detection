"""Upload validation, decode verification, static URL mapping, and temporary file cleanup."""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from backend.app.core.config import settings
from ml.constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str) -> str:
    base = Path(name or "upload").name
    cleaned = _SAFE_FILENAME_RE.sub("_", base).strip("._")
    return cleaned or "file"


def path_to_static_url(file_path: Optional[str | Path]) -> Optional[str]:
    if not file_path:
        return None
    p = Path(file_path).resolve()
    outputs_root = Path(settings.outputs_dir).resolve()
    try:
        rel = p.relative_to(outputs_root)
        return f"/static/{rel.as_posix()}"
    except ValueError:
        return None


async def read_and_validate_image_upload(upload: UploadFile) -> Tuple[np.ndarray, str, bytes]:
    """Validate uploaded image extension, size limit, and OpenCV pixel decode integrity."""
    orig_name = sanitize_filename(upload.filename or "image.jpg")
    ext = Path(orig_name).suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image extension '{ext}'. Allowed formats: {sorted(IMAGE_EXTENSIONS)}",
        )

    raw_bytes = await upload.read()
    max_bytes = settings.max_image_upload_mb * 1024 * 1024
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")
    if len(raw_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds maximum allowed size of {settings.max_image_upload_mb} MB.",
        )

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file could not be decoded as a valid raster image.",
        )
    return bgr, orig_name, raw_bytes


async def save_and_validate_video_upload(upload: UploadFile, target_dir: Path) -> Tuple[Path, str]:
    """Stream uploaded video to disk while enforcing size limits and OpenCV container readability."""
    orig_name = sanitize_filename(upload.filename or "video.mp4")
    ext = Path(orig_name).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Supported formats: .mp4, .avi, .mov",
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex[:10]
    saved_path = target_dir / f"{uid}_{orig_name}"

    max_bytes = settings.max_video_upload_mb * 1024 * 1024
    total_written = 0

    try:
        with open(saved_path, "wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds maximum allowed size of {settings.max_video_upload_mb} MB.",
                    )
                f.write(chunk)
    except HTTPException:
        saved_path.unlink(missing_ok=True)
        raise

    if total_written == 0:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded video file is empty.")

    cap = cv2.VideoCapture(str(saved_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is corrupt or cannot be decoded as a valid video stream.",
        )

    return saved_path, orig_name


def cleanup_ephemeral_processed_files(retention_hours: Optional[int] = None) -> int:
    """Remove stale temporary lab preview images in outputs/processed older than retention_hours."""
    hours = retention_hours if retention_hours is not None else settings.temp_retention_hours
    cutoff = time.time() - (hours * 3600)
    proc_dir = Path(settings.outputs_dir) / "processed"
    if not proc_dir.is_dir():
        return 0
    removed = 0
    for p in proc_dir.iterdir():
        if p.is_file() and p.name != ".gitkeep":
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    removed += 1
            except OSError:
                continue
    return removed
