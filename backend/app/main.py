"""FastAPI application entry point for RoadGuard AI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router as api_router
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.database.session import init_database
from backend.app.utils.file_validation import cleanup_ephemeral_processed_files
from ml.inference.detector import get_detector_singleton


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Path(settings.outputs_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.reports_dir).mkdir(parents=True, exist_ok=True)
    init_database()
    cleaned = cleanup_ephemeral_processed_files()
    if cleaned:
        logger.info("Cleaned %d expired temporary processed images.", cleaned)

    detector = get_detector_singleton(settings.model_checkpoint_path, settings.inference_device)
    st = detector.get_status()
    logger.info("RoadGuard AI startup complete | Model status: %s | Device: %s", st["status"], st["device"])
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="RoadGuard AI API",
        description=(
            "Intelligent Road Damage Detection, Classical Digital Image Processing, "
            "Visual Priority Assessment, Video Multi-Object Tracking & Analytics API."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    outputs_path = Path(settings.outputs_dir).resolve()
    outputs_path.mkdir(parents=True, exist_ok=True)
    application.mount("/static", StaticFiles(directory=str(outputs_path)), name="static")

    application.include_router(api_router)
    return application


app = create_app()
