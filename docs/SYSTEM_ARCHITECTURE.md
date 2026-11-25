# System Architecture — RoadGuard AI

## Architectural Overview

RoadGuard AI follows a decoupled three-tier architecture in which the **Machine Learning & Computer Vision (`ml/`)** package operates independently of both the **FastAPI Backend (`backend/`)** and the **React + TypeScript Frontend (`frontend/`)**.

```mermaid
flowchart TD
    UI["React 18 + TypeScript + Tailwind UI (Port 5173)"]
    API["FastAPI Modular Router Layer (Port 8000)"]
    DB[("SQLite / PostgreSQL via SQLAlchemy 2.x")]
    IP["ml.preprocessing (23 OpenCV Operators + PSNR/SSIM)"]
    DET["ml.inference (Singleton YOLO Detector)"]
    ASSESS["ml.assessment (Visual Priority Heuristic)"]
    TRACK["ml.tracking (OpenCV + ByteTrack / SORT)"]
    DS["ml.dataset (Kaggle CLI + VOC/YOLO/COCO + 70/15/15 Split)"]
    EXP["ml.evaluation & ml.experiments (Test Metrics + EigenCAM)"]

    UI <-->|"REST JSON + Multipart + Static Media"| API
    API --> IP
    API --> DET
    DET --> ASSESS
    API -->|"Background Job"| TRACK
    TRACK --> DET
    API <--> DB
    DS -->|"Prepared YOLO Dataset"| EXP
```

---

## Core Design Principles

1. **Strict ML/UI Independence:** Every module inside `ml/` can be imported from CLI scripts, `pytest` suites, or Jupyter/Colab notebooks without starting FastAPI or React.
2. **Singleton Checkpoint Caching & Hot-Reload:** `RoadDamageDetector` loads PyTorch/Ultralytics weights once into GPU/CPU memory under a thread lock and exposes `/api/model/reload` so newly trained checkpoints activate without server downtime.
3. **Defensive Class-Schema Guardrails:** Both the dataset parser and runtime detector validate class names against `{D00, D10, D20, D40}` and refuse to run mislabelled 5-class or 80-class COCO weights silently.
4. **Non-Blocking Video Processing:** `POST /api/analyze/video` validates upload headers and dispatches frame-by-frame inference + tracking to a background worker, updating `progress_pct` in SQLite so the frontend polls `/api/analyze/video/{id}/status` smoothly.
5. **Database Abstraction:** `backend/app/database/session.py` uses SQLAlchemy 2.x ORM with `PRAGMA foreign_keys=ON` for local SQLite (`outputs/roadguard.db`) and switches to PostgreSQL solely by changing `DATABASE_URL` in `.env`.
