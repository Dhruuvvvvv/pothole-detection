# RoadGuard AI — Intelligent Road Damage Detection, Assessment & Monitoring

RoadGuard AI is an integrated computer vision and digital image processing platform for automated road surface inspection, distress classification, visual priority estimation, multi-object video tracking, and infrastructure monitoring.

Built for both **undergraduate Digital Image Processing & Computer Vision laboratory evaluation** and **production-grade software engineering**, the platform combines an independent classical OpenCV laboratory with an Ultralytics YOLO deep-learning detection and ByteTrack/SORT video analytics pipeline.

---

## Key Capabilities

1. **Classical Digital Image Processing Lab (Module 1)** — 23 interactive OpenCV operations across Preprocessing, Spatial Filtering (Gaussian, Median, Bilateral, Average), Radiometric Enhancement (Global HE, LAB-CLAHE, Gamma, Percentile Contrast Stretching, Unsharp Masking), Edge Detection (Sobel, Canny, Laplacian/LoG), and Morphology (Erosion, Dilation, Opening, Closing, Gradient) with live histograms, difference heatmaps, runtime telemetry, PSNR, and SSIM.
2. **Multi-Instance Road Damage Detection (Module 2)** — Configurable Ultralytics YOLO pipeline (`yolo26n.pt` / `yolo11n.pt` / `yolov8n.pt`) detecting four RDD2022 categories (`D00` Longitudinal Crack, `D10` Transverse Crack, `D20` Alligator Crack, `D40` Pothole).
3. **Transparent Damage Priority Assessment (Module 3)** — Configurable heuristic computing 2D bounding-box occupancy, confidence, and category structural weighting into `Low`, `Medium`, and `High` priority bands without claiming uncalibrated physical dimensions.
4. **Video Analytics & Multi-Object Tracking (Module 4)** — Processes `.mp4`, `.avi`, and `.mov` footage asynchronously using two-stage **ByteTrack** or **SORT** trajectory association, clearly separating frame-level detections from local Track IDs.
5. **Explainability & Controlled Experimentation (Module 5)** — Evaluates test-time classical enhancement impact on labelled mAP, extracts **EigenCAM** neck feature saliency overlays, and documents 7 real-world failure modes.
6. **Full-Stack SaaS Dashboard & Reporting (Modules 6–8)** — FastAPI + SQLAlchemy (SQLite / PostgreSQL) backend and React + TypeScript + Tailwind CSS + Recharts frontend with CSV and ReportLab PDF inspection exports.

---

## Repository Structure

```text
RoadGuardAI/
├── backend/                # FastAPI application, SQLAlchemy ORM, PDF/CSV exporters, API tests
├── frontend/               # React 18 + TypeScript + Vite + Tailwind CSS + Recharts UI
├── ml/                     # Standalone Computer Vision & Machine Learning packages
│   ├── dataset/            # Kaggle CLI downloader, multi-format inspector/parsers, 70/15/15 splitter
│   ├── preprocessing/      # Classical OpenCV operators, PSNR/SSIM/entropy metrics, plot suite
│   ├── training/           # Domain-valid road augmentations & configurable YOLO trainer
│   ├── inference/          # Thread-safe singleton detector with strict class-schema verification
│   ├── evaluation/         # Held-out test evaluator, PR/loss curves, confusion matrix
│   ├── tracking/           # OpenCV video decoder/encoder + ByteTrack & SORT trackers
│   ├── assessment/         # Transparent bounding-box visual priority heuristic engine
│   └── experiments/        # Classical-IP-vs-Detection ablation & EigenCAM activation maps
├── notebooks/              # 4 executable Google Colab GPU notebooks
├── models/                 # Trained checkpoint directory (place `best.pt` here)
├── outputs/                # SQLite DB, uploaded media, annotated outputs, dataset manifests
├── reports/                # Generated inspection reports
├── scripts/                # Windows PowerShell (.ps1) & Command (.cmd) setup/launch scripts
├── tests/                  # Pytest unit & integration tests for ML/CV pipelines
└── docs/                   # Academic report, architecture, dataset, training, API & user guides
```

---

## Quick Start (Windows PowerShell / CMD)

### 1. One-Command Setup
```powershell
cd C:\Users\Dhruv\Downloads\RoadGuardAI
.\scripts\setup_windows.cmd
```

### 2. Start Backend & Frontend
Open two terminals in `C:\Users\Dhruv\Downloads\RoadGuardAI`:

**Terminal 1 (FastAPI Backend on `http://127.0.0.1:8000`):**
```powershell
.\scripts\start_backend.cmd
```

**Terminal 2 (React Frontend on `http://localhost:5173`):**
```powershell
.\scripts\start_frontend.cmd
```

### 3. Run Automated Tests
```powershell
.\scripts\run_tests.cmd
```

> **Model Readiness Note:** Before placing a trained `models/best.pt` checkpoint, the **Image Processing Lab**, **Dataset Explorer**, **Settings**, and **Overview** pages operate immediately, while **Image Analysis** and **Video Analytics** display a transparent `Model Not Configured` banner rather than fabricating predictions. Train on Google Colab via `notebooks/01_dataset_preparation.ipynb` → `notebooks/03_model_training.ipynb` and copy `best.pt` into `models/best.pt`.

---

## Documentation Index

- [Academic Project Report](docs/PROJECT_REPORT.md)
- [System Architecture](docs/SYSTEM_ARCHITECTURE.md)
- [Dataset Documentation & Discrepancy Audit](docs/DATASET_DOCUMENTATION.md)
- [Model Training & Colab Guide](docs/MODEL_TRAINING_GUIDE.md)
- [API Documentation](docs/API_DOCUMENTATION.md)
- [Windows Installation Guide](docs/INSTALLATION_GUIDE.md)
- [User Guide](docs/USER_GUIDE.md)
- [Experiment Results & Ablation Protocol](docs/EXPERIMENT_RESULTS.md)
- [Engineering Limitations & Calibration Scope](docs/LIMITATIONS.md)
