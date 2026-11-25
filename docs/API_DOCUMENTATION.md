# API Documentation — RoadGuard AI FastAPI Backend

Base URL (local development): `http://127.0.0.1:8000`
Interactive OpenAPI / Swagger UI: `http://127.0.0.1:8000/docs`

---

## 1. System & Health

### `GET /api/health`
Returns database connectivity, model readiness (`ready` vs `model_not_configured`), and CUDA GPU availability.

---

## 2. Deep Learning Inference & Tracking

### `POST /api/analyze/image`
- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `file` *(required)*: Image binary (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, max 25 MB)
  - `conf_threshold` *(float, default `0.25`)*: Detection confidence `[0.05, 0.95]`
  - `iou_threshold` *(float, default `0.45`)*: NMS IoU threshold `[0.10, 0.95]`
  - `image_size` *(int, default `640`)*: Inference resolution
  - `include_eigencam` *(bool, default `false`)*: Generates EigenCAM neck feature overlay
- **Responses:**
  - `200 OK`: Full `AnalysisSummaryResponse` with persisted `id`, `input_url`, `output_url`, `detections[]`, and priority breakdowns.
  - `400 Bad Request` / `413 Content Too Large`: Invalid image or size exceeded.
  - `503 Service Unavailable`: `{"detail": {"code": "MODEL_NOT_CONFIGURED", ...}}` when `models/best.pt` is absent.

### `POST /api/analyze/video`
- **Content-Type:** `multipart/form-data`
- **Query Params:** `wait=false` (async background execution by default; set `wait=true` for synchronous test execution)
- **Fields:**
  - `file` *(required)*: Video binary (`.mp4`, `.avi`, `.mov`, max 250 MB)
  - `tracker_type`: `"bytetrack"` (default) or `"sort"`
  - `conf_threshold`: `0.25`
  - `frame_stride`: `1` (process every Nth frame)
- **Response:** Queued `AnalysisSummaryResponse` (`status: "queued"`).

### `GET /api/analyze/video/{analysis_id}/status`
Polls background video processing progress (`progress_pct`, `status`, and final `summary` + `output_url` upon completion).

---

## 3. Classical Image Processing Lab

### `GET /api/process/catalog`
Returns metadata, mathematical formulas, and parameter specifications for all 23 classical OpenCV operations.

### `POST /api/process/image`
- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `file`: Input image
  - `operation`: Operation ID (e.g. `clahe`, `bilateral_filter`, `canny_edge`, `morph_closing`, `unsharp_masking`)
  - `parameters_json`: JSON object string of operator parameters
- **Response:** `original_url`, `processed_url`, `difference_url`, `processing_time_ms`, `original_metrics`, `processed_metrics` (including `psnr_db` and `ssim_score`), and 64-bin RGB/Luminance histograms.

### `POST /api/process/compare-filters`
Runs Gaussian, Median, Bilateral, and Average filters side-by-side at `kernel_size` and returns comparative URLs + quantitative metrics.

---

## 4. Dashboard, Analytics & Reports

- `GET /api/dashboard/summary` — Live database totals (`D00..D40`, priority bands, recent runs, model status).
- `GET /api/analytics/distribution` — Category and priority band chart series.
- `GET /api/analytics/history?limit=30` — Daily inspection timeline.
- `GET /api/reports` — Paginated list supporting `search`, `class_code`, `priority_band`, and `analysis_type` filters.
- `GET /api/reports/{analysis_id}` — Detailed report including bounding-box instances.
- `GET /api/reports/{analysis_id}/export/csv` — Downloads UTF-8 CSV report.
- `GET /api/reports/{analysis_id}/export/pdf` — Generates and downloads formatted ReportLab PDF inspection report.
- `DELETE /api/reports/{analysis_id}` — Deletes report, media artifacts, and cascading detection rows.

---

## 5. Model, Dataset & Settings

- `GET /api/model/info` — Runtime status, experiment registry, evaluation summary, and error taxonomy.
- `POST /api/model/reload` — Hot-reloads `models/best.pt` (or custom path) without restarting Uvicorn.
- `POST /api/model/import-evaluation` — Imports `evaluation_summary.json` into `model_experiments`.
- `GET /api/dataset/summary` — Reads `outputs/dataset/` manifests, class balance, issues, and GT samples.
- `GET /api/settings` & `PUT /api/settings` — Reads/updates Priority Heuristic thresholds and weights.
