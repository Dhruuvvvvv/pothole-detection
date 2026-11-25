# User Guide — RoadGuard AI Web Platform

## Navigation Walkthrough

### 1. Overview Dashboard (`/`)
Displays real-time database metrics (`Images Analysed`, `Total Detections`, `D00`, `D10`, `D20`, `D40`, `High-Priority Defects`), category distribution bar charts, daily inspection activity curves, and recent runs. When the database is empty, quick-launch cards guide you directly to the **Image Processing Lab** or **Image Analysis**.

### 2. Image Analysis (`/image-analysis`)
1. Drag and drop a road surface image (`.jpg`, `.png`, `.bmp`, `.webp`).
2. Adjust **Confidence Threshold** (default `0.25`), **NMS IoU Threshold** (default `0.45`), and **Inference Resolution** (`512`, `640`, or `960`).
3. Optionally enable **EigenCAM** neck feature activation overlay.
4. Click **Run Damage Detection**.
5. Inspect detections using the **Interactive Split Curtain** or **Side-by-Side** view, review bounding-box coordinates and priority scores in the instance table, and export the **Annotated PNG**, **CSV Report**, or **PDF Report**.

### 3. Video Analytics (`/video-analytics`)
1. Upload a road recording (`.mp4`, `.avi`, `.mov`).
2. Choose **ByteTrack** (two-stage high/low confidence recovery) or **SORT**, set confidence threshold, and choose **Frame Stride** (`1`, `2`, or `3`).
3. Click **Start Video Inspection** and monitor the live progress bar.
4. Play back the annotated tracking video, inspect the frame-level detection timeline chart, and download the output video or CSV/PDF summary.

### 4. Image Processing Lab (`/image-processing-lab`)
Works immediately even before training a deep-learning model:
1. Upload any road photograph—or click **Load Synthetic Calibration Pattern** to generate a test card featuring asphalt grain, salt-and-pepper speckle, cast shadow, cracks, and a pothole.
2. Select a category (**Preprocessing**, **Filtering**, **Enhancement**, **Edge Detection**, **Morphology**) and choose one of the **23 OpenCV operations**.
3. Tune sliders (`kernel_size`, `sigma`, `clip_limit`, `gamma`, `threshold1/2`, `iterations`) and click **Apply Operation**.
4. Switch between **Interactive Curtain**, **Side-by-Side**, and **Difference Heatmap**, inspect the quantitative signal table (`Mean`, `RMS/Michelson Contrast`, `Entropy`, `Laplacian Variance`, `Edge Density`, `PSNR`, `SSIM`), and compare Original vs Processed luminance histograms.
5. Click **Compare All 4 Spatial Filters** to benchmark Gaussian, Median, Bilateral, and Average filters simultaneously.

### 5. Model Performance (`/model-performance`)
Displays held-out test set metrics (`Precision`, `Recall`, `F1-Score`, `mAP@50`, `mAP@50-95`, `FPS`), per-class bar charts, loss curves, confusion matrix, and the 7-category computer vision failure taxonomy. Use **Import Colab Evaluation JSON** to import `evaluation_summary.json` from Google Colab.

### 6. Damage Reports (`/damage-reports`)
Search past inspections by filename, filter by damage class (`D00..D40`), priority band (`High`/`Medium`/`Low`), or modality (`image`/`video`), inspect full details in the modal viewer, download CSV/PDF reports, or delete obsolete records.

### 7. Dataset Explorer (`/dataset-explorer`)
Inspects RDD2022 target class definitions, documents the verified 5-class `aliabdelmenam/rdd-2022` mapping audit, and displays split ratios, class distributions, ground-truth visual samples, and `annotation_issues.csv` logs once dataset manifests are placed in `outputs/dataset/`.

### 8. Settings (`/settings`)
Customize `Low → Medium` and `Medium → High` visual priority thresholds, component weights (`area`, `category`, `confidence`), per-class risk factors, default inference thresholds, and hot-reload `models/best.pt`.
