# RoadGuard AI — Academic Project Report

**Title:** RoadGuard AI: An Integrated Classical Digital Image Processing and Deep Learning System for Automated Road Surface Damage Detection, Visual Priority Assessment, and Video Corridor Monitoring

---

## Abstract

Timely identification of asphalt pavement distress—such as longitudinal wheel-path cracking, transverse thermal cracking, alligator fatigue networks, and potholes—is vital to road safety and preventive civil maintenance. Manual visual surveys are slow, subjective, and hazardous in live traffic, while naive image-classification approaches fail when multiple overlapping defect categories appear within a single roadway frame. This project presents **RoadGuard AI**, an end-to-end computer vision platform integrating: (1) an independent **Classical Digital Image Processing Laboratory** implementing 23 spatial filtering, contrast enhancement, edge detection, and morphological operators with quantitative signal metrics (PSNR, SSIM, Shannon entropy, Laplacian variance, and Canny edge density); (2) an automated, leakage-safe **Road Damage Dataset 2022 (RDD2022)** preparation pipeline with strict class-schema verification; (3) a transfer-learned **Ultralytics YOLO** multi-object detector targeting four standardized distress categories (`D00`, `D10`, `D20`, `D40`); (4) a transparent **2D Visual Priority Heuristic**; (5) asynchronous **ByteTrack/SORT video analytics**; and (6) a modular **FastAPI + React/TypeScript** monitoring dashboard with CSV and PDF reporting.

---

## 1. Introduction

Roadway networks degrade continuously under cyclic axle loads, thermal expansion-contraction, moisture infiltration, and subgrade settlement. Early-stage surface cracks (`D00`, `D10`), if left unsealed, permit water ingress that weakens base layers, accelerating structural alligator fatigue (`D20`) and eventual pothole formation (`D40`). Automating pavement condition screening using vehicle-mounted monocular cameras offers an affordable alternative to dedicated laser-profilometry survey vans, provided the software system addresses real-world computer vision challenges including cast shadows, worn lane markings, asphalt repair patches, motion blur, and multi-instance localization.

---

## 2. Problem Statement

Given monocular RGB road surface images or dashcam video streams captured across diverse illumination and pavement textures:
1. Localize and classify multiple simultaneous road damage instances across `D00`, `D10`, `D20`, and `D40`.
2. Quantify the signal-level behavior of classical spatial filters, histogram enhancements, gradient operators, and morphological transforms on pavement textures.
3. Rank detected defects using transparent 2D image-space indicators without making scientifically invalid claims of physical depth or millimeter crack width from uncalibrated monocular imagery.
4. Track detections across sequential video frames while distinguishing frame-level bounding-box counts from unique physical roadway defects.

---

## 3. Objectives

1. Build a defensive multi-format dataset pipeline (`VOC_XML`, `YOLO_TXT`, `COCO_JSON`) for RDD2022 that detects source metadata discrepancies, corrupt files, duplicate hashes, and sequential frame leakage.
2. Implement and benchmark 23 classical OpenCV operations for undergraduate Digital Image Processing laboratory evaluation.
3. Fine-tune a configurable YOLO object detector (`yolo26n.pt` / `yolo11n.pt` / `yolov8n.pt`) on Google Colab GPU and evaluate strictly on a held-out 15% test split.
4. Empirically test whether classical image enhancement (`CLAHE`, `Gamma`, `Gaussian`, `Unsharp Mask`) improves or degrades detector mAP.
5. Deliver a production-quality local Windows web platform (FastAPI + SQLite/PostgreSQL + React/TypeScript) with zero fabricated metrics.

---

## 4. Literature Review

- **Classical Edge & Contrast Foundations:** Canny (1986) formalized optimal multi-stage step-edge detection combining Gaussian smoothing, gradient magnitude, non-maximum suppression, and hysteresis thresholding. Zuiderveld (1994) introduced Contrast-Limited Adaptive Histogram Equalization (CLAHE), clipping local tile histograms to enhance low-contrast regions (such as shadowed pavement cracks) without unbounded noise amplification. Tomasi and Manduchi (1998) introduced bilateral filtering, preserving crack boundary discontinuities via joint spatial and radiometric Gaussian kernels.
- **Road Damage Detection Benchmarks:** Maeda et al. (2018) established the initial smartphone-based Road Damage Dataset in Japan, categorizing linear and alligator cracks alongside potholes. Arya et al. (2021) expanded this into a multi-country transfer-learning benchmark across Japan, India, and the Czech Republic, demonstrating cross-domain pavement texture shifts. Arya et al. (2024) published **RDD2022** (47,420 road images across six countries: Japan, India, Czech Republic, Norway, United States, and China) for the IEEE BigData CRDDC'2022 challenge, establishing single-stage YOLO detectors and ensemble strategies as real-time baselines.
- **Multi-Object Video Tracking & Saliency:** Bewley et al. (2016) proposed **SORT**, pairing linear constant-velocity state estimation with Hungarian IoU assignment. Zhang et al. (2022) introduced **ByteTrack**, recovering occluded or motion-blurred trajectories via a second-stage association over low-confidence detections. For spatial feature inspection in CNN detectors, Muhammad and Yeasin (2020) formulated **EigenCAM**, projecting deep convolutional feature maps onto their principal singular vector without requiring class-specific post-NMS backpropagated gradients.

---

## 5. Dataset Description (RDD2022)

- **Primary Source:** Kaggle mirror `aliabdelmenam/rdd-2022` derived from the official `sekilab/RoadDamageDetector` RDD2022 release (Arya et al., 2024).
- **Target Taxonomy:**
  - `0: D00` — Longitudinal Crack
  - `1: D10` — Transverse Crack
  - `2: D20` — Alligator Crack
  - `3: D40` — Pothole
- **Critical Source Verification Finding:** Inspection of `aliabdelmenam/rdd-2022` metadata reveals **5 source classes**: `0: longitudinal crack`, `1: transverse crack`, `2: alligator crack`, `3: other corruption`, and `4: Pothole`. RoadGuard AI explicitly flags class `3` (`other corruption`) rather than assuming `0..3` map to `D00..D40`, excluding `other corruption` only upon explicit confirmation and remapping `4 -> 3 (D40)`.
- **Partitioning Strategy:** Reproducible `70% Train / 15% Validation / 15% Test` (`seed=42`) with exact SHA-256 deduplication and contiguous 20-frame sequence grouping (`prefix_frameNumber`) to reduce adjacent dashcam frame leakage.

---

## 6. Methodology & System Modules

### 6.1 Classical Digital Image Processing (`ml/preprocessing/`)
Implements independent OpenCV pipelines for resizing, min-max/Z-score normalization, RGB/HSV/YCrCb/LAB conversion, BT.601 grayscale projection, Non-Local Means denoising, Gaussian/Median/Bilateral/Average spatial filtering, Global HE, LAB-CLAHE, Gamma correction, percentile contrast stretching, unsharp masking, Sobel/Canny/Laplacian edge operators, and Erosion/Dilation/Opening/Closing/Gradient morphology.

### 6.2 Deep Learning Object Detection (`ml/training/`, `ml/inference/`)
Uses transfer learning from pretrained Ultralytics YOLO weights (`yolo26n.pt` / `yolo11n.pt` / `yolov8n.pt`). Domain-specific augmentation enables horizontal flips (`fliplr=0.5`), mosaic composition (`mosaic=1.0`), HSV illumination jitter, and mild affine rotation (`±5°`), while strictly disabling vertical flips (`flipud=0.0`) and 90° rotations that would invert road perspective or turn `D00` into `D10`.

### 6.3 Visual Priority Assessment (`ml/assessment/`)
For each bounding box $(x_1, y_1, x_2, y_2)$ in an image of dimensions $W \times H$:
$$\text{AreaPct} = 100 \cdot \frac{(x_2 - x_1)(y_2 - y_1)}{W \cdot H}, \quad f_{\text{area}} = \min\!\left(1, \sqrt{\frac{\text{AreaPct}}{A_{\text{sat}}}}\right)$$
$$\text{PriorityScore} = 100 \cdot \left(w_a f_{\text{area}} + w_c W_{\text{class}}(c) + w_p \cdot \text{conf}\right)$$
Default normalized weights are $w_a = 0.40, w_c = 0.35, w_p = 0.25$ with $A_{\text{sat}} = 12\%$ and bands `Low` ($< 35$), `Medium` ($35–65$), and `High` ($\ge 65$).

### 6.4 Video Multi-Object Tracking (`ml/tracking/`)
Decodes `.mp4`, `.avi`, and `.mov` streams via OpenCV, runs per-frame YOLO detection, associates trajectories via two-stage **ByteTrack** (high-confidence Hungarian IoU match followed by low-confidence recovery) or **SORT**, and writes annotated video with HUD FPS telemetry.

---

## 7. Experimental Setup & Evaluation Metrics

- **Training Hardware Target:** Google Colab NVIDIA GPU (T4 / L4 / A100) via `notebooks/03_model_training.ipynb` (`imgsz=640`, `epochs=50`, `batch=16`, early-stopping `patience=12`).
- **Evaluation Protocol:** Strictly executed once on the held-out 15% `test` split via `notebooks/04_model_evaluation.ipynb`.
- **Metrics Calculated:** Precision ($P$), Recall ($R$), $F_1 = \frac{2PR}{P+R}$, $\text{mAP@50}$, $\text{mAP@50:95}$, per-class metrics, confusion matrix, inference latency (ms/frame), and throughput (FPS).

---

## 8. Results

> **Scientific Integrity Status (Pending Colab GPU Training Execution):**
> No fabricated or placeholder accuracy numbers are reported in this document. Once `notebooks/01_dataset_preparation.ipynb`, `03_model_training.ipynb`, and `04_model_evaluation.ipynb` are executed on Google Colab GPU, import `evaluation_summary.json` into the **Model Performance** dashboard and populate `docs/EXPERIMENT_RESULTS.md` with the exact empirical test-set figures.

**Verified Software & Pipeline Validation Results (Executed Locally on Windows):**
- **Automated Test Suite:** 19/19 `pytest` unit and integration tests passing across dataset discrepancy detection, VOC/YOLO/COCO parsers, 23 classical OpenCV operators, PSNR/SSIM metrics, ByteTrack two-stage recovery, SQLite foreign-key cascade integrity, and PDF/CSV report generation.
- **Frontend Production Build:** Strict TypeScript (`tsc -b`) and Vite production bundle verified with zero compilation errors.

---

## 9. Discussion & Limitations

1. **Uncalibrated Monocular Geometry:** Bounding-box area varies with camera pitch, mounting height, and perspective distance; near-field cracks occupy more pixels than identical far-field cracks.
2. **Absence of Ground-Truth Severity Labels:** RDD2022 labels defect presence and category, not civil-engineering Pavement Condition Index (PCI) or crack depth in millimeters.
3. **Environmental Confounders:** Tree/vehicle shadows, thermoplastic lane-marking wear, dark bitumen sealants (`Repair`), standing water, and motion blur remain primary sources of false positives and false negatives.
4. **Local Trajectory IDs:** ByteTrack/SORT IDs maintain local frame continuity and do not deduplicate defects across long GPS corridors if occlusion exceeds `max_age`.

---

## 10. Conclusion & Future Scope

RoadGuard AI demonstrates that combining classical signal-level image processing with modern single-stage deep object detection, transparent visual heuristics, and temporal tracking yields an interpretable, extensible platform for roadway condition monitoring. Future extensions include stereo/LiDAR metric depth calibration, GNSS/odometry spatial geotagging for municipal GIS maps, and pixel-level crack width segmentation.

---

## 11. References

1. Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., & Sekimoto, Y. (2024). *RDD2022: A multi-national image dataset for automatic road damage detection*. Geoscience Data Journal, 11(4), 846–862.
2. Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Mraz, A., Kashiyama, T., & Sekimoto, Y. (2021). *Deep learning-based road damage detection and classification for multiple countries*. Automation in Construction, 132, 103935.
3. Maeda, H., Sekimoto, Y., Seto, T., Kashiyama, T., & Omata, H. (2018). *Road damage detection and classification using deep neural networks with smartphone images*. Computer-Aided Civil and Infrastructure Engineering, 33(12), 1127–1141.
4. Canny, J. (1986). *A computational approach to edge detection*. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6), 679–698.
5. Zuiderveld, K. (1994). *Contrast limited adaptive histogram equalization*. In Graphics Gems IV (pp. 474–485). Academic Press.
6. Tomasi, C., & Manduchi, R. (1998). *Bilateral filtering for gray and color images*. Sixth International Conference on Computer Vision (ICCV), 839–846.
7. Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). *Simple online and realtime tracking*. IEEE International Conference on Image Processing (ICIP), 3464–3468.
8. Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). *ByteTrack: Multi-object tracking by associating every detection box*. European Conference on Computer Vision (ECCV), 1–21.
9. Muhammad, M. B., & Yeasin, M. (2020). *Eigen-CAM: Class activation map using principal components*. International Joint Conference on Neural Networks (IJCNN), 1–7.
