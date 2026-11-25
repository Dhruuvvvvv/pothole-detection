# Experiment Results & Empirical Evaluation

---

## 1. Active RDD2022 YOLO11s Detector Benchmark (`models/best.pt`)

**Provenance:** Open-source fine-tuned **Ultralytics YOLO11s** checkpoint (`https://huggingface.co/Th3Samaritan/rdd2022-yolo11s`), trained and evaluated across the 6-country 4-class RDD2022 dataset (`D00`, `D10`, `D20`, `D40`) on a **held-out test split**.

### Overall Held-Out Test Performance

| Model Variant | Precision | Recall | F1-Score | mAP@50 | mAP@50-95 | GPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11s (Active `models/best.pt`)** | **`64.56%`** | **`55.11%`** | **`59.23%`** | **`59.23%`** | **`30.03%`** | `3.81 ms` (`262 FPS`) |
| **YOLO11n** | `61.87%` | `51.00%` | `55.57%` | `55.24%` | `27.46%` | `1.68 ms` (`595 FPS`) |
| **YOLOv8n** | `59.81%` | `51.79%` | `55.27%` | `54.52%` | `26.86%` | `1.57 ms` (`636 FPS`) |

### Per-Category Held-Out Test Performance (`YOLO11s`)

| Class ID | Code | Damage Category | Precision | Recall | F1-Score | Test AP@50 | Test AP@50-95 |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `0` | `D00` | Longitudinal Crack | `65.14%` | `54.96%` | `59.62%` | `58.95%` | `32.86%` |
| `1` | `D10` | Transverse Crack | `59.29%` | `54.58%` | `56.84%` | `55.25%` | `26.71%` |
| `2` | `D20` | Alligator Crack | `71.45%` | `68.21%` | `69.79%` | `73.02%` | `39.13%` |
| `3` | `D40` | Pothole | `62.35%` | `42.69%` | `50.68%` | `49.69%` | `21.40%` |

> **Reproducibility Note:** You can also fine-tune your own custom checkpoint from scratch on Google Colab GPU at any time using `notebooks/01_dataset_preparation.ipynb` → `notebooks/03_model_training.ipynb` → `notebooks/04_model_evaluation.ipynb` and overwrite `models/best.pt`.

---

## 2. Classical Image Processing Signal Benchmark (Verified Locally)

Verified across all 23 OpenCV operators via `tests/test_classical_ip.py`:

| Operator Category | Technique | Observed Signal Effect on Pavement Imagery |
| :--- | :--- | :--- |
| **Filtering** | Gaussian (`5×5, σ=1.4`) | Attenuates high-frequency aggregate grain; reduces Laplacian variance uniformly across both noise and crack edges. |
| **Filtering** | Median (`5×5`) | Eliminates impulsive salt-and-pepper aggregate speckle while preserving monotonic crack step edges better than Box/Gaussian blur. |
| **Filtering** | Bilateral (`d=9, σ_c=75, σ_s=75`) | Smooths homogeneous asphalt patches while preserving high crack-rim gradient contrast. |
| **Enhancement** | LAB-CLAHE (`clip=2.5, 8×8`) | Expands local contrast inside cast tree/vehicle shadows without saturating sunlit lane markings. |
| **Enhancement** | Gamma (`γ=0.7`) | Non-linearly lifts dark shadow and pothole-cavity luminance while compressing bright highlights. |
| **Enhancement** | Unsharp Mask (`σ=1.5, amt=1.4`) | Increases Laplacian variance and Canny edge density along hairline cracks (`D00`/`D10`). |
| **Edge Detection** | Canny (`T1=75, T2=175`) | Produces thin, single-pixel connected crack contours via non-maximum suppression and hysteresis. |
| **Morphology** | Closing (`Ellipse 5×5`) | Bridges narrow micro-gaps along fragmented crack contours. |
