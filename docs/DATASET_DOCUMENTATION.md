# Dataset Documentation — RDD2022 Acquisition, Verification & Splitting

## 1. Dataset Overview

- **Primary Mirror:** [`aliabdelmenam/rdd-2022`](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022)
- **Official Publisher Repository:** [`sekilab/RoadDamageDetector`](https://github.com/sekilab/RoadDamageDetector) (CRDDC'2022 / Arya et al., 2024)

### Target Categories

| Class ID | Code | Category Name | Description |
| :---: | :---: | :--- | :--- |
| `0` | `D00` | Longitudinal Crack | Linear wheel-path or construction-joint cracks parallel to traffic flow |
| `1` | `D10` | Transverse Crack | Thermal or reflection cracks perpendicular to road centerline |
| `2` | `D20` | Alligator Crack | Interconnected fatigue cracking mesh indicating structural distress |
| `3` | `D40` | Pothole | Localized bowl-shaped cavity in the road surface |

---

## 2. Critical Source Discrepancy Audit (`aliabdelmenam/rdd-2022`)

RoadGuard AI never assumes numeric class IDs in a Kaggle mirror match `0=D00, 1=D10, 2=D20, 3=D40`.

In `aliabdelmenam/rdd-2022`, source metadata defines **5 classes**:
```python
names = [
    "longitudinal crack",  # Source ID 0 -> Target 0 (D00)
    "transverse crack",    # Source ID 1 -> Target 1 (D10)
    "alligator crack",     # Source ID 2 -> Target 2 (D20)
    "other corruption",    # Source ID 3 -> NON-TARGET (bumps, blurred lines, manholes, repairs)
    "Pothole",             # Source ID 4 -> Target 3 (D40)
]
```

**Why this matters:** Any pipeline that blindly filters `class_id in {0, 1, 2, 3}` mislabels `other corruption` as `D40 (Pothole)` and discards every true `Pothole` (`class_id == 4`)!

**How `ml/dataset/` handles this:**
1. `inspect_dataset_directory()` reads `data.yaml` / `classes.txt` / XML `<name>` / COCO `categories` and flags `other corruption` in `non_target_classes`.
2. Calling `parse_dataset(..., non_target_policy="raise")` (the default) halts with `DatasetClassDiscrepancyError` and prints the full inspection report.
3. Passing `--confirm-exclude-non-target` (`non_target_policy="exclude_confirmed"`) transparently excludes `other corruption` bounding boxes (recording exact excluded counts) and remaps `4 ("Pothole") -> 3 ("D40")`.

---

## 3. Data Quality Checks & Leakage-Safe Splitting

`ml/dataset/validator.py` and `ml/dataset/splitter.py` perform:
- **Decode Integrity:** Every image is verified via Pillow `verify()` + full pixel decode; unreadable files are quarantined and logged to `annotation_issues.csv`.
- **Bounding-Box Validation:** Rejects NaN/Inf, out-of-frame boxes, and degenerate `< 2 px` boxes; clamps minor 1-pixel border overhang.
- **Duplicate Removal:** Computes exact **SHA-256** digests and 64-bit perceptual **dHash** fingerprints so duplicate images cannot leak between `train` and `test`.
- **Sequential Road-Session Grouping:** Groups nearby numbered frames sharing a country/camera prefix (e.g., `India_000121.jpg`..`India_000139.jpg` in 20-frame blocks) into a single `group_id` via Union-Find before assigning **70% Train / 15% Validation / 15% Test** (`seed=42`).

---

## 4. CLI Usage & Generated Artifacts

```powershell
# 1. Check Kaggle authentication status (never prints secret keys)
.\.venv\Scripts\python.exe -m ml.dataset.cli check-auth

# 2. Download & extract via Kaggle CLI
.\.venv\Scripts\python.exe -m ml.dataset.cli download --slug aliabdelmenam/rdd-2022 --download-dir .\outputs\raw_dataset

# 3. Inspect actual extracted structure & verify class metadata
.\.venv\Scripts\python.exe -m ml.dataset.cli inspect --source-dir .\outputs\raw_dataset\extracted

# 4. Prepare 70/15/15 YOLO dataset after confirming exclusion of non-target classes
.\.venv\Scripts\python.exe -m ml.dataset.cli prepare --source-dir .\outputs\raw_dataset\extracted --output-dir .\outputs\dataset --confirm-exclude-non-target
```

Generated artifacts in `outputs/dataset/`:
- `dataset_manifest.csv`
- `dataset_summary.csv`
- `class_distribution.csv`
- `annotation_issues.csv`
- `dataset_config.yaml`
- `samples/gt_*.jpg`
