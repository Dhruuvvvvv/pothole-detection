# Technical Limitations & Scientific Scope

RoadGuard AI enforces strict engineering honesty regarding what 2D monocular computer vision can and cannot infer.

---

## 1. No Physical Dimension Claims Without Camera Calibration

Monocular RGB images project 3D roadway geometry onto a 2D sensor plane without metric scale:
- Without intrinsic calibration ($f_x, f_y, c_x, c_y$), camera mounting height, pitch angle, or depth sensing (stereo / LiDAR / structured light), pixel width and bounding-box area **cannot** be converted into millimeters, centimeters, or cubic pothole volume.
- Due to perspective foreshortening, an identical pothole occupies a much larger pixel fraction in the lower near-field of a dashcam frame than near the horizon.

---

## 2. Visual Priority ≠ Civil Engineering Structural Severity

- RDD2022 provides 2D bounding boxes and four categorical distress labels (`D00`, `D10`, `D20`, `D40`), **not** ground-truth engineering severity ratings (such as ASTM D6433 Pavement Condition Index, crack aperture width in mm, or pothole depth > 50 mm).
- Accordingly, RoadGuard AI's **Estimated Visual Priority Score** (`Low` / `Medium` / `High`) is explicitly documented as an **image-space triage heuristic** combining normalized bounding-box occupancy, category weighting, and model confidence.

---

## 3. Video Track IDs ≠ Globally Unique Physical Defects

- In `ml/tracking/video_processor.py`, **ByteTrack** and **SORT** assign integer `track_id` values to maintain local frame-to-frame bounding-box continuity.
- If a defect leaves the camera field of view, is occluded by traffic for longer than `max_age` frames, or fragments from one bounding box into two as the vehicle approaches, a new `track_id` may be assigned.
- Consequently, RoadGuard AI reports **Frame-Level Detections** and **Local Trajectory Track IDs** separately and never claims that `unique_track_ids_observed` equals the exact physical count of potholes along a roadway segment without centimeter-level GNSS/IMU odometry fusion.

---

## 4. Explainability Scope (EigenCAM vs. Grad-CAM / SHAP)

- Ordinary bounding-box overlays are detection visualizations, never labelled as Grad-CAM or SHAP.
- The optional **EigenCAM** overlay (`compute_yolo_eigencam`) computes the first principal singular vector of deep YOLO neck feature activations. While useful for inspecting spatial feature focus (e.g., whether backbone activations respond to road surface vs. sky/trees), it is **class-agnostic** and coarse in spatial resolution compared to input pixels.

---

## 5. Domain Shift & Environmental Confounders

- Performance can degrade under night-time glare, wet/reflective asphalt, snow, unpaved gravel shoulders, dark bitumen crack-sealing strips (`Repair`), and camera motion blur. Retraining or domain adaptation is recommended before deploying on specialized municipal camera rigs.
