# Model Training Guide — Google Colab GPU & Local Execution

## 1. Recommended Workflow: Google Colab with NVIDIA GPU

Because RDD2022 exceeds 10 GB compressed (~47k high-resolution road images across multiple countries), training on a Google Colab T4/L4/A100 GPU is the primary workflow.

### Step-by-Step Colab Execution

1. Upload the four notebooks in `notebooks/` to Google Colab:
   - `01_dataset_preparation.ipynb`
   - `02_image_processing.ipynb`
   - `03_model_training.ipynb`
   - `04_model_evaluation.ipynb`
2. In Colab, select **Runtime → Change runtime type → Hardware accelerator: T4 GPU (or L4/A100)**.
3. Run **`01_dataset_preparation.ipynb`**:
   - Mounts Google Drive at `/content/drive/MyDrive/RoadGuardAI`.
   - Prompts you to upload `kaggle.json` (stored privately in `/root/.kaggle/kaggle.json` with `0600` permissions).
   - Downloads `aliabdelmenam/rdd-2022`, inspects `data.yaml`, confirms exclusion of `other corruption`, and builds the leakage-safe 70/15/15 dataset.
   - Copies all CSV/YAML manifests to `MyDrive/RoadGuardAI/dataset_manifests/`.
4. Run **`02_image_processing.ipynb`** to generate the 12-panel classical OpenCV study and PSNR/SSIM CSV metrics.
5. Run **`03_model_training.ipynb`**:
   - Initializes `yolo26n.pt` (or configurable `yolo11n.pt` / `yolov8n.pt` / `yolo11s.pt`).
   - Trains with perspective-safe road augmentations (`fliplr=0.5`, `flipud=0.0`, `degrees=5.0`, `mosaic=1.0`, `patience=12`).
   - Saves checkpoints to `MyDrive/RoadGuardAI/models/best.pt` and `last.pt` (supporting `RESUME = True` if Colab disconnects).
6. Run **`04_model_evaluation.ipynb`**:
   - Evaluates `best.pt` strictly on the untouched 15% `test` split.
   - Runs the Classical-IP-vs-Detection ablation (`Original` vs `Gaussian` vs `CLAHE`).
   - Saves `evaluation_summary.json` to Google Drive.

---

## 2. Transferring Trained Artifacts into the Local Application

1. Download `best.pt` from `MyDrive/RoadGuardAI/models/best.pt` and place it at:
   ```text
   RoadGuardAI/models/best.pt
   ```
2. Open the local RoadGuard AI web app (`http://localhost:5173`), navigate to **Settings** (or click the banner button on **Overview**), and click **Hot-Reload Checkpoint**.
3. Open **Model Performance**, click **Import Colab Evaluation JSON**, and select `evaluation_summary.json` to populate all test-set Precision, Recall, F1, mAP@50, mAP@50-95, per-class charts, and confusion matrices.
4. *(Optional)* Copy the CSV/YAML files from `MyDrive/RoadGuardAI/dataset_manifests/` into `RoadGuardAI/outputs/dataset/` to populate the **Dataset Explorer** tab.

---

## 3. Local Python CLI Training (Optional on Local CUDA GPU)

If you have a prepared dataset locally and a CUDA-enabled PyTorch installation:

```python
from ml.training import TrainingConfig, train_yolo_detector

summary = train_yolo_detector(
    TrainingConfig(
        dataset_yaml="./outputs/dataset/dataset_config.yaml",
        model_architecture="yolo26n.pt",
        epochs=50,
        batch_size=16,
        image_size=640,
        export_best_to="./models/best.pt",
    )
)
```
