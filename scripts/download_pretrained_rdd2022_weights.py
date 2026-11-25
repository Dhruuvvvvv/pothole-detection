"""Download & extract verified open-source YOLO11s weights and training artifacts for 4-class RDD2022."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.evaluation.evaluator import (  # noqa: E402
    _plot_per_class_chart,
    extract_checkpoint_training_artifacts,
)

REPO_ID = "Th3Samaritan/rdd2022-yolo11s"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"


def download_file(filename: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO_ID}/{filename} -> {dest} ...")
    with requests.get(f"{BASE_URL}/{filename}", stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=262144):
                if chunk:
                    f.write(chunk)
    print(f"  Saved {dest.name} ({dest.stat().st_size / (1024 * 1024):.2f} MB)")


def _f1(p: float, r: float) -> float:
    return round(2.0 * p * r / max(p + r, 1e-9), 4)


def _sync_sqlite_experiment(
    db_path: Path,
    eval_summary: dict,
    train_summary: dict,
) -> None:
    if not db_path.is_file():
        return
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='model_experiments'"
        )
        if not cur.fetchone():
            return
        # Remove replaced/stale dronefreak row if present
        cur.execute(
            "DELETE FROM model_experiments WHERE experiment_name LIKE '%dronefreak%'"
        )
        exp_name = "huggingface_Th3Samaritan_rdd2022_yolo11s"
        cur.execute(
            "SELECT id FROM model_experiments WHERE experiment_name = ? OR model_name = 'yolo11s' ORDER BY id DESC LIMIT 1",
            (exp_name,),
        )
        row = cur.fetchone()
        epochs = int(train_summary.get("epochs_completed", 80))
        batch_size = int(train_summary.get("batch_size", 32))
        image_size = int(train_summary.get("image_size", 640))
        per_class_json = json.dumps(eval_summary.get("per_class", []))
        artifacts_json = json.dumps(eval_summary.get("artifacts", {}))
        now_iso = datetime.now(timezone.utc).isoformat()

        if row:
            cur.execute(
                """
                UPDATE model_experiments
                SET experiment_name = ?,
                    model_name = 'yolo11s',
                    dataset_version = 'RDD2022-4Class',
                    epochs = ?,
                    batch_size = ?,
                    image_size = ?,
                    precision = ?,
                    recall = ?,
                    f1_score = ?,
                    map50 = ?,
                    map50_95 = ?,
                    inference_ms = ?,
                    fps = ?,
                    per_class_json = ?,
                    artifacts_json = ?,
                    checkpoint_path = ?
                WHERE id = ?
                """,
                (
                    exp_name,
                    epochs,
                    batch_size,
                    image_size,
                    eval_summary["precision"],
                    eval_summary["recall"],
                    eval_summary["f1_score"],
                    eval_summary["map50"],
                    eval_summary["map50_95"],
                    eval_summary["inference_ms"],
                    eval_summary["fps"],
                    per_class_json,
                    artifacts_json,
                    eval_summary["checkpoint_path"],
                    row[0],
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO model_experiments (
                    experiment_name, model_name, dataset_version, epochs, batch_size, image_size,
                    precision, recall, f1_score, map50, map50_95, inference_ms, fps,
                    per_class_json, artifacts_json, training_date, checkpoint_path
                ) VALUES (?, 'yolo11s', 'RDD2022-4Class', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exp_name,
                    epochs,
                    batch_size,
                    image_size,
                    eval_summary["precision"],
                    eval_summary["recall"],
                    eval_summary["f1_score"],
                    eval_summary["map50"],
                    eval_summary["map50_95"],
                    eval_summary["inference_ms"],
                    eval_summary["fps"],
                    per_class_json,
                    artifacts_json,
                    now_iso,
                    eval_summary["checkpoint_path"],
                ),
            )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download best.pt and metrics.json even if already cached locally.",
    )
    args = parser.parse_args()

    models_dir = ROOT / "models"
    exp_dir = ROOT / "outputs" / "experiments" / "rdd2022_baseline"
    eval_dir = ROOT / "outputs" / "experiments" / "evaluation"
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    best_pt = models_dir / "best.pt"
    metrics_json_path = eval_dir / "source_metrics.json"

    if args.force_download or not best_pt.is_file():
        download_file("best.pt", best_pt)
    else:
        print(f"Using existing checkpoint: {best_pt} ({best_pt.stat().st_size / (1024 * 1024):.2f} MB)")

    if args.force_download or not metrics_json_path.is_file():
        download_file("metrics.json", metrics_json_path)
    else:
        print(f"Using existing metrics file: {metrics_json_path}")

    # 1. Extract authentic 80-epoch results.csv, results.png, and training_summary.json from best.pt
    train_summary = extract_checkpoint_training_artifacts(
        checkpoint_path=best_pt,
        output_dir=exp_dir,
        experiment_name="rdd2022_yolo11s_80ep",
        source_repository=f"https://huggingface.co/{REPO_ID}",
    )
    results_csv = Path(train_summary["results_csv"])
    results_png = Path(train_summary["results_png"]) if train_summary.get("results_png") else None

    # Copy convenient top-level copies into reports/ as well
    shutil.copy2(results_csv, reports_dir / "results.csv")
    if results_png and results_png.is_file():
        shutil.copy2(results_png, reports_dir / "results.png")

    print(f"Extracted authentic {train_summary['epochs_completed']}-epoch training log -> {results_csv}")
    if results_png:
        print(f"Generated Ultralytics training curves plot -> {results_png}")

    # 2. Remove any stale plots from earlier replaced checkpoints so zero mismatch exists
    for stale_name in ("BoxPR_curve.png", "confusion_matrix.png"):
        stale_p = eval_dir / stale_name
        if stale_p.is_file():
            stale_p.unlink()

    # 3. Build 4-class evaluation summary & per-class chart from verified source_metrics.json
    raw_metrics = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    overall_11s = next(m for m in raw_metrics["test_overall"] if m["model"] == "yolo11s")
    per_cls_11s = [c for c in raw_metrics["test_per_class"] if c["model"] == "yolo11s"]

    code_to_id = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
    per_class = []
    for entry in per_cls_11s:
        p = float(entry["precision"])
        r = float(entry["recall"])
        per_class.append({
            "class_id": code_to_id[entry["cls"]],
            "class_code": entry["cls"],
            "class_name": entry["name"].title(),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": _f1(p, r),
            "map50": round(float(entry["AP50"]), 4),
            "map50_95": round(float(entry["AP50_95"]), 4),
        })
    per_class.sort(key=lambda x: x["class_id"])

    per_class_chart_path = _plot_per_class_chart(per_class, eval_dir)

    gpu_ms = float(overall_11s["gpu_ms_per_img"])
    eval_summary = {
        "provenance": f"https://huggingface.co/{REPO_ID} (Multi-country 4-class RDD2022 held-out test set)",
        "checkpoint_path": str(best_pt),
        "evaluated_split": "test",
        "model_name": "yolo11s",
        "epochs": train_summary["epochs_completed"],
        "batch_size": train_summary["batch_size"],
        "image_size": train_summary["image_size"],
        "precision": round(float(overall_11s["precision"]), 4),
        "recall": round(float(overall_11s["recall"]), 4),
        "f1_score": round(float(overall_11s["f1"]), 4),
        "map50": round(float(overall_11s["mAP50"]), 4),
        "map50_95": round(float(overall_11s["mAP50_95"]), 4),
        "inference_ms": round(gpu_ms, 3),
        "fps": round(1000.0 / max(gpu_ms, 0.01), 1),
        "per_class": per_class,
        "test_by_country": [c for c in raw_metrics.get("test_by_country", []) if c["model"] == "yolo11s"],
        "artifacts": {
            "results_csv": str(results_csv),
            "results_png": str(results_png) if results_png else None,
            "loss_curves_plot": train_summary.get("loss_curves_plot"),
            "per_class_chart": per_class_chart_path,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_eval = eval_dir / "evaluation_summary.json"
    out_eval.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    print(f"Saved complete 4-class evaluation summary -> {out_eval}")

    _sync_sqlite_experiment(ROOT / "outputs" / "roadguard.db", eval_summary, train_summary)
    print("Synchronized SQLite model_experiments table.")


if __name__ == "__main__":
    main()
