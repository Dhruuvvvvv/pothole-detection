"""Dataset summary and GT sample reader reading real generated artifacts from outputs/dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import yaml

from backend.app.core.config import settings
from backend.app.utils.file_validation import path_to_static_url
from ml.constants import TARGET_CLASS_SPECS
from ml.dataset.kaggle_downloader import check_kaggle_credentials


def get_dataset_explorer_summary() -> Dict[str, Any]:
    ds_dir = Path(settings.dataset_output_dir).resolve()
    summary_csv = ds_dir / "dataset_summary.csv"
    class_csv = ds_dir / "class_distribution.csv"
    issues_csv = ds_dir / "annotation_issues.csv"
    config_yaml = ds_dir / "dataset_config.yaml"
    samples_dir = ds_dir / "samples"

    auth_status = check_kaggle_credentials()
    target_schema = [
        {
            "class_id": s.class_id,
            "code": s.code,
            "name": s.name,
            "description": s.description,
            "color": s.hex_color,
        }
        for s in TARGET_CLASS_SPECS
    ]

    kaggle_mirror_notes = {
        "primary_dataset_slug": settings.kaggle_dataset_slug,
        "kaggle_url": "https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022",
        "official_repo_url": "https://github.com/sekilab/RoadDamageDetector",
        "kaggle_credentials_configured": auth_status.configured,
        "kaggle_auth_method": auth_status.method,
        "important_schema_finding": (
            "The Kaggle mirror `aliabdelmenam/rdd-2022` uses YOLO TXT labels with 5 source classes: "
            "0='longitudinal crack' (D00), 1='transverse crack' (D10), 2='alligator crack' (D20), "
            "3='other corruption' (non-target), and 4='Pothole' (D40). "
            "RoadGuard AI's dataset pipeline detects this discrepancy automatically and requires explicit "
            "confirmation (--confirm-exclude-non-target) to exclude class 3 and remap class 4 -> 3 (D40) cleanly."
        ),
    }

    if not summary_csv.is_file():
        return {
            "prepared": False,
            "dataset_dir": str(ds_dir),
            "target_classes": target_schema,
            "kaggle_info": kaggle_mirror_notes,
            "summary_metrics": {},
            "class_distribution": [],
            "annotation_issues": [],
            "sample_images": [],
            "config_yaml": None,
        }

    summary_metrics: Dict[str, Any] = {}
    with open(summary_csv, "r", encoding="utf-8") as sf:
        reader = csv.DictReader(sf)
        for row in reader:
            k = row.get("metric", "")
            v: Any = row.get("value", "")
            if isinstance(v, str):
                if v.isdigit():
                    v = int(v)
                else:
                    try:
                        v = float(v)
                    except ValueError:
                        if v.lower() in {"true", "false"}:
                            v = v.lower() == "true"
            summary_metrics[k] = v

    class_dist: List[Dict[str, Any]] = []
    if class_csv.is_file():
        with open(class_csv, "r", encoding="utf-8") as cf:
            for row in csv.DictReader(cf):
                class_dist.append({
                    "class_id": int(row["class_id"]),
                    "class_code": row["class_code"],
                    "class_name": row["class_name"],
                    "total_instances": int(row["total_instances"]),
                    "image_count": int(row["image_count"]),
                    "instance_share_pct": float(row["instance_share_pct"]),
                    "mean_box_area_norm": float(row["mean_box_area_norm"]),
                    "train_instances": int(row["train_instances"]),
                    "val_instances": int(row["val_instances"]),
                    "test_instances": int(row["test_instances"]),
                })

    issues_list: List[Dict[str, str]] = []
    if issues_csv.is_file():
        with open(issues_csv, "r", encoding="utf-8") as isf:
            for i, row in enumerate(csv.DictReader(isf)):
                if i >= 100:
                    break
                issues_list.append(dict(row))

    samples: List[Dict[str, str]] = []
    if samples_dir.is_dir():
        for sp in sorted(samples_dir.iterdir())[:16]:
            if sp.is_file() and sp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                url = path_to_static_url(sp)
                if url:
                    samples.append({"filename": sp.name, "url": url})

    yaml_content = None
    if config_yaml.is_file():
        yaml_content = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))

    return {
        "prepared": True,
        "dataset_dir": str(ds_dir),
        "target_classes": target_schema,
        "kaggle_info": kaggle_mirror_notes,
        "summary_metrics": summary_metrics,
        "class_distribution": class_dist,
        "annotation_issues": issues_list,
        "sample_images": samples,
        "config_yaml": yaml_content,
    }
