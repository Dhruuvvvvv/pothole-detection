"""YOLO dataset structure exporter, CSV/YAML manifest writer, and GT visual sample generator."""

from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import yaml

from ml.constants import CODE_TO_SPEC, ID_TO_SPEC, TARGET_CODES
from ml.dataset.parsers import ImageAnnotationRecord, ParsedDataset
from ml.dataset.validator import DatasetValidationSummary


def _draw_gt_boxes_on_image(image_path: Path, rec: ImageAnnotationRecord) -> Optional[object]:
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    out = img.copy()
    for box in rec.boxes:
        spec = ID_TO_SPEC[box.class_id]
        p1 = (int(round(box.xmin)), int(round(box.ymin)))
        p2 = (int(round(box.xmax)), int(round(box.ymax)))
        cv2.rectangle(out, p1, p2, spec.bgr_color, 2)
        label = f"{spec.code}: {spec.name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(th + 6, p1[1])
        cv2.rectangle(out, (p1[0], ty - th - 6), (p1[0] + tw + 8, ty), spec.bgr_color, -1)
        cv2.putText(
            out,
            label,
            (p1[0] + 4, ty - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def export_yolo_dataset(
    parsed: ParsedDataset,
    split_records: List[ImageAnnotationRecord],
    validation_summary: DatasetValidationSummary,
    output_dir: Path | str,
    copy_images: bool = True,
    max_visual_samples: int = 12,
) -> Dict[str, str]:
    """
    Export standardized YOLO directory layout alongside CSV/YAML reports and visual GT samples.

    Generates:
      - images/{train,val,test}/...
      - labels/{train,val,test}/...
      - dataset_manifest.csv
      - dataset_summary.csv
      - class_distribution.csv
      - annotation_issues.csv
      - dataset_config.yaml
      - samples/*.jpg
    """
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    for sp in ("train", "val", "test"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    samples_dir = out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "dataset_manifest.csv"
    summary_path = out / "dataset_summary.csv"
    class_dist_path = out / "class_distribution.csv"
    issues_path = out / "annotation_issues.csv"
    config_yaml_path = out / "dataset_config.yaml"

    split_image_counts = Counter()
    split_box_counts: Dict[str, Counter] = {s: Counter() for s in ("train", "val", "test")}

    with open(manifest_path, "w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow([
            "image_id",
            "assigned_split",
            "source_split",
            "group_id",
            "width",
            "height",
            "num_boxes",
            "classes_present",
            "sha256",
            "dhash",
            "exported_image_path",
            "exported_label_path",
            "source_image_path",
        ])

        for rec in split_records:
            sp = rec.assigned_split or "train"
            split_image_counts[sp] += 1

            # Disambiguate duplicate stems across country subfolders if needed
            dest_img_name = f"{rec.image_id}{rec.image_path.suffix.lower()}"
            dest_img = out / "images" / sp / dest_img_name
            if dest_img.exists() and dest_img.resolve() != rec.image_path.resolve():
                parent_tag = rec.image_path.parent.parent.name or rec.image_path.parent.name
                dest_img_name = f"{parent_tag}_{rec.image_id}{rec.image_path.suffix.lower()}"
                dest_img = out / "images" / sp / dest_img_name

            dest_lbl = out / "labels" / sp / f"{Path(dest_img_name).stem}.txt"

            if copy_images:
                shutil.copy2(rec.image_path, dest_img)

            yolo_lines = []
            present_codes = sorted({b.class_code for b in rec.boxes})
            for b in rec.boxes:
                split_box_counts[sp][b.class_code] += 1
                yolo_lines.append(
                    f"{b.class_id} {b.x_center_norm:.6f} {b.y_center_norm:.6f} {b.width_norm:.6f} {b.height_norm:.6f}"
                )
            dest_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

            writer.writerow([
                rec.image_id,
                sp,
                rec.source_split or "",
                rec.group_id or "",
                rec.width,
                rec.height,
                len(rec.boxes),
                "|".join(present_codes),
                rec.sha256 or "",
                rec.dhash or "",
                str(dest_img.relative_to(out)).replace("\\", "/") if copy_images else "",
                str(dest_lbl.relative_to(out)).replace("\\", "/"),
                str(rec.image_path),
            ])

    # Write dataset_summary.csv
    total_exported = sum(split_image_counts.values())
    with open(summary_path, "w", newline="", encoding="utf-8") as sf:
        writer = csv.writer(sf)
        writer.writerow(["metric", "value"])
        rows = [
            ("source_root", str(parsed.root_dir)),
            ("detected_annotation_format", parsed.annotation_format),
            ("total_exported_images", total_exported),
            ("train_images", split_image_counts["train"]),
            ("val_images", split_image_counts["val"]),
            ("test_images", split_image_counts["test"]),
            ("train_pct", round(100.0 * split_image_counts["train"] / max(1, total_exported), 2)),
            ("val_pct", round(100.0 * split_image_counts["val"] / max(1, total_exported), 2)),
            ("test_pct", round(100.0 * split_image_counts["test"] / max(1, total_exported), 2)),
            ("total_bounding_boxes", validation_summary.total_bounding_boxes),
            ("damaged_images", validation_summary.total_damaged_images),
            ("empty_background_images", validation_summary.total_empty_background_images),
            ("unlabelled_images", validation_summary.total_unlabelled_images),
            ("corrupt_images", validation_summary.total_corrupt_images),
            ("exact_duplicate_images_detected", validation_summary.exact_duplicate_images),
            ("imbalance_ratio_max_to_min", validation_summary.imbalance_ratio_max_to_min or "N/A"),
            ("normalized_entropy_balance", validation_summary.normalized_entropy_balance),
            ("is_imbalanced", validation_summary.is_imbalanced),
            ("imbalance_notes", validation_summary.imbalance_notes),
        ]
        writer.writerows(rows)

    # Write class_distribution.csv
    with open(class_dist_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "class_id",
            "class_code",
            "class_name",
            "total_instances",
            "image_count",
            "instance_share_pct",
            "mean_box_area_norm",
            "train_instances",
            "val_instances",
            "test_instances",
        ])
        for cs in validation_summary.class_stats:
            writer.writerow([
                cs.class_id,
                cs.class_code,
                cs.class_name,
                cs.instance_count,
                cs.image_count,
                cs.instance_share_pct,
                cs.mean_box_area_norm,
                split_box_counts["train"][cs.class_code],
                split_box_counts["val"][cs.class_code],
                split_box_counts["test"][cs.class_code],
            ])

    # Write annotation_issues.csv
    with open(issues_path, "w", newline="", encoding="utf-8") as isf:
        writer = csv.writer(isf)
        writer.writerow(["image_path", "annotation_path", "issue_type", "severity", "details"])
        for iss in parsed.issues:
            writer.writerow([iss.image_path, iss.annotation_path, iss.issue_type, iss.severity, iss.details])

    # Write Ultralytics YOLO dataset_config.yaml
    yolo_cfg = {
        "path": str(out).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(TARGET_CODES),
        "names": {CODE_TO_SPEC[code].class_id: code for code in TARGET_CODES},
        "descriptions": {CODE_TO_SPEC[code].code: CODE_TO_SPEC[code].name for code in TARGET_CODES},
    }
    config_yaml_path.write_text(yaml.safe_dump(yolo_cfg, sort_keys=False), encoding="utf-8")

    # Save visual ground-truth sample images
    damaged_records = [r for r in split_records if r.boxes]
    saved_samples: List[str] = []
    for rec in damaged_records[:max_visual_samples]:
        vis = _draw_gt_boxes_on_image(rec.image_path, rec)
        if vis is not None:
            sample_out = samples_dir / f"gt_{rec.assigned_split}_{rec.image_id}.jpg"
            cv2.imwrite(str(sample_out), vis)
            saved_samples.append(str(sample_out))

    return {
        "output_dir": str(out),
        "dataset_manifest": str(manifest_path),
        "dataset_summary": str(summary_path),
        "class_distribution": str(class_dist_path),
        "annotation_issues": str(issues_path),
        "dataset_config": str(config_yaml_path),
        "samples_saved": str(len(saved_samples)),
    }
