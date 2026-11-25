"""Unit and integration tests for Phase B — RDD2022 Dataset Pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from ml.dataset.exporter import export_yolo_dataset
from ml.dataset.inspector import inspect_dataset_directory
from ml.dataset.parsers import (
    DatasetClassDiscrepancyError,
    parse_dataset,
    validate_and_normalize_xyxy,
)
from ml.dataset.splitter import assign_leakage_safe_splits
from ml.dataset.validator import validate_parsed_dataset


def _make_road_image(path: Path, width: int = 320, height: int = 240, seed_offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(100 + seed_offset)
    img = rng.integers(85, 135, size=(height, width, 3), dtype=np.uint8)
    # Draw synthetic dark crack line so dHash varies across distinct frames
    cv2.line(img, (20 + seed_offset % 80, 30), (200, 190 - seed_offset % 60), (25, 25, 25), 3)
    cv2.imwrite(str(path), img)


def test_kaggle_rdd2022_five_class_discrepancy_is_caught_and_remapped(tmp_path: Path) -> None:
    """
    Verify that the 5-class Kaggle mirror (`aliabdelmenam/rdd-2022`) containing
    `3: other corruption` and `4: Pothole`:
      1. Raises DatasetClassDiscrepancyError by default (never silently maps 3 -> D40!)
      2. Cleanly excludes `other corruption` and maps `4 (Pothole) -> 3 (D40)` when confirmed.
    """
    ds = tmp_path / "kaggle_mirror"
    (ds / "train" / "images").mkdir(parents=True)
    (ds / "train" / "labels").mkdir(parents=True)

    data_yaml = {
        "names": [
            "longitudinal crack",
            "transverse crack",
            "alligator crack",
            "other corruption",
            "Pothole",
        ]
    }
    (ds / "data.yaml").write_text(yaml.safe_dump(data_yaml), encoding="utf-8")

    _make_road_image(ds / "train" / "images" / "sample_001.jpg", seed_offset=1)
    # Contains class 0 (D00), class 1 (D10), class 2 (D20), class 3 (other corruption!), and class 4 (Pothole -> D40)
    (ds / "train" / "labels" / "sample_001.txt").write_text(
        "0 0.25 0.50 0.12 0.30\n"
        "1 0.50 0.30 0.25 0.10\n"
        "2 0.60 0.60 0.20 0.20\n"
        "3 0.80 0.80 0.15 0.15\n"
        "4 0.40 0.70 0.18 0.18\n",
        encoding="utf-8",
    )

    inspection = inspect_dataset_directory(ds)
    assert inspection.detected_annotation_format == "YOLO_TXT"
    assert inspection.has_class_discrepancy is True
    assert "other corruption" in inspection.non_target_classes

    # Must refuse silent remapping
    with pytest.raises(DatasetClassDiscrepancyError):
        parse_dataset(ds, non_target_policy="raise")

    # Confirmed exclusion drops class 3 and remaps class 4 -> class_id 3 (D40)
    parsed = parse_dataset(ds, non_target_policy="exclude_confirmed")
    assert parsed.excluded_non_target_counts == {"other corruption": 1}
    assert len(parsed.records) == 1
    boxes = parsed.records[0].boxes
    assert len(boxes) == 4
    codes = [b.class_code for b in boxes]
    ids = [b.class_id for b in boxes]
    assert codes == ["D00", "D10", "D20", "D40"]
    assert ids == [0, 1, 2, 3]


def test_unverified_yolo_txt_without_metadata_raises(tmp_path: Path) -> None:
    ds = tmp_path / "unverified_yolo"
    (ds / "images").mkdir(parents=True)
    (ds / "labels").mkdir(parents=True)
    _make_road_image(ds / "images" / "img1.jpg", seed_offset=2)
    (ds / "labels" / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(DatasetClassDiscrepancyError):
        parse_dataset(ds, allow_missing_target_classes=True)


def test_voc_xml_parsing_leakage_split_and_export(tmp_path: Path) -> None:
    """Test Pascal VOC XML parsing, corrupt image handling, sequential leakage grouping, and 70/15/15 export."""
    ds = tmp_path / "rdd_voc"
    img_dir = ds / "India" / "train" / "images"
    ann_dir = ds / "India" / "train" / "annotations" / "xmls"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    classes_cycle = ["D00", "D10", "D20", "D40"]
    # Create 20 distinct road recording sequence blocks (2 frames per block = 40 valid images)
    for block in range(20):
        for offset in (1, 2):
            frame_no = block * 100 + offset
            stem = f"India_{frame_no:06d}"
            img_path = img_dir / f"{stem}.jpg"
            _make_road_image(img_path, width=400, height=300, seed_offset=frame_no)
            cls_code = classes_cycle[block % 4]
            xml_text = f"""<annotation>
                <filename>{stem}.jpg</filename>
                <size><width>400</width><height>300</height><depth>3</depth></size>
                <object>
                    <name>{cls_code}</name>
                    <bndbox><xmin>40</xmin><ymin>50</ymin><xmax>180</xmax><ymax>210</ymax></bndbox>
                </object>
            </annotation>"""
            (ann_dir / f"{stem}.xml").write_text(xml_text, encoding="utf-8")

    # Add 1 corrupt image and 1 invalid bbox XML to test quality auditing
    bad_img = img_dir / "India_999991.jpg"
    bad_img.write_bytes(b"NOT_A_VALID_JPEG")
    (ann_dir / "India_999991.xml").write_text("<annotation></annotation>", encoding="utf-8")

    _make_road_image(img_dir / "India_999992.jpg", width=400, height=300, seed_offset=999)
    (ann_dir / "India_999992.xml").write_text(
        """<annotation>
            <filename>India_999992.jpg</filename>
            <size><width>400</width><height>300</height></size>
            <object><name>D40</name><bndbox><xmin>250</xmin><ymin>200</ymin><xmax>100</xmax><ymax>50</ymax></bndbox></object>
        </annotation>""",
        encoding="utf-8",
    )

    parsed = parse_dataset(ds)
    val_summary = validate_parsed_dataset(parsed)
    assert val_summary.total_corrupt_images == 1
    assert val_summary.issue_counts_by_type.get("invalid_bbox", 0) == 1

    split_records, split_counts = assign_leakage_safe_splits(
        parsed.records,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        sequence_block_size=20,
    )

    # Verify zero leakage: adjacent frames in the same block MUST share the exact same split
    by_id = {r.image_id: r.assigned_split for r in split_records}
    for block in range(20):
        f1 = f"India_{block * 100 + 1:06d}"
        f2 = f"India_{block * 100 + 2:06d}"
        assert by_id[f1] == by_id[f2], f"Sequence leakage between {f1} and {f2}!"

    assert split_counts["train"] > 0 and split_counts["val"] > 0 and split_counts["test"] > 0

    out_dir = tmp_path / "exported_yolo"
    exported = export_yolo_dataset(parsed, split_records, val_summary, out_dir)
    for key in ("dataset_manifest", "dataset_summary", "class_distribution", "annotation_issues", "dataset_config"):
        assert Path(exported[key]).is_file()
    assert int(exported["samples_saved"]) > 0


def test_coco_json_parsing_and_bbox_normalization(tmp_path: Path) -> None:
    ds = tmp_path / "coco_ds"
    (ds / "images").mkdir(parents=True)
    for i in range(4):
        _make_road_image(ds / "images" / f"road_{i}.jpg", width=200, height=100, seed_offset=50 + i)

    coco_doc = {
        "images": [{"id": i + 1, "file_name": f"road_{i}.jpg", "width": 200, "height": 100} for i in range(4)],
        "categories": [
            {"id": 1, "name": "D00"},
            {"id": 2, "name": "D10"},
            {"id": 3, "name": "D20"},
            {"id": 4, "name": "D40"},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [20, 10, 60, 40]},
            {"id": 11, "image_id": 2, "category_id": 2, "bbox": [10, 20, 80, 30]},
            {"id": 12, "image_id": 3, "category_id": 3, "bbox": [30, 15, 50, 50]},
            {"id": 13, "image_id": 4, "category_id": 4, "bbox": [50, 25, 40, 40]},
        ],
    }
    (ds / "annotations.json").write_text(json.dumps(coco_doc), encoding="utf-8")

    parsed = parse_dataset(ds)
    assert parsed.annotation_format == "COCO_JSON"
    assert len(parsed.records) == 4
    b0 = parsed.records[0].boxes[0]
    assert b0.class_code == "D00"
    assert pytest.approx(b0.x_center_norm, rel=1e-4) == 50.0 / 200.0
    assert pytest.approx(b0.y_center_norm, rel=1e-4) == 30.0 / 100.0

    # Degenerate negative width box must fail normalization
    bad_res, err = validate_and_normalize_xyxy(100, 100, 90, 120, 200, 200)
    assert bad_res is None and err is not None
