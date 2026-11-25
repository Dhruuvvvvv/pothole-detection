"""Multi-format annotation parsers (Pascal VOC XML, YOLO TXT, COCO JSON) with strict class verification."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from ml.constants import (
    CANONICAL_LABEL_ALIASES,
    CODE_TO_ID,
    IMAGE_EXTENSIONS,
    TARGET_CODES,
    normalize_label_token,
)
from ml.dataset.inspector import DatasetInspectionReport, inspect_dataset_directory


class DatasetClassDiscrepancyError(ValueError):
    """Raised when dataset classes diverge from {D00, D10, D20, D40} without explicit user confirmation."""

    def __init__(self, message: str, inspection: Optional[DatasetInspectionReport] = None):
        super().__init__(message)
        self.inspection = inspection


@dataclass
class AnnotationBox:
    class_id: int
    class_code: str
    source_label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    x_center_norm: float
    y_center_norm: float
    width_norm: float
    height_norm: float

    @property
    def area_px(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)

    @property
    def area_norm(self) -> float:
        return self.width_norm * self.height_norm


@dataclass
class AnnotationIssue:
    image_path: str
    annotation_path: str
    issue_type: str
    severity: str
    details: str


@dataclass
class ImageAnnotationRecord:
    image_id: str
    image_path: Path
    annotation_path: Optional[Path]
    width: int
    height: int
    source_split: Optional[str] = None
    assigned_split: Optional[str] = None
    group_id: Optional[str] = None
    sha256: Optional[str] = None
    dhash: Optional[str] = None
    is_corrupt: bool = False
    has_annotation_file: bool = False
    boxes: List[AnnotationBox] = field(default_factory=list)


@dataclass
class ParsedDataset:
    root_dir: Path
    annotation_format: str
    records: List[ImageAnnotationRecord]
    unlabelled_images: List[Path]
    issues: List[AnnotationIssue]
    excluded_non_target_counts: Dict[str, int]
    inspection: DatasetInspectionReport


def _detect_source_split_from_path(path: Path) -> Optional[str]:
    low_parts = [p.lower() for p in path.parts]
    for token, canonical in (
        ("train", "train"),
        ("training", "train"),
        ("val", "val"),
        ("valid", "val"),
        ("validation", "val"),
        ("test", "test"),
        ("testing", "test"),
    ):
        if token in low_parts:
            return canonical
    return None


def _verify_and_read_image_size(image_path: Path) -> Tuple[bool, int, int, Optional[str]]:
    """Verify image integrity via PIL decode and return (ok, width, height, error_msg)."""
    try:
        with Image.open(image_path) as im:
            im.verify()
        with Image.open(image_path) as im2:
            im2.load()
            w, h = im2.size
        if w <= 0 or h <= 0:
            return False, 0, 0, f"Non-positive dimensions ({w}x{h})"
        return True, int(w), int(h), None
    except Exception as exc:
        return False, 0, 0, str(exc)


def validate_and_normalize_xyxy(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int,
    min_box_px: float = 2.0,
) -> Tuple[Optional[Tuple[float, float, float, float, float, float, float, float]], Optional[str]]:
    """Validate pixel bounding box and convert to normalized YOLO (xc, yc, w, h) coordinates."""
    vals = (xmin, ymin, xmax, ymax)
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        return None, "Bounding box contains NaN or Inf coordinates"

    # Reject boxes completely outside the image frame
    if xmax <= 0 or ymax <= 0 or xmin >= img_w or ymin >= img_h:
        return None, f"Box ({xmin:.1f},{ymin:.1f},{xmax:.1f},{ymax:.1f}) lies completely outside ({img_w}x{img_h})"

    # Clamp minor annotation overhang at borders
    c_xmin = max(0.0, min(float(img_w), float(xmin)))
    c_ymin = max(0.0, min(float(img_h), float(ymin)))
    c_xmax = max(0.0, min(float(img_w), float(xmax)))
    c_ymax = max(0.0, min(float(img_h), float(ymax)))

    bw_px = c_xmax - c_xmin
    bh_px = c_ymax - c_ymin
    if bw_px < min_box_px or bh_px < min_box_px:
        return None, f"Degenerate bounding box after clamping: width={bw_px:.2f}px, height={bh_px:.2f}px"

    xc = ((c_xmin + c_xmax) / 2.0) / float(img_w)
    yc = ((c_ymin + c_ymax) / 2.0) / float(img_h)
    wn = bw_px / float(img_w)
    hn = bh_px / float(img_h)
    return (c_xmin, c_ymin, c_xmax, c_ymax, xc, yc, wn, hn), None


def validate_and_denormalize_yolo_box(
    xc: float,
    yc: float,
    wn: float,
    hn: float,
    img_w: int,
    img_h: int,
    min_box_px: float = 2.0,
) -> Tuple[Optional[Tuple[float, float, float, float, float, float, float, float]], Optional[str]]:
    """Validate normalized YOLO box (xc, yc, w, h) and compute pixel + clamped normalized coords."""
    vals = (xc, yc, wn, hn)
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        return None, "YOLO box contains NaN or Inf values"
    if wn <= 0.0 or hn <= 0.0 or wn > 1.05 or hn > 1.05:
        return None, f"Invalid normalized width/height: w={wn:.4f}, h={hn:.4f}"
    if xc < -0.05 or xc > 1.05 or yc < -0.05 or yc > 1.05:
        return None, f"Normalized center out of range [0,1]: xc={xc:.4f}, yc={yc:.4f}"

    xmin = (xc - wn / 2.0) * img_w
    ymin = (yc - hn / 2.0) * img_h
    xmax = (xc + wn / 2.0) * img_w
    ymax = (yc + hn / 2.0) * img_h
    return validate_and_normalize_xyxy(xmin, ymin, xmax, ymax, img_w, img_h, min_box_px=min_box_px)


def _choose_matching_annotation(
    img_path: Path,
    candidates_by_stem: Dict[str, List[Path]],
) -> Optional[Path]:
    candidates = candidates_by_stem.get(img_path.stem, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Prefer candidate sharing the most path components (e.g. India/train/images <-> India/train/annotations)
    ranked = sorted(
        candidates,
        key=lambda p: (len(set(p.parts) ^ set(img_path.parts)), len(str(p))),
    )
    return ranked[0]


def parse_dataset(
    root_dir: Path | str,
    annotation_format: Optional[str] = None,
    non_target_policy: str = "raise",
    explicit_yolo_id_map: Optional[Dict[int, str]] = None,
    allow_missing_target_classes: bool = False,
) -> ParsedDataset:
    """
    Parse a road damage dataset in VOC_XML, YOLO_TXT, or COCO_JSON format.

    Parameters
    ----------
    root_dir:
        Dataset root directory.
    annotation_format:
        Optional override ('VOC_XML', 'YOLO_TXT', 'COCO_JSON'). Auto-detected if None.
    non_target_policy:
        - 'raise' (default): Raise DatasetClassDiscrepancyError if non-target classes
          (e.g. 'other corruption' at class 3 in `aliabdelmenam/rdd-2022` or 'D44'/'Repair' in XML)
          are present, requiring explicit confirmation rather than silently mislabelling or dropping.
        - 'exclude_confirmed': Exclude confirmed non-target bounding boxes while mapping verified
          target classes cleanly to {0: D00, 1: D10, 2: D20, 3: D40}.
    explicit_yolo_id_map:
        Optional verified dictionary mapping source YOLO integer class IDs to label strings,
        required ONLY if a YOLO TXT dataset has no `data.yaml` / `classes.txt` metadata.
    allow_missing_target_classes:
        Allow parsing subsets/smoke-test datasets that do not contain all 4 classes.
    """
    if non_target_policy not in {"raise", "exclude_confirmed"}:
        raise ValueError("non_target_policy must be 'raise' or 'exclude_confirmed'")

    root = Path(root_dir).resolve()
    inspection = inspect_dataset_directory(root)
    if not inspection.exists:
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    fmt = annotation_format or inspection.detected_annotation_format
    if fmt == "NONE":
        raise ValueError(f"No supported annotations (VOC_XML, YOLO_TXT, COCO_JSON) found under {root}")

    # Enforce missing-target check unless explicitly allowed
    if inspection.discovered_source_classes and not allow_missing_target_classes:
        missing = [code for code, ok in inspection.target_coverage.items() if not ok]
        if missing:
            raise DatasetClassDiscrepancyError(
                f"Dataset is missing expected target categories {missing}. "
                f"Discovered classes: {inspection.discovered_source_classes}",
                inspection=inspection,
            )

    # Enforce non-target discrepancy check before parsing
    if inspection.non_target_classes and non_target_policy == "raise":
        raise DatasetClassDiscrepancyError(
            "Class discrepancy detected: dataset contains non-target categories "
            f"{inspection.non_target_classes} (discovered classes: {inspection.discovered_source_classes}). "
            "Confirm exclusion by passing non_target_policy='exclude_confirmed' so non-target boxes "
            "are transparently excluded and target classes are safely mapped to 0=D00, 1=D10, 2=D20, 3=D40.",
            inspection=inspection,
        )

    all_files = [p for p in root.rglob("*") if p.is_file()]
    image_files = sorted(p for p in all_files if p.suffix.lower() in IMAGE_EXTENSIONS)

    issues: List[AnnotationIssue] = []
    excluded_counts: Dict[str, int] = defaultdict(int)
    records: List[ImageAnnotationRecord] = []
    unlabelled_images: List[Path] = []

    if fmt == "VOC_XML":
        xml_by_stem: Dict[str, List[Path]] = defaultdict(list)
        for p in all_files:
            if p.suffix.lower() == ".xml":
                xml_by_stem[p.stem].append(p)

        for img_path in image_files:
            src_split = _detect_source_split_from_path(img_path)
            ann_path = _choose_matching_annotation(img_path, xml_by_stem)
            ok, w, h, err = _verify_and_read_image_size(img_path)
            if not ok:
                issues.append(AnnotationIssue(str(img_path), str(ann_path or ""), "corrupt_image", "error", err or "Unreadable image"))
                records.append(
                    ImageAnnotationRecord(
                        image_id=img_path.stem,
                        image_path=img_path,
                        annotation_path=ann_path,
                        width=0,
                        height=0,
                        source_split=src_split,
                        is_corrupt=True,
                        has_annotation_file=ann_path is not None,
                    )
                )
                continue

            if ann_path is None:
                # Official RDD2022 `test/images` has no XML labels; separate from unexpected missing train labels
                unlabelled_images.append(img_path)
                if src_split != "test":
                    issues.append(
                        AnnotationIssue(str(img_path), "", "missing_annotation", "warning", "Image has no matching XML annotation file")
                    )
                continue

            boxes: List[AnnotationBox] = []
            try:
                xml_root = ET.parse(ann_path).getroot()
            except Exception as exc:
                issues.append(AnnotationIssue(str(img_path), str(ann_path), "malformed_xml", "error", str(exc)))
                continue

            size_el = xml_root.find("size")
            if size_el is not None:
                xml_w = int(float(size_el.findtext("width") or 0))
                xml_h = int(float(size_el.findtext("height") or 0))
                if (xml_w and xml_w != w) or (xml_h and xml_h != h):
                    issues.append(
                        AnnotationIssue(
                            str(img_path),
                            str(ann_path),
                            "dimension_mismatch",
                            "info",
                            f"XML declared ({xml_w}x{xml_h}) vs actual decoded image ({w}x{h}); using actual dimensions.",
                        )
                    )

            for obj in xml_root.findall("object"):
                raw_name = (obj.findtext("name") or "").strip()
                norm_name = normalize_label_token(raw_name)
                canonical = CANONICAL_LABEL_ALIASES.get(norm_name)
                if canonical is None:
                    if non_target_policy == "raise":
                        raise DatasetClassDiscrepancyError(
                            f"Unexpected XML label '{raw_name}' in {ann_path}. Confirm non_target_policy='exclude_confirmed'.",
                            inspection=inspection,
                        )
                    excluded_counts[raw_name] += 1
                    continue

                bnd = obj.find("bndbox")
                if bnd is None:
                    issues.append(AnnotationIssue(str(img_path), str(ann_path), "missing_bndbox", "error", f"Object '{raw_name}' missing <bndbox>"))
                    continue
                try:
                    xmin = float(bnd.findtext("xmin") or "nan")
                    ymin = float(bnd.findtext("ymin") or "nan")
                    xmax = float(bnd.findtext("xmax") or "nan")
                    ymax = float(bnd.findtext("ymax") or "nan")
                except ValueError as exc:
                    issues.append(AnnotationIssue(str(img_path), str(ann_path), "invalid_bbox_number", "error", str(exc)))
                    continue

                norm_res, box_err = validate_and_normalize_xyxy(xmin, ymin, xmax, ymax, w, h)
                if norm_res is None:
                    issues.append(AnnotationIssue(str(img_path), str(ann_path), "invalid_bbox", "error", box_err or "Invalid box"))
                    continue

                c_xmin, c_ymin, c_xmax, c_ymax, xc, yc, wn, hn = norm_res
                boxes.append(
                    AnnotationBox(
                        class_id=CODE_TO_ID[canonical],
                        class_code=canonical,
                        source_label=raw_name,
                        xmin=c_xmin,
                        ymin=c_ymin,
                        xmax=c_xmax,
                        ymax=c_ymax,
                        x_center_norm=xc,
                        y_center_norm=yc,
                        width_norm=wn,
                        height_norm=hn,
                    )
                )

            records.append(
                ImageAnnotationRecord(
                    image_id=img_path.stem,
                    image_path=img_path,
                    annotation_path=ann_path,
                    width=w,
                    height=h,
                    source_split=src_split,
                    has_annotation_file=True,
                    boxes=boxes,
                )
            )

    elif fmt == "YOLO_TXT":
        id_to_name = dict(inspection.source_id_to_name)
        if explicit_yolo_id_map is not None:
            id_to_name = dict(explicit_yolo_id_map)

        if not id_to_name:
            raise DatasetClassDiscrepancyError(
                "YOLO_TXT dataset lacks class metadata (data.yaml / classes.txt). "
                "Refusing to assume numeric IDs 0..3 map to D00/D10/D20/D40 without explicit_yolo_id_map.",
                inspection=inspection,
            )

        # Verify every mapped source name
        verified_id_to_code: Dict[int, Optional[str]] = {}
        for src_id, src_name in id_to_name.items():
            canonical = CANONICAL_LABEL_ALIASES.get(normalize_label_token(src_name))
            if canonical is None:
                if non_target_policy == "raise":
                    raise DatasetClassDiscrepancyError(
                        f"YOLO class ID {src_id} ('{src_name}') is not one of {TARGET_CODES}. "
                        "Pass non_target_policy='exclude_confirmed' to exclude it and remap target IDs cleanly.",
                        inspection=inspection,
                    )
                verified_id_to_code[src_id] = None
            else:
                verified_id_to_code[src_id] = canonical

        ignore_txt_names = {"classes.txt", "readme.txt", "license.txt", "cracklabelmap.txt"}
        txt_by_stem: Dict[str, List[Path]] = defaultdict(list)
        for p in all_files:
            if p.suffix.lower() == ".txt" and p.name.lower() not in ignore_txt_names:
                txt_by_stem[p.stem].append(p)

        for img_path in image_files:
            src_split = _detect_source_split_from_path(img_path)
            ann_path = _choose_matching_annotation(img_path, txt_by_stem)
            ok, w, h, err = _verify_and_read_image_size(img_path)
            if not ok:
                issues.append(AnnotationIssue(str(img_path), str(ann_path or ""), "corrupt_image", "error", err or "Unreadable image"))
                records.append(
                    ImageAnnotationRecord(
                        image_id=img_path.stem,
                        image_path=img_path,
                        annotation_path=ann_path,
                        width=0,
                        height=0,
                        source_split=src_split,
                        is_corrupt=True,
                        has_annotation_file=ann_path is not None,
                    )
                )
                continue

            if ann_path is None:
                unlabelled_images.append(img_path)
                issues.append(
                    AnnotationIssue(str(img_path), "", "missing_annotation", "warning", "Image has no matching YOLO .txt label file")
                )
                continue

            boxes = []
            lines = ann_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line_no, raw_line in enumerate(lines, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) != 5:
                    issues.append(
                        AnnotationIssue(
                            str(img_path),
                            str(ann_path),
                            "malformed_yolo_line",
                            "error",
                            f"Line {line_no}: expected 5 tokens '<cls> <xc> <yc> <w> <h>', got {len(parts)}",
                        )
                    )
                    continue
                try:
                    src_cls = int(parts[0])
                    xc, yc, wn, hn = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError as exc:
                    issues.append(AnnotationIssue(str(img_path), str(ann_path), "invalid_yolo_number", "error", f"Line {line_no}: {exc}"))
                    continue

                if src_cls not in verified_id_to_code:
                    if non_target_policy == "raise":
                        raise DatasetClassDiscrepancyError(
                            f"Undeclared numeric class ID {src_cls} encountered in {ann_path}:{line_no} (metadata keys: {sorted(id_to_name.keys())}).",
                            inspection=inspection,
                        )
                    issues.append(
                        AnnotationIssue(str(img_path), str(ann_path), "unknown_class_id", "error", f"Line {line_no}: undeclared class ID {src_cls}")
                    )
                    continue

                canonical = verified_id_to_code[src_cls]
                src_label = id_to_name.get(src_cls, str(src_cls))
                if canonical is None:
                    excluded_counts[src_label] += 1
                    continue

                denorm_res, box_err = validate_and_denormalize_yolo_box(xc, yc, wn, hn, w, h)
                if denorm_res is None:
                    issues.append(AnnotationIssue(str(img_path), str(ann_path), "invalid_bbox", "error", f"Line {line_no}: {box_err}"))
                    continue

                c_xmin, c_ymin, c_xmax, c_ymax, n_xc, n_yc, n_wn, n_hn = denorm_res
                boxes.append(
                    AnnotationBox(
                        class_id=CODE_TO_ID[canonical],
                        class_code=canonical,
                        source_label=src_label,
                        xmin=c_xmin,
                        ymin=c_ymin,
                        xmax=c_xmax,
                        ymax=c_ymax,
                        x_center_norm=n_xc,
                        y_center_norm=n_yc,
                        width_norm=n_wn,
                        height_norm=n_hn,
                    )
                )

            records.append(
                ImageAnnotationRecord(
                    image_id=img_path.stem,
                    image_path=img_path,
                    annotation_path=ann_path,
                    width=w,
                    height=h,
                    source_split=src_split,
                    has_annotation_file=True,
                    boxes=boxes,
                )
            )

    elif fmt == "COCO_JSON":
        coco_files = [
            p for p in all_files
            if p.suffix.lower() == ".json"
        ]
        image_lookup: Dict[str, Path] = {p.name: p for p in image_files}
        referenced_images: set[Path] = set()

        for cp in coco_files:
            try:
                coco = json.loads(cp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(coco, dict) or not {"images", "annotations", "categories"}.issubset(coco.keys()):
                continue

            cat_map: Dict[int, Optional[str]] = {}
            cat_raw_name: Dict[int, str] = {}
            for cat in coco["categories"]:
                cid = int(cat["id"])
                cname = str(cat["name"])
                cat_raw_name[cid] = cname
                canonical = CANONICAL_LABEL_ALIASES.get(normalize_label_token(cname))
                if canonical is None and non_target_policy == "raise":
                    raise DatasetClassDiscrepancyError(
                        f"COCO category '{cname}' (id={cid}) in {cp} is not in {TARGET_CODES}.",
                        inspection=inspection,
                    )
                cat_map[cid] = canonical

            anns_by_img: Dict[int, List[dict]] = defaultdict(list)
            for ann in coco["annotations"]:
                anns_by_img[int(ann["image_id"])].append(ann)

            src_split = _detect_source_split_from_path(cp)
            for img_entry in coco["images"]:
                fname = Path(str(img_entry["file_name"])).name
                img_path = image_lookup.get(fname)
                if img_path is None:
                    issues.append(AnnotationIssue(fname, str(cp), "missing_image", "error", f"COCO entry '{fname}' not found on disk"))
                    continue
                referenced_images.add(img_path)
                ok, w, h, err = _verify_and_read_image_size(img_path)
                if not ok:
                    issues.append(AnnotationIssue(str(img_path), str(cp), "corrupt_image", "error", err or "Unreadable image"))
                    records.append(
                        ImageAnnotationRecord(
                            image_id=img_path.stem,
                            image_path=img_path,
                            annotation_path=cp,
                            width=0,
                            height=0,
                            source_split=src_split,
                            is_corrupt=True,
                            has_annotation_file=True,
                        )
                    )
                    continue

                boxes = []
                for ann in anns_by_img.get(int(img_entry["id"]), []):
                    cid = int(ann["category_id"])
                    raw_label = cat_raw_name.get(cid, str(cid))
                    canonical = cat_map.get(cid)
                    if canonical is None:
                        excluded_counts[raw_label] += 1
                        continue
                    bbox = ann.get("bbox", [])
                    if len(bbox) != 4:
                        issues.append(AnnotationIssue(str(img_path), str(cp), "invalid_bbox", "error", f"COCO bbox must have 4 numbers: {bbox}"))
                        continue
                    bx, by, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    norm_res, box_err = validate_and_normalize_xyxy(bx, by, bx + bw, by + bh, w, h)
                    if norm_res is None:
                        issues.append(AnnotationIssue(str(img_path), str(cp), "invalid_bbox", "error", box_err or "Invalid COCO bbox"))
                        continue
                    c_xmin, c_ymin, c_xmax, c_ymax, xc, yc, wn, hn = norm_res
                    boxes.append(
                        AnnotationBox(
                            class_id=CODE_TO_ID[canonical],
                            class_code=canonical,
                            source_label=raw_label,
                            xmin=c_xmin,
                            ymin=c_ymin,
                            xmax=c_xmax,
                            ymax=c_ymax,
                            x_center_norm=xc,
                            y_center_norm=yc,
                            width_norm=wn,
                            height_norm=hn,
                        )
                    )

                records.append(
                    ImageAnnotationRecord(
                        image_id=img_path.stem,
                        image_path=img_path,
                        annotation_path=cp,
                        width=w,
                        height=h,
                        source_split=src_split,
                        has_annotation_file=True,
                        boxes=boxes,
                    )
                )

        for img_path in image_files:
            if img_path not in referenced_images:
                unlabelled_images.append(img_path)
    else:
        raise ValueError(f"Unsupported annotation_format: {fmt}")

    return ParsedDataset(
        root_dir=root,
        annotation_format=fmt,
        records=records,
        unlabelled_images=unlabelled_images,
        issues=issues,
        excluded_non_target_counts=dict(excluded_counts),
        inspection=inspection,
    )
