"""Format-agnostic dataset directory structure and metadata inspector."""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ml.constants import (
    CANONICAL_LABEL_ALIASES,
    IMAGE_EXTENSIONS,
    TARGET_CODES,
    normalize_label_token,
)


@dataclass
class DirectorySummary:
    relative_path: str
    image_count: int
    xml_count: int
    txt_count: int
    json_count: int


@dataclass
class DatasetInspectionReport:
    root_dir: str
    exists: bool
    total_files: int = 0
    total_images: int = 0
    image_formats: Dict[str, int] = field(default_factory=dict)
    xml_count: int = 0
    yolo_txt_count: int = 0
    coco_json_count: int = 0
    detected_annotation_format: str = "NONE"
    yaml_metadata_files: List[str] = field(default_factory=list)
    discovered_source_classes: List[str] = field(default_factory=list)
    source_id_to_name: Dict[int, str] = field(default_factory=dict)
    target_coverage: Dict[str, bool] = field(default_factory=dict)
    non_target_classes: List[str] = field(default_factory=list)
    has_class_discrepancy: bool = False
    discrepancy_messages: List[str] = field(default_factory=list)
    directory_summaries: List[DirectorySummary] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_pbtxt_classes(pbtxt_path: Path) -> Dict[int, str]:
    """Parse TensorFlow Object Detection API label_map.pbtxt / crackLabelMap.txt."""
    text = pbtxt_path.read_text(encoding="utf-8", errors="ignore")
    mapping: Dict[int, str] = {}
    for block in re.findall(r"item\s*\{([^}]*)\}", text, flags=re.DOTALL):
        id_match = re.search(r"id\s*:\s*(\d+)", block)
        name_match = re.search(r"name\s*:\s*['\"]([^'\"]+)['\"]", block)
        if id_match and name_match:
            mapping[int(id_match.group(1))] = name_match.group(1).strip()
    return mapping


def inspect_dataset_directory(root_dir: Path | str) -> DatasetInspectionReport:
    """Inspect a dataset directory without making structural or class-ID assumptions."""
    root = Path(root_dir).resolve()
    if not root.exists():
        return DatasetInspectionReport(root_dir=str(root), exists=False)

    all_files = [p for p in root.rglob("*") if p.is_file()]
    image_files = [p for p in all_files if p.suffix.lower() in IMAGE_EXTENSIONS]
    image_stems = {p.stem for p in image_files}

    ext_counter = Counter(p.suffix.lower() for p in image_files)
    xml_files = [p for p in all_files if p.suffix.lower() == ".xml"]

    # Distinguish genuine YOLO label .txt files from classes.txt / README.txt / manifests
    ignore_txt_names = {"classes.txt", "readme.txt", "license.txt", "cracklabelmap.txt"}
    yolo_txt_files: List[Path] = []
    classes_txt_files: List[Path] = []
    pbtxt_files: List[Path] = []

    for p in all_files:
        low_name = p.name.lower()
        if low_name == "classes.txt":
            classes_txt_files.append(p)
        elif low_name.endswith(".pbtxt") or low_name == "cracklabelmap.txt":
            pbtxt_files.append(p)
        elif p.suffix.lower() == ".txt" and low_name not in ignore_txt_names:
            if p.stem in image_stems or any(part.lower() in {"labels", "label"} for part in p.parts):
                yolo_txt_files.append(p)

    # Check JSON files for COCO structure (`images` + `annotations` + `categories`)
    coco_json_files: List[Path] = []
    coco_categories: Dict[int, str] = {}
    for p in all_files:
        if p.suffix.lower() == ".json" and p.stat().st_size < 250 * 1024 * 1024:
            try:
                data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(data, dict) and {"images", "annotations", "categories"}.issubset(data.keys()):
                    coco_json_files.append(p)
                    for cat in data.get("categories", []):
                        if "id" in cat and "name" in cat:
                            coco_categories[int(cat["id"])] = str(cat["name"])
            except Exception:
                continue

    # Check YAML metadata files (`data.yaml`, `dataset.yaml`, `dataset_config.yaml`)
    yaml_files: List[Path] = []
    yaml_id_to_name: Dict[int, str] = {}
    for p in all_files:
        if p.suffix.lower() in {".yaml", ".yml"}:
            try:
                obj = yaml.safe_load(p.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(obj, dict) and "names" in obj:
                    yaml_files.append(p)
                    names = obj["names"]
                    if isinstance(names, dict) and not yaml_id_to_name:
                        for k, v in names.items():
                            yaml_id_to_name[int(k)] = str(v)
                    elif isinstance(names, list) and not yaml_id_to_name:
                        for idx, v in enumerate(names):
                            yaml_id_to_name[idx] = str(v)
            except Exception:
                continue

    # Determine dominant annotation format
    present_formats = []
    if xml_files:
        present_formats.append("VOC_XML")
    if yolo_txt_files:
        present_formats.append("YOLO_TXT")
    if coco_json_files:
        present_formats.append("COCO_JSON")

    if len(present_formats) == 0:
        detected_format = "NONE"
    elif len(present_formats) == 1:
        detected_format = present_formats[0]
    else:
        # Prefer original rich XML if mixed, or primary format with largest count
        counts = {
            "VOC_XML": len(xml_files),
            "YOLO_TXT": len(yolo_txt_files),
            "COCO_JSON": len(coco_json_files),
        }
        detected_format = max(present_formats, key=lambda f: counts[f])

    # Gather actual class names from annotations & metadata
    source_id_to_name: Dict[int, str] = {}
    discovered_names: List[str] = []

    if yaml_id_to_name:
        source_id_to_name = dict(sorted(yaml_id_to_name.items()))
        discovered_names = list(source_id_to_name.values())
    elif classes_txt_files:
        lines = [
            ln.strip()
            for ln in classes_txt_files[0].read_text(encoding="utf-8", errors="ignore").splitlines()
            if ln.strip()
        ]
        source_id_to_name = {i: name for i, name in enumerate(lines)}
        discovered_names = lines
    elif coco_categories:
        source_id_to_name = dict(sorted(coco_categories.items()))
        discovered_names = list(source_id_to_name.values())
    elif pbtxt_files and not xml_files:
        source_id_to_name = _parse_pbtxt_classes(pbtxt_files[0])
        discovered_names = list(source_id_to_name.values())

    if xml_files:
        # Inspect XML <object><name> tags directly (sample up to 2000 XMLs for fast inspection; full pass in parser)
        xml_names_found: set[str] = set()
        for xp in xml_files[:2000]:
            try:
                tree_root = ET.parse(xp).getroot()
                for obj in tree_root.findall("object"):
                    nm = (obj.findtext("name") or "").strip()
                    if nm:
                        xml_names_found.add(nm)
            except Exception:
                continue
        if xml_names_found:
            discovered_names = sorted(xml_names_found)

    # Check target class coverage & discrepancies
    mapped_targets: set[str] = set()
    non_target_classes: List[str] = []
    for raw_name in discovered_names:
        norm = normalize_label_token(raw_name)
        canonical = CANONICAL_LABEL_ALIASES.get(norm)
        if canonical:
            mapped_targets.add(canonical)
        else:
            non_target_classes.append(raw_name)

    target_coverage = {code: (code in mapped_targets) for code in TARGET_CODES}
    discrepancy_messages: List[str] = []

    if detected_format == "YOLO_TXT" and not source_id_to_name:
        # Scan sample TXT files to see what numeric IDs exist
        observed_ids: set[int] = set()
        for tp in yolo_txt_files[:500]:
            for line in tp.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    observed_ids.add(int(parts[0]))
        discrepancy_messages.append(
            f"YOLO TXT annotations found (observed numeric IDs: {sorted(observed_ids)}) WITHOUT a data.yaml or classes.txt file. "
            "Numeric IDs cannot be safely assumed to map to D00/D10/D20/D40 without explicit confirmation."
        )

    if non_target_classes:
        discrepancy_messages.append(
            f"Dataset contains non-target class labels {non_target_classes} alongside target classes. "
            "Explicit confirmation is required to exclude non-target annotations and remap target indices cleanly."
        )

    missing_targets = [code for code, present in target_coverage.items() if not present]
    if discovered_names and missing_targets:
        discrepancy_messages.append(
            f"Expected target categories missing from discovered metadata: {missing_targets}."
        )

    # Build per-directory summary
    dir_summaries: List[DirectorySummary] = []
    candidate_dirs = [root] + sorted(p for p in root.rglob("*") if p.is_dir())
    for d in candidate_dirs[:80]:
        direct_files = [f for f in d.iterdir() if f.is_file()]
        im_c = sum(1 for f in direct_files if f.suffix.lower() in IMAGE_EXTENSIONS)
        xm_c = sum(1 for f in direct_files if f.suffix.lower() == ".xml")
        tx_c = sum(1 for f in direct_files if f.suffix.lower() == ".txt" and f.name.lower() not in ignore_txt_names)
        js_c = sum(1 for f in direct_files if f.suffix.lower() == ".json")
        if im_c or xm_c or tx_c or js_c:
            rel = "." if d == root else str(d.relative_to(root)).replace("\\", "/")
            dir_summaries.append(
                DirectorySummary(
                    relative_path=rel,
                    image_count=im_c,
                    xml_count=xm_c,
                    txt_count=tx_c,
                    json_count=js_c,
                )
            )

    return DatasetInspectionReport(
        root_dir=str(root),
        exists=True,
        total_files=len(all_files),
        total_images=len(image_files),
        image_formats=dict(ext_counter),
        xml_count=len(xml_files),
        yolo_txt_count=len(yolo_txt_files),
        coco_json_count=len(coco_json_files),
        detected_annotation_format=detected_format,
        yaml_metadata_files=[str(p.relative_to(root)).replace("\\", "/") for p in yaml_files],
        discovered_source_classes=discovered_names,
        source_id_to_name=source_id_to_name,
        target_coverage=target_coverage,
        non_target_classes=non_target_classes,
        has_class_discrepancy=bool(discrepancy_messages),
        discrepancy_messages=discrepancy_messages,
        directory_summaries=dir_summaries,
    )
