"""Dataset quality validation: duplicate detection, class distribution, imbalance, and issue auditing."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ml.constants import CODE_TO_SPEC, TARGET_CODES
from ml.dataset.parsers import AnnotationIssue, ImageAnnotationRecord, ParsedDataset


@dataclass
class ClassStat:
    class_id: int
    class_code: str
    class_name: str
    instance_count: int
    image_count: int
    instance_share_pct: float
    mean_box_area_norm: float


@dataclass
class DatasetValidationSummary:
    total_valid_labelled_images: int
    total_unlabelled_images: int
    total_corrupt_images: int
    total_empty_background_images: int
    total_damaged_images: int
    total_bounding_boxes: int
    exact_duplicate_groups: int
    exact_duplicate_images: int
    near_duplicate_groups: int
    issue_counts_by_type: Dict[str, int]
    excluded_non_target_counts: Dict[str, int]
    class_stats: List[ClassStat]
    imbalance_ratio_max_to_min: Optional[float]
    normalized_entropy_balance: float
    is_imbalanced: bool
    imbalance_notes: str
    image_resolutions_top5: List[Tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(file_path: Path, hash_size: int = 8) -> Optional[str]:
    """Compute 64-bit horizontal gradient perceptual hash (dHash) for near-duplicate grouping."""
    try:
        with Image.open(file_path) as im:
            gray = im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = list(gray.getdata())
        bits = []
        w = hash_size + 1
        for row in range(hash_size):
            offset = row * w
            for col in range(hash_size):
                bits.append(1 if pixels[offset + col] > pixels[offset + col + 1] else 0)
        val = 0
        for b in bits:
            val = (val << 1) | b
        return f"{val:016x}"
    except Exception:
        return None


def validate_parsed_dataset(
    parsed: ParsedDataset,
    compute_hashes: bool = True,
    imbalance_threshold: float = 2.0,
) -> DatasetValidationSummary:
    """Audit parsed records for duplicates, corrupt files, empty annotations, and class imbalance."""
    valid_records: List[ImageAnnotationRecord] = [r for r in parsed.records if not r.is_corrupt]
    corrupt_count = sum(1 for r in parsed.records if r.is_corrupt)

    sha_groups: Dict[str, List[ImageAnnotationRecord]] = defaultdict(list)
    dhash_groups: Dict[str, List[ImageAnnotationRecord]] = defaultdict(list)

    if compute_hashes:
        for rec in valid_records:
            rec.sha256 = compute_sha256(rec.image_path)
            sha_groups[rec.sha256].append(rec)
            rec.dhash = compute_dhash(rec.image_path)
            if rec.dhash:
                dhash_groups[rec.dhash].append(rec)

    exact_dup_groups = 0
    exact_dup_images = 0
    for sha, grp in sha_groups.items():
        if len(grp) > 1:
            exact_dup_groups += 1
            exact_dup_images += len(grp) - 1
            anchor = grp[0].image_path
            for dup in grp[1:]:
                parsed.issues.append(
                    AnnotationIssue(
                        image_path=str(dup.image_path),
                        annotation_path=str(dup.annotation_path or ""),
                        issue_type="exact_duplicate_image",
                        severity="warning",
                        details=f"Exact SHA-256 duplicate of {anchor.name}",
                    )
                )

    near_dup_groups = sum(1 for grp in dhash_groups.values() if len(grp) > 1)

    instance_counts = Counter()
    images_per_class: Dict[str, set[str]] = {code: set() for code in TARGET_CODES}
    area_sums: Dict[str, float] = defaultdict(float)
    resolutions = Counter()

    empty_bg = 0
    damaged_imgs = 0
    total_boxes = 0

    for rec in valid_records:
        resolutions[f"{rec.width}x{rec.height}"] += 1
        if not rec.boxes:
            empty_bg += 1
        else:
            damaged_imgs += 1
        for box in rec.boxes:
            total_boxes += 1
            instance_counts[box.class_code] += 1
            images_per_class[box.class_code].add(str(rec.image_path))
            area_sums[box.class_code] += box.area_norm

    class_stats: List[ClassStat] = []
    non_zero_counts: List[int] = []

    for code in TARGET_CODES:
        spec = CODE_TO_SPEC[code]
        cnt = instance_counts[code]
        if cnt > 0:
            non_zero_counts.append(cnt)
        share = (100.0 * cnt / total_boxes) if total_boxes > 0 else 0.0
        mean_area = (area_sums[code] / cnt) if cnt > 0 else 0.0
        class_stats.append(
            ClassStat(
                class_id=spec.class_id,
                class_code=code,
                class_name=spec.name,
                instance_count=cnt,
                image_count=len(images_per_class[code]),
                instance_share_pct=round(share, 2),
                mean_box_area_norm=round(mean_area, 5),
            )
        )

    if len(non_zero_counts) == len(TARGET_CODES) and min(non_zero_counts) > 0:
        imbalance_ratio: Optional[float] = round(max(non_zero_counts) / min(non_zero_counts), 2)
    else:
        imbalance_ratio = None

    if total_boxes > 0:
        entropy = 0.0
        for code in TARGET_CODES:
            p = instance_counts[code] / total_boxes
            if p > 0:
                entropy -= p * math.log(p)
        norm_entropy = round(entropy / math.log(len(TARGET_CODES)), 4)
    else:
        norm_entropy = 0.0

    missing_classes = [code for code in TARGET_CODES if instance_counts[code] == 0]
    is_imbalanced = bool(missing_classes) or (imbalance_ratio is not None and imbalance_ratio >= imbalance_threshold)

    if missing_classes:
        notes = f"Zero instances present for classes: {missing_classes}."
    elif imbalance_ratio is not None and is_imbalanced:
        dominant = max(class_stats, key=lambda c: c.instance_count)
        minority = min(class_stats, key=lambda c: c.instance_count)
        notes = (
            f"Class imbalance detected (max/min ratio = {imbalance_ratio}x): "
            f"{dominant.class_code} ({dominant.instance_count}) vs {minority.class_code} ({minority.instance_count})."
        )
    else:
        notes = f"Balanced distribution across classes (max/min ratio = {imbalance_ratio}x)."

    issue_counts = Counter(iss.issue_type for iss in parsed.issues)

    return DatasetValidationSummary(
        total_valid_labelled_images=len(valid_records),
        total_unlabelled_images=len(parsed.unlabelled_images),
        total_corrupt_images=corrupt_count,
        total_empty_background_images=empty_bg,
        total_damaged_images=damaged_imgs,
        total_bounding_boxes=total_boxes,
        exact_duplicate_groups=exact_dup_groups,
        exact_duplicate_images=exact_dup_images,
        near_duplicate_groups=near_dup_groups,
        issue_counts_by_type=dict(issue_counts),
        excluded_non_target_counts=dict(parsed.excluded_non_target_counts),
        class_stats=class_stats,
        imbalance_ratio_max_to_min=imbalance_ratio,
        normalized_entropy_balance=norm_entropy,
        is_imbalanced=is_imbalanced,
        imbalance_notes=notes,
        image_resolutions_top5=resolutions.most_common(5),
    )
