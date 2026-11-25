"""Leakage-safe 70/15/15 dataset splitting with sequential road-frame & duplicate grouping."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from ml.constants import TARGET_CODES
from ml.dataset.parsers import ImageAnnotationRecord


_STEM_SEQ_PATTERN = re.compile(r"^(?P<prefix>[A-Za-z_]+?)[_-]?(?P<num>\d+)$")


def _extract_sequence_key(stem: str, sequence_block_size: int = 20) -> str:
    """
    Group nearby numeric filenames from road recording sequences into blocks.
    For example, `India_000124` and `India_000127` both map to `seq:india:block_6`,
    preventing adjacent vehicle-mounted camera frames from leaking across train and test.
    """
    m = _STEM_SEQ_PATTERN.match(stem)
    if m and sequence_block_size > 1:
        prefix = m.group("prefix").strip("_").lower()
        num = int(m.group("num"))
        block = num // sequence_block_size
        return f"seq:{prefix}:block_{block}"
    return f"id:{stem.lower()}"


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_leakage_safe_splits(
    records: List[ImageAnnotationRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    sequence_block_size: int = 20,
    deduplicate_exact_within_dataset: bool = True,
) -> Tuple[List[ImageAnnotationRecord], Dict[str, int]]:
    """
    Partition valid records into reproducible 70/15/15 (train/val/test) splits while preventing leakage.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.4f}")

    valid: List[ImageAnnotationRecord] = []
    seen_sha: set[str] = set()
    removed_exact_dups = 0

    for rec in records:
        if rec.is_corrupt:
            continue
        if deduplicate_exact_within_dataset and rec.sha256:
            if rec.sha256 in seen_sha:
                removed_exact_dups += 1
                continue
            seen_sha.add(rec.sha256)
        valid.append(rec)

    if not valid:
        return [], {"train": 0, "val": 0, "test": 0, "removed_exact_duplicates": removed_exact_dups}

    uf = _UnionFind()
    key_owner: Dict[str, int] = {}

    for idx, rec in enumerate(valid):
        uf.find(idx)
        keys = [_extract_sequence_key(rec.image_id, sequence_block_size=sequence_block_size)]
        if rec.sha256:
            keys.append(f"sha:{rec.sha256}")
        if rec.dhash:
            keys.append(f"dhash:{rec.dhash}")
        for k in keys:
            if k in key_owner:
                uf.union(key_owner[k], idx)
            else:
                key_owner[k] = idx

    groups_map: Dict[int, List[ImageAnnotationRecord]] = defaultdict(list)
    for idx, rec in enumerate(valid):
        root_idx = uf.find(idx)
        groups_map[root_idx].append(rec)

    rng = random.Random(seed)
    group_items = list(groups_map.items())
    group_items.sort(key=lambda item: item[1][0].image_id)
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: sum(len(r.boxes) for r in item[1]) // 4, reverse=True)

    target_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    total_images = len(valid)
    total_class_counts = Counter()
    for rec in valid:
        for b in rec.boxes:
            total_class_counts[b.class_code] += 1

    split_img_counts = {"train": 0, "val": 0, "test": 0}
    split_cls_counts: Dict[str, Counter] = {s: Counter() for s in ("train", "val", "test")}

    for root_idx, grp_records in group_items:
        gid = f"grp_{root_idx:06d}"
        grp_len = len(grp_records)
        grp_cls = Counter()
        for r in grp_records:
            r.group_id = gid
            for b in r.boxes:
                grp_cls[b.class_code] += 1

        best_split = "train"
        best_score = -1e18
        for sp in ("train", "val", "test"):
            desired_imgs = target_ratios[sp] * total_images
            img_deficit = (desired_imgs - split_img_counts[sp]) / max(1.0, desired_imgs)

            cls_deficit_sum = 0.0
            active_cls = 0
            for code in TARGET_CODES:
                if grp_cls[code] > 0 and total_class_counts[code] > 0:
                    desired_c = target_ratios[sp] * total_class_counts[code]
                    cls_deficit_sum += (desired_c - split_cls_counts[sp][code]) / max(1.0, desired_c)
                    active_cls += 1

            cls_deficit = (cls_deficit_sum / active_cls) if active_cls > 0 else img_deficit
            combined_score = 0.45 * img_deficit + 0.55 * cls_deficit
            if combined_score > best_score:
                best_score = combined_score
                best_split = sp

        for r in grp_records:
            r.assigned_split = best_split
        split_img_counts[best_split] += grp_len
        split_cls_counts[best_split].update(grp_cls)

    summary = {
        "train": split_img_counts["train"],
        "val": split_img_counts["val"],
        "test": split_img_counts["test"],
        "total_groups": len(group_items),
        "removed_exact_duplicates": removed_exact_dups,
    }
    return valid, summary
