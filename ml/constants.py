"""Canonical class definitions, color palettes, and domain constants for RoadGuard AI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class DamageClassSpec:
    class_id: int
    code: str
    name: str
    description: str
    bgr_color: Tuple[int, int, int]
    hex_color: str
    default_category_weight: float


TARGET_CLASS_SPECS: List[DamageClassSpec] = [
    DamageClassSpec(
        class_id=0,
        code="D00",
        name="Longitudinal Crack",
        description="Linear wheel-path or joint cracks parallel to road travel direction",
        bgr_color=(246, 130, 59),   # Electric Blue (#3B82F6 in RGB -> BGR)
        hex_color="#3B82F6",
        default_category_weight=0.55,
    ),
    DamageClassSpec(
        class_id=1,
        code="D10",
        name="Transverse Crack",
        description="Thermal or reflection cracks perpendicular to road centerline",
        bgr_color=(212, 182, 6),    # Cyan (#06B6D4 in RGB -> BGR)
        hex_color="#06B6D4",
        default_category_weight=0.62,
    ),
    DamageClassSpec(
        class_id=2,
        code="D20",
        name="Alligator Crack",
        description="Interconnected fatigue cracking network indicating structural base distress",
        bgr_color=(11, 158, 245),   # Amber (#F59E0B in RGB -> BGR)
        hex_color="#F59E0B",
        default_category_weight=0.82,
    ),
    DamageClassSpec(
        class_id=3,
        code="D40",
        name="Pothole",
        description="Bowl-shaped localized surface cavity presenting immediate vehicular hazard",
        bgr_color=(68, 68, 239),    # Crimson (#EF4444 in RGB -> BGR)
        hex_color="#EF4444",
        default_category_weight=1.00,
    ),
]

TARGET_CODES: List[str] = [spec.code for spec in TARGET_CLASS_SPECS]
CODE_TO_ID: Dict[str, int] = {spec.code: spec.class_id for spec in TARGET_CLASS_SPECS}
ID_TO_SPEC: Dict[int, DamageClassSpec] = {spec.class_id: spec for spec in TARGET_CLASS_SPECS}
CODE_TO_SPEC: Dict[str, DamageClassSpec] = {spec.code: spec for spec in TARGET_CLASS_SPECS}

# Normalized lowercase label -> canonical RDD2022 code mapping.
# Only unambiguous aliases for the four target classes are included here.
CANONICAL_LABEL_ALIASES: Dict[str, str] = {
    "d00": "D00",
    "longitudinal crack": "D00",
    "longitudinal cracks": "D00",
    "longitudinal_crack": "D00",
    "linear crack longitudinal": "D00",
    "d10": "D10",
    "transverse crack": "D10",
    "transverse cracks": "D10",
    "transverse_crack": "D10",
    "linear crack transverse": "D10",
    "d20": "D20",
    "alligator crack": "D20",
    "alligator cracks": "D20",
    "aligator crack": "D20",
    "aligator cracks": "D20",
    "alligator_crack": "D20",
    "crocodile crack": "D20",
    "d40": "D40",
    "pothole": "D40",
    "potholes": "D40",
}

# Known non-target classes present in raw RDD2020/RDD2022 XMLs or Kaggle repackagings
# (e.g., `other corruption` at class index 3 in `aliabdelmenam/rdd-2022`).
KNOWN_NON_TARGET_LABELS: set[str] = {
    "other corruption",
    "other_corruption",
    "other",
    "corruption",
    "repair",
    "d01",
    "d0w0",
    "d11",
    "d43",
    "d44",
    "d50",
    "block crack",
}

IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS: set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def normalize_label_token(raw_label: str) -> str:
    """Normalize whitespace and casing for label comparison without guessing semantics."""
    cleaned = " ".join(str(raw_label).strip().lower().replace("-", " ").split())
    return cleaned
