"""CLI entry point for RoadGuard AI automated dataset acquisition, inspection, and preparation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml.dataset.exporter import export_yolo_dataset
from ml.dataset.inspector import inspect_dataset_directory
from ml.dataset.kaggle_downloader import check_kaggle_credentials, download_kaggle_dataset
from ml.dataset.parsers import DatasetClassDiscrepancyError, parse_dataset
from ml.dataset.splitter import assign_leakage_safe_splits
from ml.dataset.validator import validate_parsed_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RoadGuard AI — Automated RDD2022 Dataset Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. check-auth
    subparsers.add_parser("check-auth", help="Verify Kaggle API authentication without exposing credentials")

    # 2. download
    dl_p = subparsers.add_parser("download", help="Download RDD2022 via Kaggle CLI and extract archives")
    dl_p.add_argument("--slug", default="aliabdelmenam/rdd-2022", help="Kaggle dataset slug")
    dl_p.add_argument("--download-dir", default="./outputs/raw_dataset", help="Download directory")
    dl_p.add_argument("--force", action="store_true", help="Force re-download")

    # 3. inspect
    in_p = subparsers.add_parser("inspect", help="Inspect dataset directory structure and class metadata")
    in_p.add_argument("--source-dir", required=True, help="Root folder of extracted dataset")

    # 4. prepare
    pr_p = subparsers.add_parser("prepare", help="Validate, split (70/15/15), and export YOLO dataset")
    pr_p.add_argument("--source-dir", required=True, help="Root folder of extracted dataset")
    pr_p.add_argument("--output-dir", default="./outputs/dataset", help="Output folder for prepared YOLO dataset")
    pr_p.add_argument(
        "--confirm-exclude-non-target",
        action="store_true",
        help="Confirm excluding non-target classes (e.g. 'other corruption' in aliabdelmenam/rdd-2022)",
    )
    pr_p.add_argument("--seed", type=int, default=42, help="Random seed for 70/15/15 split")
    pr_p.add_argument("--sequence-block", type=int, default=20, help="Frame sequence grouping window")
    pr_p.add_argument("--no-copy-images", action="store_true", help="Export manifests/labels without copying image binaries")

    args = parser.parse_args()

    if args.command == "check-auth":
        status = check_kaggle_credentials()
        print(json.dumps(status.__dict__, indent=2))
        return 0 if status.configured else 1

    if args.command == "download":
        res = download_kaggle_dataset(
            dataset_slug=args.slug,
            download_dir=args.download_dir,
            force=args.force,
        )
        print(json.dumps(res, indent=2))
        return 0 if res.get("success") else 1

    if args.command == "inspect":
        rep = inspect_dataset_directory(args.source_dir)
        print(json.dumps(rep.to_dict(), indent=2))
        return 0

    if args.command == "prepare":
        policy = "exclude_confirmed" if args.confirm_exclude_non_target else "raise"
        try:
            parsed = parse_dataset(args.source_dir, non_target_policy=policy)
        except DatasetClassDiscrepancyError as exc:
            print("DATASET CLASS DISCREPANCY DETECTED:", file=sys.stderr)
            print(str(exc), file=sys.stderr)
            if exc.inspection:
                print(json.dumps(exc.inspection.to_dict(), indent=2), file=sys.stderr)
            return 2

        val_summary = validate_parsed_dataset(parsed, compute_hashes=True)
        split_records, split_counts = assign_leakage_safe_splits(
            parsed.records,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=args.seed,
            sequence_block_size=args.sequence_block,
        )
        exported = export_yolo_dataset(
            parsed=parsed,
            split_records=split_records,
            validation_summary=val_summary,
            output_dir=args.output_dir,
            copy_images=not args.no_copy_images,
        )
        print(json.dumps({"splits": split_counts, "exported": exported, "validation": val_summary.to_dict()}, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
