"""Automated Kaggle CLI dataset acquisition and archive extraction with recovery workflows."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


OFFICIAL_RDD2022_S3_URLS: Dict[str, str] = {
    "India": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip",
    "Czech": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Czech.zip",
    "Japan": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Japan.zip",
    "United_States": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_United_States.zip",
    "China_MotorBike": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_China_MotorBike.zip",
    "China_Drone": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_China_Drone.zip",
    "Norway": "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_Norway.zip",
}


@dataclass
class KaggleAuthStatus:
    configured: bool
    method: str
    config_path: Optional[str] = None
    recovery_instructions: List[str] = field(default_factory=list)


def check_kaggle_credentials() -> KaggleAuthStatus:
    """Verify Kaggle authentication availability without reading or logging secret values."""
    env_user = os.environ.get("KAGGLE_USERNAME", "").strip()
    env_key = os.environ.get("KAGGLE_KEY", "").strip()
    env_token = os.environ.get("KAGGLE_API_TOKEN", "").strip()

    if (env_user and env_key) or env_token:
        return KaggleAuthStatus(
            configured=True,
            method="environment_variables",
            config_path=None,
        )

    custom_dir = os.environ.get("KAGGLE_CONFIG_DIR", "").strip()
    candidate_dirs = [Path(custom_dir)] if custom_dir else [
        Path.home() / ".kaggle",
        Path.home() / ".config" / "kaggle",
    ]

    for cdir in candidate_dirs:
        kfile = cdir / "kaggle.json"
        if kfile.is_file() and kfile.stat().st_size > 10:
            return KaggleAuthStatus(
                configured=True,
                method="kaggle_json_file",
                config_path=str(kfile),
            )

    return KaggleAuthStatus(
        configured=False,
        method="missing",
        config_path=str(Path.home() / ".kaggle" / "kaggle.json"),
        recovery_instructions=[
            "1. Sign in to https://www.kaggle.com/settings and click 'Create New Token' under API to download kaggle.json.",
            "2. On Windows PowerShell, place it securely at ~/.kaggle/kaggle.json:",
            "     New-Item -ItemType Directory -Force -Path \"$env:USERPROFILE\\.kaggle\" | Out-Null",
            "     Move-Item -Force -Path \"path\\to\\kaggle.json\" -Destination \"$env:USERPROFILE\\.kaggle\\kaggle.json\"",
            "3. Alternatively, set KAGGLE_USERNAME and KAGGLE_KEY in your shell session (never commit credentials to Git).",
            "4. Alternatively, run notebooks/01_dataset_preparation.ipynb in Google Colab where GPU + high-speed disk are available.",
        ],
    )


def _resolve_kaggle_executable() -> List[str]:
    """Locate the kaggle CLI executable inside the active Python environment or PATH."""
    venv_scripts = Path(sys.executable).parent
    for name in ("kaggle.exe", "kaggle"):
        candidate = venv_scripts / name
        if candidate.is_file():
            return [str(candidate)]
    which_kaggle = shutil.which("kaggle")
    if which_kaggle:
        return [which_kaggle]
    return [sys.executable, "-m", "kaggle.cli"]


def extract_archives_in_directory(
    directory: Path,
    extract_to: Optional[Path] = None,
    remove_archives: bool = False,
) -> List[Path]:
    """Extract all valid .zip and .tar* archives found in `directory` recursively."""
    directory = Path(directory)
    target_root = Path(extract_to) if extract_to else directory
    target_root.mkdir(parents=True, exist_ok=True)

    extracted: List[Path] = []
    seen_archives: set[Path] = set()

    # Two passes handle outer zip containing country zips
    for _ in range(2):
        archives = [
            p for p in list(directory.rglob("*")) + list(target_root.rglob("*"))
            if p.is_file() and p.suffix.lower() in {".zip", ".tar", ".gz", ".tgz"}
        ]
        new_found = False
        for archive_path in archives:
            resolved = archive_path.resolve()
            if resolved in seen_archives:
                continue
            seen_archives.add(resolved)

            dest = target_root / archive_path.stem if archive_path.parent != target_root else target_root
            dest.mkdir(parents=True, exist_ok=True)

            if archive_path.suffix.lower() == ".zip":
                if not zipfile.is_zipfile(archive_path):
                    raise ValueError(
                        f"Archive '{archive_path}' is not a valid ZIP file (it may be an HTML error page or truncated download)."
                    )
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(dest)
                extracted.append(dest)
                new_found = True
                if remove_archives:
                    archive_path.unlink(missing_ok=True)
            elif tarfile.is_tarfile(archive_path):
                with tarfile.open(archive_path, "r:*") as tf:
                    tf.extractall(dest)
                extracted.append(dest)
                new_found = True
                if remove_archives:
                    archive_path.unlink(missing_ok=True)
        if not new_found:
            break

    return extracted


def download_kaggle_dataset(
    dataset_slug: str = "aliabdelmenam/rdd-2022",
    download_dir: Path | str = "./outputs/raw_dataset",
    extract_dir: Optional[Path | str] = None,
    force: bool = False,
) -> Dict[str, object]:
    """Download and extract a Kaggle dataset using the Kaggle CLI with strict error reporting."""
    auth = check_kaggle_credentials()
    if not auth.configured:
        return {
            "success": False,
            "error": "Kaggle API credentials are not configured.",
            "auth_method": auth.method,
            "recovery_instructions": auth.recovery_instructions,
        }

    download_path = Path(download_dir).resolve()
    extract_path = Path(extract_dir).resolve() if extract_dir else download_path / "extracted"
    download_path.mkdir(parents=True, exist_ok=True)
    extract_path.mkdir(parents=True, exist_ok=True)

    cmd = _resolve_kaggle_executable() + [
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(download_path),
    ]
    if force:
        cmd.append("--force")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,
        )
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to execute Kaggle CLI: {exc}",
            "command": " ".join(cmd),
            "recovery_instructions": auth.recovery_instructions,
        }

    if proc.returncode != 0:
        stderr_msg = (proc.stderr or proc.stdout or "").strip()
        return {
            "success": False,
            "error": f"Kaggle CLI exited with code {proc.returncode}: {stderr_msg}",
            "command": " ".join(cmd),
            "recovery_instructions": [
                "Verify that your Kaggle token is valid and not revoked at https://www.kaggle.com/settings.",
                f"Verify the dataset slug '{dataset_slug}' is accessible on https://www.kaggle.com/datasets/{dataset_slug}.",
                "Ensure at least 25 GB of free disk space is available for downloading and extracting RDD2022.",
                "If local bandwidth or disk is constrained, run notebooks/01_dataset_preparation.ipynb in Google Colab.",
            ],
        }

    extracted_dirs = extract_archives_in_directory(download_path, extract_to=extract_path, remove_archives=False)
    return {
        "success": True,
        "dataset_slug": dataset_slug,
        "download_dir": str(download_path),
        "extract_dir": str(extract_path),
        "extracted_paths": [str(p) for p in extracted_dirs],
    }
