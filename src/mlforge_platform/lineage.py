"""Record Git and DVC lineage consistently in every project run."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def lineage_tags(data_path: str = "data") -> dict[str, str]:
    """Return portable provenance tags for MLflow.

    DVC's directory hash is stable for a checked-out dataset version. A project
    can override it with DATASET_VERSION when its data is assembled elsewhere.
    """
    git_sha = os.getenv("GIT_SHA") or _command("git", "rev-parse", "HEAD")
    dataset_version = os.getenv("DATASET_VERSION")
    if not dataset_version:
        dvc_file = Path(f"{data_path}.dvc")
        dataset_version = _command("dvc", "get-url", str(dvc_file)) if dvc_file.exists() else "unknown"
    return {"git_sha": git_sha, "dataset_version": dataset_version}
