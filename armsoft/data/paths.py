"""One shared data root for every ARM-SOFT worktree and branch."""
from __future__ import annotations
import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "ARM-SOFT-data"

def data_root() -> Path:
    return Path(os.environ.get("ARM_SOFT_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()

def object_capture_root() -> Path:
    return data_root() / "raw" / "object"

def object_derived_root() -> Path:
    return data_root() / "derived" / "object"

def object_annotation_root() -> Path:
    return data_root() / "annotations" / "object_segmentation"
