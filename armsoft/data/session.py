"""Versioned, lossless-on-disk RGB-D capture sessions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..sources.base import RGBDFrame

SESSION_SCHEMA_VERSION = "armsoft.capture.v1"


class CaptureSessionWriter:
    """Write aligned colour/depth pairs without changing the source arrays."""

    def __init__(self, root: str | Path, *, purpose: str, notes: str = ""):
        self.root = Path(root)
        if (self.root / "session.json").exists() or (self.root / "manifest.jsonl").exists():
            raise FileExistsError(
                f"capture session already exists at {self.root}; choose a new --out path"
            )
        self.purpose = purpose
        self.notes = notes
        self.color_dir = self.root / "color"
        self.depth_dir = self.root / "depth"
        self.color_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = self.root / "manifest.jsonl"
        self._count = 0
        self._intrinsics: dict | None = None

    def write(self, frame: RGBDFrame) -> dict:
        if frame.color_bgr.shape[:2] != frame.depth_m.shape:
            raise ValueError("colour and depth must be pixel-aligned")
        if frame.depth_m.dtype != np.float32:
            depth = frame.depth_m.astype(np.float32)
        else:
            depth = frame.depth_m

        frame_id = f"frame_{self._count:06d}"
        color_rel = Path("color") / f"{frame_id}.png"
        depth_rel = Path("depth") / f"{frame_id}.npy"
        if not cv2.imwrite(str(self.root / color_rel), frame.color_bgr):
            raise OSError(f"could not write {color_rel}")
        np.save(self.root / depth_rel, depth, allow_pickle=False)

        intrinsics = frame.intrinsics.to_dict()
        if self._intrinsics is None:
            self._intrinsics = intrinsics
        elif intrinsics != self._intrinsics:
            raise ValueError("camera intrinsics changed inside one capture session")

        valid = depth > 0
        record = {
            "id": frame_id,
            "color": color_rel.as_posix(),
            "depth": depth_rel.as_posix(),
            "source_index": int(frame.index),
            "timestamp": float(frame.timestamp),
            "source": frame.source,
            "valid_depth_fraction": float(valid.mean()),
        }
        with self._manifest.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        self._count += 1
        return record

    def close(self) -> dict:
        metadata = {
            "schema": SESSION_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": self.purpose,
            "notes": self.notes,
            "frame_count": self._count,
            "intrinsics": self._intrinsics,
            "units": {"depth": "metres"},
            "colour_order_on_disk": "BGR decoded by OpenCV; PNG itself is standard RGB",
        }
        (self.root / "session.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return metadata

    def __enter__(self) -> "CaptureSessionWriter":
        return self

    def __exit__(self, *_):
        self.close()
