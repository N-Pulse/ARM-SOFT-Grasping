"""Read sessions produced by :mod:`armsoft.data.session`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ..sources.base import CameraIntrinsics, RGBDFrame
from .session import SESSION_SCHEMA_VERSION


def read_capture_session(root: str | Path) -> Iterator[tuple[dict, RGBDFrame]]:
    root = Path(root)
    metadata = json.loads((root / "session.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != SESSION_SCHEMA_VERSION:
        raise ValueError(f"unsupported capture schema: {metadata.get('schema')!r}")
    intrinsics = CameraIntrinsics(**metadata["intrinsics"])
    with (root / "manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            color = cv2.imread(str(root / record["color"]), cv2.IMREAD_COLOR)
            if color is None:
                raise FileNotFoundError(root / record["color"])
            depth = np.load(root / record["depth"], allow_pickle=False)
            yield record, RGBDFrame(
                color_bgr=color,
                depth_m=depth,
                intrinsics=intrinsics,
                index=int(record["source_index"]),
                timestamp=float(record["timestamp"]),
                source=f"session:{root.name}",
            )
