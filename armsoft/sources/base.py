"""
armsoft/sources/base.py
=======================
The types every frame source speaks.

:class:`RGBDFrame` is the model's one and only input: an aligned colour image
and a depth image in metres, plus the intrinsics that relate them.  Implement
:meth:`FrameSource.read` and anything — a camera, a dataset, a renderer — can
drive the pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics of the (colour-aligned) depth image."""
    width:  int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def deproject(self, depth_m: np.ndarray) -> np.ndarray:
        """
        Depth image (H, W) in metres → point map (H, W, 3) in metres,
        camera frame: +X right, +Y down, +Z forward (RealSense convention).
        Pixels with depth 0 produce the origin and must be masked out.
        """
        h, w = depth_m.shape
        u = np.arange(w, dtype=np.float32)[None, :]
        v = np.arange(h, dtype=np.float32)[:, None]
        z = depth_m.astype(np.float32)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=-1)

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height,
                "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}


# D405 @ 640x480 — nominal values, close enough for the simulator.
DEFAULT_INTRINSICS = CameraIntrinsics(640, 480, 385.0, 385.0, 320.0, 240.0)


@dataclass
class RGBDFrame:
    """
    One aligned RGB-D frame — the model's single input type.

    Attributes
    ----------
    color_bgr : (H, W, 3) uint8
        Colour image, OpenCV BGR order.
    depth_m : (H, W) float32
        Depth in **metres**, aligned to ``color_bgr``.  0.0 means "no reading".
    intrinsics : CameraIntrinsics
        Intrinsics of the aligned pair.
    index : int
        Monotonic frame counter from the source.
    timestamp : float
        Seconds (``time.time()`` for live sources, synthetic for others).
    source : str
        Human-readable name of the producing source ("realsense", "sim", ...).
    """
    color_bgr: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics = DEFAULT_INTRINSICS
    index: int = 0
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"

    def points(self) -> np.ndarray:
        """Point map (H, W, 3) in metres, camera frame."""
        return self.intrinsics.deproject(self.depth_m)


class FrameSource:
    """Iterable source of :class:`RGBDFrame`.  Subclasses implement ``read()``."""

    name = "base"

    #: Ground-truth / assumed table normal (unit, camera frame) when the source
    #: knows it.  ``None`` means "detect it or use the CLI default".
    table_normal_hint: np.ndarray | None = None

    def open(self) -> "FrameSource":
        return self

    def close(self) -> None:
        pass

    def read(self) -> RGBDFrame | None:
        raise NotImplementedError

    def frames(self, limit: int | None = None) -> Iterator[RGBDFrame]:
        """Yield up to ``limit`` frames (``None`` = until the source stops)."""
        n = 0
        while limit is None or n < limit:
            frame = self.read()
            if frame is None:
                return
            n += 1
            yield frame

    def __enter__(self):
        return self.open()

    def __exit__(self, *_):
        self.close()
