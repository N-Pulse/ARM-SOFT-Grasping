"""
armsoft/sources/recorded.py
===========================
Replay of previously captured frames, and the writer that produces them.

Record real frames once with a camera, then develop and test against them
anywhere — no hardware, and byte-identical results every run.
"""

from __future__ import annotations

import glob
import os
import time

import numpy as np

from .base import CameraIntrinsics, FrameSource, RGBDFrame


class RecordedSource(FrameSource):
    """Replays ``frame_*.npz`` files written by ``run_inference.py --record``."""

    name = "recording"

    def __init__(self, path: str, loop: bool = False):
        self.path = path
        self.loop = loop
        self._files: list[str] = []
        self._i = 0

    def open(self) -> "RecordedSource":
        if os.path.isdir(self.path):
            self._files = sorted(glob.glob(os.path.join(self.path, "*.npz")))
            if not self._files:
                raise FileNotFoundError(
                    f"no .npz frames in the directory {self.path!r} — record "
                    f"some first with '--source camera --record <dir>'")
        elif os.path.isfile(self.path):
            self._files = [self.path]
        else:
            raise FileNotFoundError(f"no such file or directory: {self.path!r}")
        self._i = 0
        return self

    def read(self) -> RGBDFrame | None:
        if self._i >= len(self._files):
            if not self.loop:
                return None
            self._i = 0
        with np.load(self._files[self._i]) as z:
            k = z["intrinsics"]
            intr = CameraIntrinsics(int(k[0]), int(k[1]), *(float(x) for x in k[2:6]))
            frame = RGBDFrame(color_bgr=z["color_bgr"], depth_m=z["depth_m"],
                              intrinsics=intr, index=self._i,
                              timestamp=time.time(), source=self.name)
        self._i += 1
        return frame


def save_frame(frame: RGBDFrame, path: str) -> None:
    """Write one frame to ``path`` (.npz) for later replay by RecordedSource."""
    K = frame.intrinsics
    np.savez_compressed(
        path, color_bgr=frame.color_bgr, depth_m=frame.depth_m,
        intrinsics=np.array([K.width, K.height, K.fx, K.fy, K.cx, K.cy],
                            dtype=np.float64))
