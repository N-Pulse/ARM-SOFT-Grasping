"""
Plugging your own camera (or dataset, or renderer) into the model.

    python examples/custom_source.py

Implement `read()` to return an RGBDFrame and everything downstream — object
isolation, shape fitting, grasp planning — works unchanged.  This example
builds frames from two numpy arrays, which is all any source ever has to do.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from armsoft import (CameraIntrinsics, FrameSource, GraspPipeline, RGBDFrame,
                     SimulatedSource)


class MyFrameSource(FrameSource):
    """A source that hands the model arrays from wherever you like."""

    name = "my-camera"

    def __init__(self, n_frames=10):
        # Intrinsics of YOUR camera, for the colour-aligned depth image.
        self.intrinsics = CameraIntrinsics(width=640, height=480,
                                           fx=385.0, fy=385.0,
                                           cx=320.0, cy=240.0)
        self._n = n_frames
        self._i = 0
        # Stand-in for your real capture device: the built-in renderer.
        self._demo = SimulatedSource(shape="cylinder", intrinsics=self.intrinsics)

    def open(self):
        # Connect to the device / open the dataset here.
        return self

    def read(self):
        if self._i >= self._n:
            return None                      # None ends the stream

        # ── replace these two lines with your own capture ──────────────────
        demo = self._demo.read()
        color_bgr = demo.color_bgr           # (H, W, 3) uint8, BGR
        depth_m = demo.depth_m               # (H, W) float32, METRES, 0 = no data
        # ───────────────────────────────────────────────────────────────────

        assert color_bgr.dtype == np.uint8 and depth_m.dtype == np.float32

        frame = RGBDFrame(color_bgr=color_bgr, depth_m=depth_m,
                          intrinsics=self.intrinsics, index=self._i,
                          source=self.name)
        self._i += 1
        return frame

    def close(self):
        # Release the device here.
        pass


if __name__ == "__main__":
    with MyFrameSource(n_frames=10) as source:
        # The table normal in camera coordinates: measure it once with
        # armsoft.detect_table_plane(source), or hard-code it as here.
        pipeline = GraspPipeline(table_normal=(0.0, -1.0, 0.0))

        for frame in source.frames():
            result = pipeline.process(frame)
            if result.ok:
                print(f"frame {result.frame_index}: {result.shape} at "
                      f"{result.distance_m:.3f} m, jaw "
                      f"{result.jaw_opening_m * 1e3:.0f} mm, stable={result.stable}")
            else:
                print(f"frame {result.frame_index}: {result.status}")
