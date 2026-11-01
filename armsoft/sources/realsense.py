"""
armsoft/sources/realsense.py
============================
Live Intel RealSense depth camera.

``pyrealsense2`` is imported inside :meth:`RealSenseSource.open`, so this module
stays importable on machines without the SDK — which is why the rest of the
package can depend on it unconditionally.
"""

from __future__ import annotations

import time

import numpy as np

from .base import CameraIntrinsics, FrameSource, RGBDFrame


class RealSenseSource(FrameSource):
    """
    Live Intel RealSense D405 source.

    ``pyrealsense2`` is imported inside ``open()``, so this module stays
    importable on machines without the SDK.
    """

    name = "realsense"

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 filters: bool = True):
        self.width, self.height, self.fps = width, height, fps
        self._filters = filters
        self._pipe = self._align = self._parts = None
        self._intr = None
        self._scale = 0.001
        self._i = 0

    def open(self) -> "RealSenseSource":
        try:
            import pyrealsense2 as rs  # noqa: PLC0415 — optional, camera-only
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed in this environment, so a real "
                "camera cannot be opened.\n"
                "  install it with:  pip install pyrealsense2\n"
                "  or run without a camera:  --source sim  /  --source replay"
            ) from exc

        if len(rs.context().query_devices()) == 0:
            raise RuntimeError(
                "no RealSense device found.\n"
                "  check: USB 3 port and cable, no other process holding the "
                "camera,\n"
                "         and on Linux that the udev rules are installed.\n"
                "  or run without a camera:  --source sim  /  --source replay"
            )

        pipe = rs.pipeline()
        cfg  = rs.config()
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16,  self.fps)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        profile = pipe.start(cfg)

        sensor = profile.get_device().first_depth_sensor()
        try:
            sensor.set_option(rs.option.visual_preset, 4)
        except Exception:
            pass
        self._scale = sensor.get_depth_scale()

        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._intr = CameraIntrinsics(intr.width, intr.height,
                                      intr.fx, intr.fy, intr.ppx, intr.ppy)

        parts = []
        if self._filters:
            spatial, temporal, holes = rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()
            spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            spatial.set_option(rs.option.filter_smooth_delta, 20)
            temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
            temporal.set_option(rs.option.filter_smooth_delta, 20)
            parts = [spatial, temporal, holes]

        self._pipe, self._align, self._parts = pipe, rs.align(rs.stream.color), parts
        return self

    def read(self) -> RGBDFrame | None:
        if self._pipe is None:
            raise RuntimeError("RealSenseSource.open() was not called")
        frames  = self._pipe.wait_for_frames()
        aligned = self._align.process(frames)
        df, cf  = aligned.get_depth_frame(), aligned.get_color_frame()
        if not df or not cf:
            return None
        for f in self._parts:
            df = f.process(df)

        depth = np.asanyarray(df.get_data()).astype(np.float32) * self._scale
        bgr   = np.asanyarray(cf.get_data())
        frame = RGBDFrame(color_bgr=bgr, depth_m=depth, intrinsics=self._intr,
                          index=self._i, timestamp=time.time(), source=self.name)
        self._i += 1
        return frame

    def close(self) -> None:
        if self._pipe is not None:
            self._pipe.stop()
            self._pipe = None
