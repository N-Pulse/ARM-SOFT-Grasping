"""
armsoft/sources/__init__.py
===========================
Frame sources, and the factory that picks one.

    from armsoft.sources import create_frame_source

    with create_frame_source("auto") as source:   # camera if present, else sim
        for frame in source.frames(limit=10):
            ...
"""

from __future__ import annotations

from .base import (CameraIntrinsics, DEFAULT_INTRINSICS, FrameSource,
                   RGBDFrame)
from .simulated import SimulatedSource
from .realsense import RealSenseSource
from .recorded import RecordedSource, save_frame

__all__ = ["CameraIntrinsics", "DEFAULT_INTRINSICS", "FrameSource", "RGBDFrame",
           "SimulatedSource", "RealSenseSource", "RecordedSource", "save_frame",
           "camera_available", "create_frame_source"]


def camera_available() -> bool:
    """True when ``pyrealsense2`` is installed *and* a device is connected."""
    try:
        import pyrealsense2 as rs  # noqa: PLC0415
    except Exception:
        return False
    try:
        return len(rs.context().query_devices()) > 0
    except Exception:
        return False


def create_frame_source(kind: str = "auto", **kwargs) -> FrameSource:
    """
    Build a frame source.

    kind
        ``"auto"``   real camera when one is reachable, else the simulator.
        ``"camera"`` force RealSense (raises if unavailable).
        ``"sim"``    force the simulator.
        ``"replay"`` ``RecordedSource``; pass ``path=...``.

    Extra keyword arguments go to the selected source's constructor.
    """
    kind = (kind or "auto").lower()
    sim_keys = {"shape", "distance_m", "diameter_m", "height_m", "yaw_deg",
                "noise_m", "worms", "worm_length_px", "worm_thickness_px",
                "worm_drop_depth", "depth_holes", "n_frames", "drift_m",
                "intrinsics", "seed"}
    cam_keys = {"width", "height", "fps", "filters"}

    if kind == "replay":
        return RecordedSource(**{k: v for k, v in kwargs.items()
                                 if k in {"path", "loop"}})
    if kind == "sim":
        return SimulatedSource(**{k: v for k, v in kwargs.items() if k in sim_keys})
    if kind == "camera":
        return RealSenseSource(**{k: v for k, v in kwargs.items() if k in cam_keys})
    if kind != "auto":
        raise ValueError(f"unknown frame source {kind!r}")

    if camera_available():
        print("[frame_source] RealSense camera detected — using live capture.")
        return RealSenseSource(**{k: v for k, v in kwargs.items() if k in cam_keys})
    print("[frame_source] no RealSense camera — falling back to SIMULATION mode.")
    return SimulatedSource(**{k: v for k, v in kwargs.items() if k in sim_keys})
