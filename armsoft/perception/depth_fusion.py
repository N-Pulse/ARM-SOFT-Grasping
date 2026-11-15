"""Reusable 2-D to 3-D measurements for model predictions."""
from __future__ import annotations

import numpy as np

from armsoft.sources.base import CameraIntrinsics


def _median_valid(values: np.ndarray) -> float | None:
    valid = values[np.isfinite(values) & (values > 0)]
    return float(np.median(valid)) if valid.size else None


def deproject_pixel(u: float, v: float, depth_m: float | None,
                    intrinsics: CameraIntrinsics) -> tuple[float, float, float] | None:
    """Convert one image pixel and depth into D405 camera coordinates."""
    if depth_m is None or not np.isfinite(depth_m) or depth_m <= 0:
        return None
    return (
        (float(u) - intrinsics.cx) * float(depth_m) / intrinsics.fx,
        (float(v) - intrinsics.cy) * float(depth_m) / intrinsics.fy,
        float(depth_m),
    )


def box_core_depth(depth_m: np.ndarray, box_xyxy, fraction: float = .5) -> tuple[float, float, float | None]:
    """Measure median depth in a box centre region, excluding edge background."""
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    half_width = max(.5, (x2 - x1) * fraction / 2)
    half_height = max(.5, (y2 - y1) * fraction / 2)
    height, width = depth_m.shape[:2]
    core = depth_m[
        max(0, int(v - half_height)):min(height, int(v + half_height) + 1),
        max(0, int(u - half_width)):min(width, int(u + half_width) + 1),
    ]
    return u, v, _median_valid(core)


def mask_depth(depth_m: np.ndarray, polygon_xy) -> tuple[float, float, float | None]:
    """Measure polygon centroid and median depth without depending on OpenCV."""
    points = np.asarray(polygon_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        raise ValueError("a mask polygon needs at least three (x, y) points")
    # Rasterize lazily here so the reusable result package remains lightweight.
    import cv2
    mask = np.zeros(depth_m.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    u, v = points.mean(axis=0).tolist()
    moments = cv2.moments(points)
    if moments["m00"]:
        u, v = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    return float(u), float(v), _median_valid(depth_m[mask > 0])
