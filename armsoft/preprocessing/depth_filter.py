"""Depth-based region filtering shared by annotation and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sources.base import CameraIntrinsics


@dataclass(frozen=True)
class DepthFilterConfig:
    min_depth_m: float = 0.10
    max_depth_m: float = 0.80
    roi_xyxy: tuple[float, float, float, float] | None = None
    table_plane: tuple[float, float, float, float] | None = None
    min_height_above_table_m: float = 0.005
    background_bgr: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        if self.min_depth_m < 0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("expected 0 <= min_depth_m < max_depth_m")
        if self.roi_xyxy is not None:
            x1, y1, x2, y2 = self.roi_xyxy
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("roi_xyxy must be normalized inside 0..1")
        if self.table_plane is not None:
            normal = np.asarray(self.table_plane[:3], dtype=float)
            if np.linalg.norm(normal) < 1e-9:
                raise ValueError("table plane normal cannot be zero")
        if self.min_height_above_table_m < 0:
            raise ValueError("min_height_above_table_m must be >= 0")

    def to_dict(self) -> dict:
        return {
            "min_depth_m": self.min_depth_m,
            "max_depth_m": self.max_depth_m,
            "roi_xyxy": self.roi_xyxy,
            "table_plane": self.table_plane,
            "min_height_above_table_m": self.min_height_above_table_m,
            "background_bgr": self.background_bgr,
        }


def valid_depth_mask(depth_m: np.ndarray, config: DepthFilterConfig,
                     intrinsics: CameraIntrinsics | None = None) -> np.ndarray:
    """Return the combined distance, ROI and optional table-height mask."""
    if depth_m.ndim != 2:
        raise ValueError("depth_m must have shape (H, W)")
    mask = (np.isfinite(depth_m) & (depth_m >= config.min_depth_m)
            & (depth_m <= config.max_depth_m))

    if config.roi_xyxy is not None:
        h, w = depth_m.shape
        x1, y1, x2, y2 = config.roi_xyxy
        roi = np.zeros((h, w), dtype=bool)
        left, right = int(round(x1 * w)), int(round(x2 * w))
        top, bottom = int(round(y1 * h)), int(round(y2 * h))
        roi[top:bottom, left:right] = True
        mask &= roi

    if config.table_plane is not None:
        if intrinsics is None:
            raise ValueError("intrinsics are required for table-plane filtering")
        nx, ny, nz, d = config.table_plane
        normal = np.asarray([nx, ny, nz], dtype=np.float32)
        length = float(np.linalg.norm(normal))
        normal /= length
        d /= length
        points = intrinsics.deproject(depth_m)
        signed_height = points @ normal + d
        mask &= signed_height >= config.min_height_above_table_m
    return mask


def apply_depth_filter(color_bgr: np.ndarray, depth_m: np.ndarray,
                       config: DepthFilterConfig,
                       intrinsics: CameraIntrinsics | None = None,
                       dim_background: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return a filtered colour copy and the boolean mask; inputs stay intact."""
    if color_bgr.shape[:2] != depth_m.shape:
        raise ValueError("colour and depth must be pixel-aligned")
    mask = valid_depth_mask(depth_m, config, intrinsics)
    filtered = color_bgr.copy()
    if dim_background:
        filtered[~mask] = (filtered[~mask].astype(np.float32) * 0.15).astype(np.uint8)
    else:
        filtered[~mask] = config.background_bgr
    return filtered, mask
