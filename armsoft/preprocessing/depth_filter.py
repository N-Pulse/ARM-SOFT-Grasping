"""Depth-based region filtering shared by annotation and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sources.base import CameraIntrinsics


def estimate_table_plane(depth_m: np.ndarray, intrinsics: CameraIntrinsics,
                         roi_xyxy=(0.03, 0.35, 0.97, 0.97), *,
                         min_depth_m: float = 0.10,
                         max_depth_m: float = 1.50,
                         distance_threshold_m: float = 0.008,
                         iterations: int = 500,
                         seed: int = 0) -> tuple[float, float, float, float]:
    """Fit the dominant table plane in a depth ROI with deterministic RANSAC.

    The returned unit normal points towards the camera, so positive signed
    distance means a point is above the table.
    """
    h, w = depth_m.shape
    x1, y1, x2, y2 = roi_xyxy
    left, right = int(x1 * w), int(x2 * w)
    top, bottom = int(y1 * h), int(y2 * h)
    points = intrinsics.deproject(depth_m)[top:bottom:2, left:right:2].reshape(-1, 3)
    valid = (np.isfinite(points[:, 2]) & (points[:, 2] >= min_depth_m)
             & (points[:, 2] <= max_depth_m))
    points = points[valid]
    if len(points) < 100:
        raise ValueError("not enough valid depth points to estimate the table plane")
    if len(points) > 25000:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), 25000, replace=False)]
    rng = np.random.default_rng(seed)
    best = None
    best_count = 0
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        d = -float(normal @ sample[0])
        inliers = np.abs(points @ normal + d) <= distance_threshold_m
        count = int(inliers.sum())
        if count > best_count:
            best, best_count = inliers, count
    if best is None or best_count < max(100, int(0.15 * len(points))):
        raise ValueError("could not find a dominant table plane")
    table = points[best]
    centroid = table.mean(axis=0)
    _, _, vh = np.linalg.svd(table - centroid, full_matrices=False)
    normal = vh[-1]
    if normal @ centroid > 0:  # point from the table towards the camera origin
        normal = -normal
    normal /= np.linalg.norm(normal)
    d = -float(normal @ centroid)
    return float(normal[0]), float(normal[1]), float(normal[2]), d


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
