"""Portable, testable RGB-D point-cloud pipeline.

The core intentionally has no Open3D dependency.  It consumes the project's
aligned :class:`RGBDFrame`, keeps points in the RealSense camera coordinate
system (+X right, +Y down, +Z forward), and returns metric XYZ plus RGB.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..preprocessing import DepthFilterConfig, valid_depth_mask
from ..sources.base import RGBDFrame


@dataclass(frozen=True)
class PointCloudConfig:
    min_depth_m: float = 0.10
    max_depth_m: float = 1.50
    roi_xyxy: tuple[float, float, float, float] | None = None
    table_plane: tuple[float, float, float, float] | None = None
    min_height_above_table_m: float = 0.005
    voxel_size_m: float = 0.005
    remove_outliers: bool = True
    outlier_neighbors: int = 16
    outlier_std_ratio: float = 2.0
    max_points: int = 100_000

    def __post_init__(self):
        DepthFilterConfig(self.min_depth_m, self.max_depth_m, self.roi_xyxy,
                          self.table_plane, self.min_height_above_table_m)
        if self.voxel_size_m < 0:
            raise ValueError("voxel_size_m must be >= 0")
        if self.outlier_neighbors < 2:
            raise ValueError("outlier_neighbors must be >= 2")
        if self.outlier_std_ratio <= 0 or self.max_points < 1:
            raise ValueError("outlier_std_ratio and max_points must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PointCloudResult:
    points_xyz_m: np.ndarray
    colors_rgb: np.ndarray
    source_pixels_uv: np.ndarray
    input_points: int
    valid_points: int

    @property
    def point_count(self) -> int:
        return int(len(self.points_xyz_m))

    def summary(self) -> dict:
        result = {
            "input_points": self.input_points,
            "valid_points": self.valid_points,
            "output_points": self.point_count,
        }
        if self.point_count:
            result.update({
                "centroid_m": self.points_xyz_m.mean(axis=0).round(6).tolist(),
                "bounds_min_m": self.points_xyz_m.min(axis=0).round(6).tolist(),
                "bounds_max_m": self.points_xyz_m.max(axis=0).round(6).tolist(),
            })
        return result


def _voxel_downsample(points: np.ndarray, colors: np.ndarray,
                      pixels: np.ndarray, size: float):
    if size <= 0 or len(points) == 0:
        return points, colors, pixels
    keys = np.floor(points / size).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                  return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    reduced_points = np.column_stack([
        np.bincount(inverse, weights=points[:, axis]) / count
        for axis in range(3)
    ]).astype(np.float32)
    reduced_colors = np.column_stack([
        np.bincount(inverse, weights=colors[:, axis]) / count
        for axis in range(3)
    ]).astype(np.float32)
    return reduced_points, reduced_colors, pixels[first]


def _statistical_filter(points: np.ndarray, colors: np.ndarray,
                        pixels: np.ndarray, neighbors: int, std_ratio: float):
    if len(points) <= neighbors:
        return points, colors, pixels
    distances, _ = cKDTree(points).query(points, k=min(neighbors + 1, len(points)))
    mean_distance = distances[:, 1:].mean(axis=1)
    threshold = float(mean_distance.mean() + std_ratio * mean_distance.std())
    keep = mean_distance <= threshold
    return points[keep], colors[keep], pixels[keep]


class PointCloudPipeline:
    """Convert aligned RGB-D frames into filtered coloured point clouds."""

    def __init__(self, config: PointCloudConfig = PointCloudConfig()):
        self.config = config

    def process(self, frame: RGBDFrame) -> PointCloudResult:
        if frame.color_bgr.shape[:2] != frame.depth_m.shape:
            raise ValueError("colour and depth must be pixel-aligned")
        depth_config = DepthFilterConfig(
            self.config.min_depth_m, self.config.max_depth_m,
            self.config.roi_xyxy, self.config.table_plane,
            self.config.min_height_above_table_m,
        )
        mask = valid_depth_mask(frame.depth_m, depth_config, frame.intrinsics)
        points = frame.points()[mask]
        colors = frame.color_bgr[..., ::-1][mask].astype(np.float32) / 255.0
        vv, uu = np.indices(frame.depth_m.shape)
        pixels = np.column_stack((uu[mask], vv[mask])).astype(np.int32)
        finite = np.isfinite(points).all(axis=1)
        points, colors, pixels = points[finite], colors[finite], pixels[finite]
        valid_points = len(points)
        points, colors, pixels = _voxel_downsample(
            points, colors, pixels, self.config.voxel_size_m)
        if self.config.remove_outliers:
            points, colors, pixels = _statistical_filter(
                points, colors, pixels, self.config.outlier_neighbors,
                self.config.outlier_std_ratio)
        if len(points) > self.config.max_points:
            indices = np.linspace(0, len(points) - 1,
                                  self.config.max_points, dtype=np.int64)
            points, colors, pixels = points[indices], colors[indices], pixels[indices]
        return PointCloudResult(points.astype(np.float32), colors.astype(np.float32),
                                pixels, int(mask.size), int(valid_points))


def save_ply(path: str | Path, result: PointCloudResult) -> Path:
    """Write an interoperable ASCII PLY with metric XYZ and uint8 RGB."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.clip(np.rint(result.colors_rgb * 255), 0, 255).astype(np.uint8)
    header = ("ply\nformat ascii 1.0\n"
              f"element vertex {result.point_count}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(header)
        for point, color in zip(result.points_xyz_m, colors):
            stream.write(f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                         f"{color[0]} {color[1]} {color[2]}\n")
    path.with_suffix(".json").write_text(
        json.dumps(result.summary(), indent=2) + "\n", encoding="utf-8")
    return path
