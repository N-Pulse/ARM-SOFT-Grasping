"""Robustly lift 2-D mechanical-hand keypoints into the D405 camera frame."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..sources.base import RGBDFrame
from .keypoints import KEYPOINT_NAMES

FUSION_SCHEMA_VERSION = "armsoft.mechanical-hand-rgbd.v1"


@dataclass(frozen=True)
class DepthSample:
    ok: bool
    depth_m: float | None
    confidence: float
    support_pixels: int
    search_radius_px: int
    dispersion_m: float | None
    reason: str | None = None


@dataclass(frozen=True)
class Keypoint3D:
    name: str
    pixel_xy: tuple[float, float]
    pose_confidence: float
    xyz_m: tuple[float, float, float] | None
    distance_m: float | None
    depth_confidence: float
    fused_confidence: float
    depth_support_pixels: int
    depth_search_radius_px: int
    reason: str | None = None


@dataclass
class MechanicalHandPose3D:
    frame_index: int
    timestamp: float
    source: str
    instance_confidence: float
    bbox_xyxy: list[float]
    keypoints: list[Keypoint3D]
    schema: str = FUSION_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def sample_depth(depth_m: np.ndarray, x: float, y: float, *,
                 valid_mask: np.ndarray | None = None,
                 min_depth_m: float = 0.07, max_depth_m: float = 1.5,
                 max_radius_px: int = 8, min_support: int = 4,
                 surface_tolerance_m: float = 0.025) -> DepthSample:
    """Find a local surface depth while rejecting a different background plane."""
    if max_radius_px < 1 or min_support < 1:
        raise ValueError("max_radius_px and min_support must be positive")
    h, w = depth_m.shape
    cx, cy = int(round(x)), int(round(y))
    if not (0 <= cx < w and 0 <= cy < h):
        return DepthSample(False, None, 0.0, 0, 0, None, "outside_image")
    if valid_mask is not None and valid_mask.shape != depth_m.shape:
        raise ValueError("valid_mask must match depth shape")

    for radius in range(0, max_radius_px + 1, 2):
        radius = max(radius, 1)
        x1, x2 = max(0, cx - radius), min(w, cx + radius + 1)
        y1, y2 = max(0, cy - radius), min(h, cy + radius + 1)
        patch = depth_m[y1:y2, x1:x2]
        keep = np.isfinite(patch) & (patch >= min_depth_m) & (patch <= max_depth_m)
        if valid_mask is not None:
            keep &= valid_mask[y1:y2, x1:x2]
        ys, xs = np.nonzero(keep)
        if len(xs) < min_support:
            continue
        values = patch[ys, xs].astype(np.float64)
        distances = np.hypot((xs + x1) - x, (ys + y1) - y)
        nearest = values[np.argsort(distances)[:min(12, len(values))]]
        reference = float(np.median(nearest))
        inliers = values[np.abs(values - reference) <= surface_tolerance_m]
        if len(inliers) < min_support:
            continue
        depth = float(np.median(inliers))
        mad = float(np.median(np.abs(inliers - depth)))
        support_score = min(1.0, len(inliers) / max(12.0, min_support))
        radius_score = 1.0 - 0.35 * (radius / max(max_radius_px, 1))
        dispersion_score = float(np.exp(-mad / 0.008))
        confidence = float(np.clip(support_score * radius_score * dispersion_score, 0, 1))
        return DepthSample(True, depth, confidence, int(len(inliers)), radius, mad)
    return DepthSample(False, None, 0.0, 0, max_radius_px, None, "no_reliable_depth")


def fuse_pose_3d(frame: RGBDFrame, keypoints_xy: np.ndarray,
                 keypoint_confidence: np.ndarray, *,
                 instance_confidence: float, bbox_xyxy: list[float],
                 valid_mask: np.ndarray | None = None,
                 min_pose_confidence: float = 0.25,
                 max_radius_px: int = 8) -> MechanicalHandPose3D:
    """Fuse one model instance; coordinates are +X right, +Y down, +Z forward."""
    xy = np.asarray(keypoints_xy, dtype=float)
    confidence = np.asarray(keypoint_confidence, dtype=float)
    if xy.shape != (len(KEYPOINT_NAMES), 2):
        raise ValueError(f"expected keypoints shape ({len(KEYPOINT_NAMES)}, 2)")
    if confidence.shape != (len(KEYPOINT_NAMES),):
        raise ValueError(f"expected {len(KEYPOINT_NAMES)} keypoint confidences")
    intr = frame.intrinsics
    output = []
    for name, (x, y), pose_conf in zip(KEYPOINT_NAMES, xy, confidence):
        pose_conf = float(pose_conf)
        if not np.isfinite([x, y, pose_conf]).all():
            output.append(Keypoint3D(
                name, (0.0, 0.0), 0.0, None, None, 0.0, 0.0,
                0, 0, "invalid_pose_output",
            ))
            continue
        if pose_conf < min_pose_confidence:
            output.append(Keypoint3D(
                name, (float(x), float(y)), pose_conf, None, None, 0.0, 0.0,
                0, 0, "low_pose_confidence",
            ))
            continue
        sample = sample_depth(frame.depth_m, x, y, valid_mask=valid_mask,
                              max_radius_px=max_radius_px)
        if not sample.ok:
            output.append(Keypoint3D(
                name, (float(x), float(y)), pose_conf, None, None,
                sample.confidence, 0.0, sample.support_pixels,
                sample.search_radius_px, sample.reason,
            ))
            continue
        z = float(sample.depth_m)
        px = (float(x) - intr.cx) * z / intr.fx
        py = (float(y) - intr.cy) * z / intr.fy
        distance = float(np.linalg.norm([px, py, z]))
        fused_confidence = float(np.sqrt(pose_conf * sample.confidence))
        output.append(Keypoint3D(
            name, (float(x), float(y)), pose_conf, (px, py, z), distance,
            sample.confidence, fused_confidence, sample.support_pixels,
            sample.search_radius_px,
        ))
    return MechanicalHandPose3D(
        frame_index=frame.index, timestamp=frame.timestamp, source=frame.source,
        instance_confidence=float(instance_confidence), bbox_xyxy=bbox_xyxy,
        keypoints=output,
    )
