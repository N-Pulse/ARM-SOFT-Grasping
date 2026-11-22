"""Adapter from Ultralytics results to the common ARM-SOFT result contract."""
from __future__ import annotations

import numpy as np

from armsoft.sources.base import CameraIntrinsics

from .depth_fusion import box_core_depth, deproject_pixel, mask_depth
from .results import PerceptionInstance


def _array(value) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _scalar(value) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _identity(result, index: int) -> tuple[int, str, float]:
    boxes = getattr(result, "boxes", None)
    class_id = int(_scalar(boxes.cls[index])) if boxes is not None and index < len(boxes) else 0
    confidence = _scalar(boxes.conf[index]) if boxes is not None and index < len(boxes) else 0.0
    names = getattr(result, "names", {})
    class_name = names.get(class_id, class_id) if isinstance(names, dict) else names[class_id]
    return class_id, str(class_name), confidence


def _instance(result, index, geometry_type, center, depth, intrinsics, **geometry):
    class_id, class_name, confidence = _identity(result, index)
    return PerceptionInstance(
        class_id=class_id, class_name=class_name, confidence=confidence,
        geometry_type=geometry_type, center_px=(float(center[0]), float(center[1])),
        distance_m=depth, xyz_m=deproject_pixel(center[0], center[1], depth, intrinsics),
        **geometry,
    )


def adapt_ultralytics_result(result, depth_m: np.ndarray,
                             intrinsics: CameraIntrinsics,
                             *, keypoint_threshold: float = .25) -> list[PerceptionInstance]:
    """Normalize detection, segmentation, or pose output and fuse aligned depth."""
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    masks = getattr(result, "masks", None)
    instances: list[PerceptionInstance] = []

    if keypoints is not None and boxes is not None:
        xy = _array(keypoints.xy)
        confidence = _array(keypoints.conf) if getattr(keypoints, "conf", None) is not None else np.ones(xy.shape[:2])
        box_values = _array(boxes.xyxy)
        for index, points in enumerate(xy):
            visible = confidence[index] >= keypoint_threshold
            if visible.any():
                u, v = np.median(points[visible], axis=0).tolist()
            else:
                x1, y1, x2, y2 = box_values[index]
                u, v = (x1 + x2) / 2, (y1 + y2) / 2
            _, _, depth = box_core_depth(depth_m, box_values[index], fraction=.35)
            instances.append(_instance(
                result, index, "pose", (u, v), depth, intrinsics,
                box_xyxy=tuple(float(value) for value in box_values[index]),
                keypoints_xy=tuple((float(x), float(y)) for x, y in points),
                keypoints_visible=int(visible.sum()), keypoints_total=len(points),
            ))
        return instances

    if masks is not None:
        box_values = _array(boxes.xyxy) if boxes is not None else None
        for index, polygon in enumerate(masks.xy):
            points = np.asarray(polygon, dtype=np.float32)
            if len(points) < 3:
                continue
            u, v, depth = mask_depth(depth_m, points)
            geometry = {"polygon_xy": tuple((float(x), float(y)) for x, y in points)}
            if box_values is not None and index < len(box_values):
                geometry["box_xyxy"] = tuple(float(value) for value in box_values[index])
            instances.append(_instance(result, index, "segmentation", (u, v), depth, intrinsics, **geometry))
        return instances

    if boxes is not None:
        for index, box in enumerate(_array(boxes.xyxy)):
            u, v, depth = box_core_depth(depth_m, box)
            instances.append(_instance(
                result, index, "detection", (u, v), depth, intrinsics,
                box_xyxy=tuple(float(value) for value in box),
            ))
    return instances
