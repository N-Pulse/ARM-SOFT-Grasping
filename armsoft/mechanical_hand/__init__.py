"""Mechanical-hand perception APIs."""

from .keypoints import KEYPOINT_NAMES, SKELETON, KeypointIndex
from .tracker import Fingertip, OrangeFingertipTracker, draw_fingertips
from .annotations import (
    POSE_SCHEMA_VERSION, PoseAnnotation, PosePoint, bbox_from_mask,
    empty_keypoints, to_yolo_pose_line,
)

__all__ = [
    "Fingertip",
    "KEYPOINT_NAMES",
    "KeypointIndex",
    "OrangeFingertipTracker",
    "SKELETON",
    "draw_fingertips",
    "POSE_SCHEMA_VERSION",
    "PoseAnnotation",
    "PosePoint",
    "bbox_from_mask",
    "empty_keypoints",
    "to_yolo_pose_line",
]
