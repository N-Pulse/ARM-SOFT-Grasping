"""Canonical, editable 17-keypoint annotation format for the mechanical hand."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .keypoints import KEYPOINT_NAMES

POSE_SCHEMA_VERSION = "armsoft.mechanical-hand-pose.v1"
REVIEW_STATES = {"pending", "approved", "rejected"}


@dataclass
class PosePoint:
    x: float = 0.0
    y: float = 0.0
    visibility: int = 0  # YOLO: 0 absent, 1 labelled/occluded, 2 visible
    source: str = "unset"

    def validate(self, width: int, height: int) -> None:
        if self.visibility not in (0, 1, 2):
            raise ValueError("visibility must be 0, 1 or 2")
        if self.visibility and not (0 <= self.x < width and 0 <= self.y < height):
            raise ValueError("visible/occluded keypoint lies outside the image")


@dataclass
class PoseAnnotation:
    frame_id: str
    source_session: str
    source_color: str
    width: int
    height: int
    bbox_xyxy: list[float]
    keypoints: list[PosePoint]
    fingertip_candidates_xy: list[list[float]] = field(default_factory=list)
    review_status: str = "pending"
    class_id: int = 0
    class_name: str = "mechanical_hand"
    schema: str = POSE_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema != POSE_SCHEMA_VERSION:
            raise ValueError(f"unsupported pose schema {self.schema!r}")
        if self.review_status not in REVIEW_STATES:
            raise ValueError(f"unknown review status {self.review_status!r}")
        if len(self.keypoints) != len(KEYPOINT_NAMES):
            raise ValueError(f"expected {len(KEYPOINT_NAMES)} keypoints")
        for point in self.keypoints:
            point.validate(self.width, self.height)
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain four values")

    def save(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PoseAnnotation":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["keypoints"] = [PosePoint(**point) for point in data["keypoints"]]
        annotation = cls(**data)
        annotation.validate()
        return annotation


def empty_keypoints() -> list[PosePoint]:
    return [PosePoint() for _ in KEYPOINT_NAMES]


def bbox_from_mask(mask: np.ndarray, fallback_width: int, fallback_height: int,
                   padding_px: int = 10) -> list[float]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return [0.0, 0.0, float(fallback_width - 1), float(fallback_height - 1)]
    return [float(max(0, xs.min() - padding_px)),
            float(max(0, ys.min() - padding_px)),
            float(min(fallback_width - 1, xs.max() + padding_px)),
            float(min(fallback_height - 1, ys.max() + padding_px))]


def to_yolo_pose_line(annotation: PoseAnnotation) -> str:
    """Serialize one hand instance using Ultralytics YOLO Pose format."""
    annotation.validate()
    x1, y1, x2, y2 = annotation.bbox_xyxy
    cx = ((x1 + x2) / 2.0) / annotation.width
    cy = ((y1 + y2) / 2.0) / annotation.height
    width = (x2 - x1) / annotation.width
    height = (y2 - y1) / annotation.height
    values = [float(annotation.class_id), cx, cy, width, height]
    for point in annotation.keypoints:
        if point.visibility == 0:
            values.extend((0.0, 0.0, 0.0))
        else:
            values.extend((point.x / annotation.width,
                           point.y / annotation.height,
                           float(point.visibility)))
    return str(annotation.class_id) + " " + " ".join(
        f"{value:.6f}" for value in values[1:]
    )
