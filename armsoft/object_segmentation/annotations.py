"""Stable annotation contract independent of SAM and YOLO versions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

ANNOTATION_SCHEMA_VERSION = "armsoft.object-segmentation.v1"
REVIEW_STATES = {"pending", "approved", "rejected"}


@dataclass
class ObjectAnnotation:
    class_id: int
    class_name: str
    polygon_xy: list[list[float]]
    bbox_xyxy: list[float]
    source: str
    confidence: float | None = None

    def validate(self, width: int, height: int) -> None:
        if self.class_id < 0 or not self.class_name:
            raise ValueError("object class must have a non-negative id and name")
        if len(self.polygon_xy) < 3:
            raise ValueError("segmentation polygon needs at least 3 points")
        points = np.asarray(self.polygon_xy, dtype=float)
        if points.shape[1:] != (2,) or not np.isfinite(points).all():
            raise ValueError("polygon_xy must contain finite [x, y] points")
        if ((points[:, 0] < 0) | (points[:, 0] >= width)
                | (points[:, 1] < 0) | (points[:, 1] >= height)).any():
            raise ValueError("polygon point is outside the image")


@dataclass
class FrameAnnotation:
    frame_id: str
    source_session: str
    source_color: str
    width: int
    height: int
    objects: list[ObjectAnnotation] = field(default_factory=list)
    review_status: str = "pending"
    schema: str = ANNOTATION_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {self.schema!r}")
        if self.review_status not in REVIEW_STATES:
            raise ValueError(f"unknown review status {self.review_status!r}")
        for obj in self.objects:
            obj.validate(self.width, self.height)

    def save(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FrameAnnotation":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["objects"] = [ObjectAnnotation(**item) for item in data.get("objects", [])]
        annotation = cls(**data)
        annotation.validate()
        return annotation


def masks_from_components(mask: np.ndarray, min_area_px: int = 500,
                          max_instances: int = 20) -> list[np.ndarray]:
    """Split a binary depth mask into large connected candidate instances."""
    binary = np.ascontiguousarray(mask.astype(np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            candidates.append((area, labels == label))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in candidates[:max_instances]]


def mask_to_polygon(mask: np.ndarray, simplify_ratio: float = 0.002) -> list[list[float]]:
    """Convert the largest external contour into an editable pixel polygon."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = simplify_ratio * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in polygon]


def mask_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
