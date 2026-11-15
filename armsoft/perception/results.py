"""Task-independent perception result types used by live and offline pipelines."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerceptionInstance:
    """One detected, segmented, or posed instance in image and camera space."""

    class_id: int
    class_name: str
    confidence: float
    geometry_type: str
    center_px: tuple[float, float]
    distance_m: float | None = None
    xyz_m: tuple[float, float, float] | None = None
    box_xyxy: tuple[float, float, float, float] | None = None
    polygon_xy: tuple[tuple[float, float], ...] | None = None
    keypoints_xy: tuple[tuple[float, float], ...] | None = None
    keypoints_visible: int | None = None
    keypoints_total: int | None = None

    def to_dict(self) -> dict:
        """Return a JSON-safe record while retaining the legacy Live Test keys."""
        record = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "geometry_type": self.geometry_type,
            "center_px": [round(value, 2) for value in self.center_px],
            "distance_m": round(self.distance_m, 4) if self.distance_m is not None else None,
            "xyz_m": [round(value, 4) for value in self.xyz_m] if self.xyz_m is not None else None,
            "depth_valid": self.distance_m is not None,
        }
        if self.box_xyxy is not None:
            record["box_xyxy"] = [round(value, 2) for value in self.box_xyxy]
        if self.polygon_xy is not None:
            record["polygon_xy"] = [[round(x, 2), round(y, 2)] for x, y in self.polygon_xy]
        if self.keypoints_xy is not None:
            record["keypoints_xy"] = [[round(x, 2), round(y, 2)] for x, y in self.keypoints_xy]
            record["keypoints_visible"] = int(self.keypoints_visible or 0)
            record["keypoints_total"] = int(self.keypoints_total or len(self.keypoints_xy))
        return record
