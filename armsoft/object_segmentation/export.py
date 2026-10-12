"""YOLO segmentation serialization kept separate from editable annotations."""

from __future__ import annotations

from .annotations import FrameAnnotation


def to_yolo_lines(annotation: FrameAnnotation) -> list[str]:
    annotation.validate()
    lines = []
    for obj in annotation.objects:
        coords = []
        for x, y in obj.polygon_xy:
            coords.extend((min(max(x / annotation.width, 0.0), 1.0),
                           min(max(y / annotation.height, 0.0), 1.0)))
        values = " ".join(f"{value:.6f}" for value in coords)
        lines.append(f"{obj.class_id} {values}")
    return lines
