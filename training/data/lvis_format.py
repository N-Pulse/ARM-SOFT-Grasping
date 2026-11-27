"""
training/data/lvis_format.py
============================
Pure helpers to select LVIS v1 images for our six classes and write them as YOLO labels.
No network, no image I/O: everything here is unit-tested.
Selection rules: docs/guides/training-data.md, section 11.
"""

from __future__ import annotations

import random
from typing import Iterable, Mapping, Sequence

from training.data.yolo_format import CLASSES

# LVIS v1 category name → our class. Caution: LVIS ``doorknob`` mixes door handles with
# oven, fridge, car and toilet handles and round knobs, while CV-REQ-002 asks for lever-style
# door handles. It is kept, but its images must be checked visually; reject bad ones with
# ``lvis.exclude_images`` in the dataset config.
LVIS_TO_CLASS = {
    "doorknob": "door_handle",
    "ball": "ball", "tennis_ball": "ball", "soccer_ball": "ball", "baseball": "ball",
    "basketball": "ball", "volleyball": "ball", "softball": "ball", "ping-pong_ball": "ball",
    "beachball": "ball", "bowling_ball": "ball",
    "can": "can", "beer_can": "can", "aerosol_can": "can",
    "box": "box", "carton": "box", "crate": "box",
    "bottle": "bottle", "water_bottle": "bottle", "wine_bottle": "bottle",
    "beer_bottle": "bottle", "thermos_bottle": "bottle",
    "pen": "pen_pencil", "pencil": "pen_pencil", "marker": "pen_pencil", "crayon": "pen_pencil",
}

DEFAULT_RULES = {"max_objects": 3, "max_overlap": 0.1, "min_area": 0.005}


def category_map(categories: Iterable[Mapping], mapping: Mapping[str, str] = LVIS_TO_CLASS
                 ) -> dict[int, str]:
    """LVIS ``categories`` → {category_id: our class}; unknown names in ``mapping`` are an error."""
    by_name = {c["name"]: c["id"] for c in categories}
    missing = sorted(set(mapping) - set(by_name))
    if missing:
        raise ValueError(f"LVIS has no category {missing}")
    for target in mapping.values():
        if target not in CLASSES:
            raise ValueError(f"unknown target class {target!r}")
    return {by_name[name]: cls for name, cls in mapping.items()}


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two COCO boxes ``[x, y, w, h]`` (pixels)."""
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def check_image(image: Mapping, boxes: Sequence[tuple[str, Sequence[float]]],
                cat_ids_of: Mapping[str, set[int]], rules: Mapping[str, float]) -> str | None:
    """
    Reason to reject one image, or ``None`` if it is usable. LVIS is federated: an image
    whose class is in ``not_exhaustive_category_ids`` has unlabelled objects, so it is skipped.
    """
    if not boxes:
        return "no_object"
    not_exhaustive = set(image.get("not_exhaustive_category_ids", []))
    if any(cat_ids_of[cls] & not_exhaustive for cls in {cls for cls, _ in boxes}):
        return "not_exhaustive"
    if len(boxes) > rules["max_objects"]:
        return "too_many_objects"
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if box_iou(boxes[i][1], boxes[j][1]) > rules["max_overlap"]:
                return "overlap"
    return None


def visible_classes(image: Mapping, boxes: Sequence[tuple[str, Sequence[float]]],
                    min_area: float) -> set[str]:
    """Classes with at least one box covering ``min_area`` of the image."""
    area = image["width"] * image["height"]
    return {cls for cls, (_, _, w, h) in boxes if w * h >= min_area * area}


def select_per_class(candidates: Mapping[str, Iterable[int]], per_class: int,
                     seed: int = 0) -> dict[int, str]:
    """Pick up to ``per_class`` image ids per class, deterministically, each image once.

    Classes are filled in the fixed CLASSES order; an image already picked for an earlier
    class is not counted again. Returns {image_id: class it was picked for}.
    """
    picked: dict[int, str] = {}
    for cls in CLASSES:
        pool = sorted(set(candidates.get(cls, ())) - set(picked))
        random.Random(f"{seed}:{cls}").shuffle(pool)
        for image_id in pool[:per_class]:
            picked[image_id] = cls
    return picked


def coco_box_to_yolo(bbox: Sequence[float], width: int, height: int):
    """COCO ``[x, y, w, h]`` (pixels) → YOLO ``(xc, yc, w, h)`` (relative), or None if empty."""
    x, y, w, h = bbox
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height)


def yolo_lines(boxes: Iterable[tuple[str, Sequence[float]]], width: int, height: int) -> list[str]:
    lines = []
    for cls, bbox in boxes:
        box = coco_box_to_yolo(bbox, width, height)
        if box is not None:
            lines.append(f"{CLASSES.index(cls)} " + " ".join(f"{v:.6f}" for v in box))
    return lines


def image_stem(image_id: int) -> str:
    """File stem of an LVIS image in our datasets: prefixed so it never collides with Open Images."""
    return f"lvis_{image_id:012d}"
