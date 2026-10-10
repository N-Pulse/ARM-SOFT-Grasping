"""
training/data/yolo_format.py
============================
Pure helpers to turn Open Images detections into a YOLO-format dataset.
No network, no FiftyOne: everything here is unit-tested.

YOLO label line:  ``<class_id> <x_centre> <y_centre> <width> <height>``,
all coordinates relative to the image size, in [0, 1].
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Iterable, Mapping, Sequence

CLASSES = ["ball", "can", "box", "bottle"]   # must match data.yaml `names`

# Open Images V7 boxable label → our class.  Several OI labels may map to the
# same class (e.g. add "Football" → "ball" if "Ball" alone is too scarce).
OI_TO_CLASS = {
    "Ball":    "ball",
    "Tin can": "can",
    "Box":     "box",
    "Bottle":  "bottle",
}


def oi_labels_for(cls: str) -> list[str]:
    return [oi for oi, c in OI_TO_CLASS.items() if c == cls]


def filter_detections(dets: Iterable[tuple[str, Sequence[float], Mapping[str, bool]]]):
    """
    Apply the Open Images flag rules (training/decisions.md) to one image:
    skip every detection with IsGroupOf=True (one box around several
    objects) or IsDepiction=True (drawing / printed image, not graspable).

    Returns ``(kept [(label, bbox)], {"group_of": n, "depiction": n})``.
    """
    kept, skipped = [], {"group_of": 0, "depiction": 0}
    for label, bbox, attrs in dets:
        if label not in OI_TO_CLASS:
            continue
        if attrs.get("IsGroupOf"):
            skipped["group_of"] += 1
        elif attrs.get("IsDepiction"):
            skipped["depiction"] += 1
        else:
            kept.append((label, bbox))
    return kept, skipped


def oi_box_to_yolo(bbox: Sequence[float]):
    """FiftyOne ``[x_top_left, y_top_left, w, h]`` (relative) → YOLO ``(xc, yc, w, h)``."""
    x, y, w, h = bbox
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(1.0, x + w), min(1.0, y + h)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    return (x0 + w / 2, y0 + h / 2, w, h)


def yolo_lines(detections: Iterable[tuple[str, Sequence[float]]]) -> list[str]:
    lines = []
    for label, bbox in detections:
        cls = OI_TO_CLASS.get(label)
        if cls is None:
            continue
        box = oi_box_to_yolo(bbox)
        if box is None:
            continue
        lines.append(f"{CLASSES.index(cls)} " + " ".join(f"{v:.6f}" for v in box))
    return lines


def split_ids(ids: Iterable[str], fractions=(0.7, 0.15, 0.15),
              seed: int = 0) -> dict[str, list[str]]:
    """Deterministic train/val/test split (independent of input order)."""
    ids = sorted(set(ids))
    random.Random(seed).shuffle(ids)
    n_train = round(len(ids) * fractions[0])
    n_val   = round(len(ids) * fractions[1])
    return {"train": ids[:n_train],
            "val":   ids[n_train:n_train + n_val],
            "test":  ids[n_train + n_val:]}


def class_counts(lines: Iterable[str]) -> dict[str, int]:
    c = Counter(CLASSES[int(line.split()[0])] for line in lines if line.strip())
    return {cls: c.get(cls, 0) for cls in CLASSES}
