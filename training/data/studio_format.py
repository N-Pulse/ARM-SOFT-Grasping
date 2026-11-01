"""
training/data/studio_format.py
==============================
Pure helpers to turn Dataset Studio annotations into YOLO detection labels.
Studio class IDs differ between sessions, so classes are matched by name only.
No image I/O: everything here is unit-tested.
"""

from __future__ import annotations

from pathlib import PureWindowsPath
from typing import Iterable, Mapping

from training.data.yolo_format import CLASSES

STUDIO_TO_CLASS = {"can": "can", "box": "box", "ball-pen": "pen_pencil"}

# Whole sessions per split. Test holds objects absent from train and val: the tall
# can and the red cube. Sessions not listed are out of the six classes, or unusable.
SESSION_SPLITS = {
    "train": ["can_1", "can_2", "can_3", "can_4", "can_6", "test-black-box-3", "stylo_green_1"],
    "val":   ["can_5", "box-beige-1"],
    "test":  ["can_7", "test-box-2"],
}


def box_line(class_name: str, bbox_xyxy, width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty box {bbox_xyxy}")
    cls = CLASSES.index(STUDIO_TO_CLASS[class_name])
    return (f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")


def frame_lines(annotation: Mapping) -> list[str]:
    """YOLO lines of one frame; a frame with an unknown class is an error, not a silent gap."""
    return [box_line(o["class_name"], o["bbox_xyxy"], annotation["width"], annotation["height"])
            for o in annotation["objects"]]


def raw_session_name(source_session: str) -> str:
    """Annotations store the absolute Windows path of their raw session."""
    return PureWindowsPath(source_session).name


def sessions_are_disjoint(splits: Mapping[str, Iterable[str]]) -> bool:
    names = [s for sessions in splits.values() for s in sessions]
    return len(names) == len(set(names))
