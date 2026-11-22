"""Canonical keypoint order shared by annotation, training and inference."""

from __future__ import annotations

from enum import IntEnum


class KeypointIndex(IntEnum):
    WRIST = 0
    PALM_CENTER = 1
    THUMB_BASE = 2
    THUMB_JOINT = 3
    THUMB_TIP = 4
    INDEX_BASE = 5
    INDEX_JOINT = 6
    INDEX_TIP = 7
    MIDDLE_BASE = 8
    MIDDLE_JOINT = 9
    MIDDLE_TIP = 10
    RING_BASE = 11
    RING_JOINT = 12
    RING_TIP = 13
    PINKY_BASE = 14
    PINKY_JOINT = 15
    PINKY_TIP = 16


KEYPOINT_NAMES = tuple(point.name.lower() for point in KeypointIndex)

# Pairs index into KEYPOINT_NAMES and define the lines used for validation and
# visualization. Changing this schema requires a new dataset version.
SKELETON = (
    (KeypointIndex.WRIST, KeypointIndex.PALM_CENTER),
    (KeypointIndex.PALM_CENTER, KeypointIndex.THUMB_BASE),
    (KeypointIndex.THUMB_BASE, KeypointIndex.THUMB_JOINT),
    (KeypointIndex.THUMB_JOINT, KeypointIndex.THUMB_TIP),
    (KeypointIndex.PALM_CENTER, KeypointIndex.INDEX_BASE),
    (KeypointIndex.INDEX_BASE, KeypointIndex.INDEX_JOINT),
    (KeypointIndex.INDEX_JOINT, KeypointIndex.INDEX_TIP),
    (KeypointIndex.PALM_CENTER, KeypointIndex.MIDDLE_BASE),
    (KeypointIndex.MIDDLE_BASE, KeypointIndex.MIDDLE_JOINT),
    (KeypointIndex.MIDDLE_JOINT, KeypointIndex.MIDDLE_TIP),
    (KeypointIndex.PALM_CENTER, KeypointIndex.RING_BASE),
    (KeypointIndex.RING_BASE, KeypointIndex.RING_JOINT),
    (KeypointIndex.RING_JOINT, KeypointIndex.RING_TIP),
    (KeypointIndex.PALM_CENTER, KeypointIndex.PINKY_BASE),
    (KeypointIndex.PINKY_BASE, KeypointIndex.PINKY_JOINT),
    (KeypointIndex.PINKY_JOINT, KeypointIndex.PINKY_TIP),
)
