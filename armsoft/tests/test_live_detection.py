"""Live Test records for detection-only (box) models."""
from __future__ import annotations

import unittest

import numpy as np

from armsoft.sources.base import CameraIntrinsics
from tools.dataset_web import box_core_depth, live_object


class FakeBoxes:
    """The few attributes of Ultralytics Boxes that the Live Test reads."""
    def __init__(self, cls, conf):
        self.cls = np.asarray(cls, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)

    def __len__(self):
        return len(self.cls)


class LiveDetectionTests(unittest.TestCase):
    def setUp(self):
        self.intrinsics = CameraIntrinsics(width=64, height=48, fx=100.0, fy=100.0, cx=32.0, cy=24.0)

    def test_box_depth_ignores_background_at_the_edges(self):
        depth = np.full((48, 64), 0.80, dtype=np.float32)      # table behind
        depth[14:34, 22:42] = 0.40                              # object, smaller than its box
        u, v, z = box_core_depth(depth, (16, 8, 48, 40))
        self.assertEqual((u, v), (32.0, 24.0))
        self.assertAlmostEqual(z, 0.40)

    def test_box_depth_is_none_without_valid_pixels(self):
        depth = np.zeros((48, 64), dtype=np.float32)
        depth[20:28, 28:36] = np.nan
        self.assertIsNone(box_core_depth(depth, (24, 16, 40, 32))[2])

    def test_box_partly_outside_the_image_is_clipped(self):
        depth = np.full((48, 64), 0.5, dtype=np.float32)
        self.assertAlmostEqual(box_core_depth(depth, (-20, -20, 10, 10))[2], 0.5)

    def test_live_object_has_class_confidence_distance_and_xyz(self):
        boxes = FakeBoxes([0, 3], [0.9, 0.81234])
        record = live_object({0: "ball", 1: "can", 2: "box", 3: "bottle"}, boxes, 1, 42.0, 24.0, 0.5, self.intrinsics)
        self.assertEqual(record["class_id"], 3)
        self.assertEqual(record["class_name"], "bottle")
        self.assertEqual(record["confidence"], 0.8123)
        self.assertEqual(record["distance_m"], 0.5)
        self.assertEqual(record["xyz_m"], [0.05, 0.0, 0.5])

    def test_live_object_without_depth_has_no_position(self):
        record = live_object(["ball"], FakeBoxes([0], [0.5]), 0, 10.0, 10.0, None, self.intrinsics)
        self.assertIsNone(record["distance_m"])
        self.assertIsNone(record["xyz_m"])


if __name__ == "__main__":
    unittest.main()
