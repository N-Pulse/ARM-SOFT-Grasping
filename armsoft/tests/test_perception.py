"""Common perception contracts across Ultralytics tasks."""
from __future__ import annotations

import unittest

import numpy as np

from armsoft.perception import adapt_ultralytics_result
from armsoft.sources.base import CameraIntrinsics


class ArrayValue:
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, item):
        return self.value[item]


class Boxes:
    def __init__(self, boxes, classes=(0,), confidence=(.9,)):
        self.xyxy = ArrayValue(boxes)
        self.cls = ArrayValue(classes)
        self.conf = ArrayValue(confidence)

    def __len__(self):
        return len(self.cls.value)


class Value:
    def __init__(self, **values):
        self.__dict__.update(values)


class PerceptionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.depth = np.full((48, 64), .5, dtype=np.float32)
        self.intrinsics = CameraIntrinsics(64, 48, 100., 100., 32., 24.)

    def result(self, **values):
        defaults = {"boxes": None, "masks": None, "keypoints": None, "names": {0: "box"}}
        defaults.update(values)
        return Value(**defaults)

    def test_detection_has_box_depth_and_common_contract(self):
        result = self.result(boxes=Boxes([[22, 14, 42, 34]]))
        record = adapt_ultralytics_result(result, self.depth, self.intrinsics)[0].to_dict()
        self.assertEqual(record["geometry_type"], "detection")
        self.assertEqual(record["class_name"], "box")
        self.assertEqual(record["center_px"], [32., 24.])
        self.assertEqual(record["xyz_m"], [0., 0., .5])
        self.assertTrue(record["depth_valid"])

    def test_segmentation_uses_mask_depth_and_polygon(self):
        result = self.result(
            boxes=Boxes([[20, 12, 44, 36]]),
            masks=Value(xy=[np.array([[22, 14], [42, 14], [42, 34], [22, 34]])]),
        )
        record = adapt_ultralytics_result(result, self.depth, self.intrinsics)[0].to_dict()
        self.assertEqual(record["geometry_type"], "segmentation")
        self.assertEqual(len(record["polygon_xy"]), 4)
        self.assertEqual(record["distance_m"], .5)

    def test_pose_uses_visible_keypoints_and_reports_counts(self):
        result = self.result(
            boxes=Boxes([[20, 12, 44, 36]]),
            keypoints=Value(
                xy=ArrayValue([[[30, 20], [34, 28], [0, 0]]]),
                conf=ArrayValue([[.9, .8, .1]]),
            ),
        )
        record = adapt_ultralytics_result(result, self.depth, self.intrinsics)[0].to_dict()
        self.assertEqual(record["geometry_type"], "pose")
        self.assertEqual(record["center_px"], [32., 24.])
        self.assertEqual(record["keypoints_visible"], 2)
        self.assertEqual(record["keypoints_total"], 3)

    def test_missing_depth_is_explicit_not_zero_xyz(self):
        result = self.result(boxes=Boxes([[22, 14, 42, 34]]))
        record = adapt_ultralytics_result(result, np.zeros_like(self.depth), self.intrinsics)[0].to_dict()
        self.assertFalse(record["depth_valid"])
        self.assertIsNone(record["distance_m"])
        self.assertIsNone(record["xyz_m"])


if __name__ == "__main__":
    unittest.main()
