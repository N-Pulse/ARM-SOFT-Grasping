"""Tests for editable object annotations and YOLO-Seg serialization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ..data import CaptureSessionWriter
from ..sources.base import CameraIntrinsics, RGBDFrame
from ..object_segmentation import (
    FrameAnnotation, ObjectAnnotation, mask_to_polygon, masks_from_components,
)
from ..object_segmentation.export import to_yolo_lines
from tools.annotate_object_segmentation import main as annotate_main


class ProposalTests(unittest.TestCase):
    def test_components_are_filtered_and_sorted_by_area(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[1:11, 1:11] = True
        mask[15:17, 15:17] = True
        components = masks_from_components(mask, min_area_px=10)
        self.assertEqual(len(components), 1)
        self.assertEqual(int(components[0].sum()), 100)
        self.assertGreaterEqual(len(mask_to_polygon(components[0])), 4)


class AnnotationTests(unittest.TestCase):
    def annotation(self) -> FrameAnnotation:
        return FrameAnnotation(
            frame_id="frame_000001", source_session="/tmp/session",
            source_color="color/frame_000001.png", width=100, height=50,
            objects=[ObjectAnnotation(
                class_id=0, class_name="object",
                polygon_xy=[[10, 5], [90, 5], [90, 45], [10, 45]],
                bbox_xyxy=[10, 5, 90, 45], source="depth-components",
            )],
        )

    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotation.json"
            self.annotation().save(path)
            loaded = FrameAnnotation.load(path)
            self.assertEqual(loaded.frame_id, "frame_000001")
            self.assertEqual(loaded.review_status, "pending")
            self.assertEqual(len(loaded.objects), 1)

    def test_yolo_coordinates_are_normalized(self):
        line = to_yolo_lines(self.annotation())[0]
        values = line.split()
        self.assertEqual(values[0], "0")
        coords = [float(item) for item in values[1:]]
        self.assertTrue(all(0 <= value <= 1 for value in coords))
        self.assertEqual(coords[:2], [0.1, 0.1])

    def test_raw_bypass_creates_manual_annotations_without_derived_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); session = root / "raw"; output = root / "annotations"
            intrinsics = CameraIntrinsics(8, 6, 4.0, 4.0, 4.0, 3.0)
            frame = RGBDFrame(np.zeros((6, 8, 3), dtype=np.uint8),
                              np.ones((6, 8), dtype=np.float32), intrinsics)
            with CaptureSessionWriter(session, purpose="object-segmentation") as writer:
                writer.write(frame)
            progress = root / "progress.json"
            with patch("sys.argv", ["annotate_object_segmentation.py", str(session),
                                     "--raw", "--out", str(output),
                                     "--progress-file", str(progress)]):
                self.assertEqual(annotate_main(), 0)
            run = __import__("json").loads((output / "annotation_run.json").read_text())
            progress_data = __import__("json").loads(progress.read_text())
            annotation = FrameAnnotation.load(output / "annotations" / "frame_000000.json")
            self.assertEqual(run["preprocessing_version"], "raw_rgbd")
            self.assertIsNone(run["derived_masks"])
            self.assertEqual(annotation.objects, [])
            self.assertEqual(progress_data, {"processed": 1, "total": 1,
                                             "generated": 1, "preserved": 0})
            with patch("sys.argv", ["annotate_object_segmentation.py", str(session),
                                     "--raw", "--out", str(output),
                                     "--progress-file", str(progress)]):
                self.assertEqual(annotate_main(), 0)
            regenerated = __import__("json").loads(progress.read_text())
            self.assertEqual(regenerated, {"processed": 1, "total": 1,
                                           "generated": 0, "preserved": 1})


if __name__ == "__main__":
    unittest.main()
