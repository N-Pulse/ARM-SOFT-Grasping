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
from ..augmentation import AugmentationConfig, RGBDAugmentor
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


class RGBDAugmentationTests(unittest.TestCase):
    def test_geometric_transforms_keep_modalities_and_labels_aligned(self):
        color = np.zeros((20, 30, 3), dtype=np.uint8)
        color[5:15, 8:18] = (20, 80, 200)
        depth = np.ones((20, 30), dtype=np.float32)
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:15, 8:18] = True
        intrinsics = CameraIntrinsics(30, 20, 20.0, 20.0, 15.0, 10.0)
        frame = RGBDFrame(color, depth, intrinsics)
        annotation = ObjectAnnotation(
            class_id=3, class_name="yellow_marker",
            polygon_xy=[[8, 5], [17, 5], [17, 14], [8, 14]],
            bbox_xyxy=[8, 5, 17, 14], source="test",
        )
        config = AugmentationConfig(
            horizontal_flip_p=1.0, rotation_deg=0.0,
            translation_fraction=0.0, scale_range=(1.0, 1.0),
            perspective_p=0.0, brightness_delta=0.0,
            contrast_range=(1.0, 1.0), gamma_range=(1.0, 1.0),
            gaussian_noise_std=0.0, depth_dropout_p=0.0,
            depth_noise_std_at_1m=0.0, depth_scale_range=(1.0, 1.0),
            depth_offset_m=0.0,
        )
        result = RGBDAugmentor(config, seed=4).augment(
            frame, [mask], [annotation],
        )
        self.assertEqual(result.frame.color_bgr.shape, color.shape)
        self.assertTrue(np.array_equal(result.frame.depth_m, depth))
        self.assertTrue(np.array_equal(result.masks[0], np.fliplr(mask)))
        self.assertEqual(result.annotations[0].class_id, 3)
        self.assertEqual(result.annotations[0].class_name, "yellow_marker")
        self.assertEqual(result.annotations[0].bbox_xyxy, [12.0, 5.0, 21.0, 14.0])

    def test_depth_uses_nearest_interpolation_and_can_create_holes(self):
        depth = np.arange(30 * 20, dtype=np.float32).reshape(20, 30)
        frame = RGBDFrame(np.zeros((20, 30, 3), dtype=np.uint8), depth,
                          CameraIntrinsics(30, 20, 20, 20, 15, 10))
        config = AugmentationConfig(
            rotation_deg=0, translation_fraction=0, scale_range=(1, 1),
            perspective_p=0, horizontal_flip_p=0, brightness_delta=0,
            contrast_range=(1, 1), gamma_range=(1, 1), gaussian_noise_std=0,
            depth_dropout_p=1, depth_noise_std_at_1m=0,
            depth_scale_range=(1, 1), depth_offset_m=0,
        )
        result = RGBDAugmentor(config, seed=1).augment(frame)
        self.assertTrue(np.all(result.frame.depth_m == 0))


if __name__ == "__main__":
    unittest.main()
