from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from armsoft.pointcloud import (PointCloudConfig, PointCloudPipeline,
                                save_ply)
from armsoft.sources.base import CameraIntrinsics, RGBDFrame


class PointCloudPipelineTests(unittest.TestCase):
    def frame(self):
        color = np.zeros((2, 3, 3), dtype=np.uint8)
        color[0, 1] = (10, 20, 30)  # BGR
        depth = np.array([[0.0, 1.0, 2.0], [0.5, np.nan, 1.5]], dtype=np.float32)
        intrinsics = CameraIntrinsics(3, 2, 2.0, 2.0, 1.0, 0.5)
        return RGBDFrame(color, depth, intrinsics)

    def test_deprojects_valid_depth_and_converts_bgr_to_rgb(self):
        result = PointCloudPipeline(PointCloudConfig(
            min_depth_m=0.4, max_depth_m=1.6, voxel_size_m=0,
            remove_outliers=False)).process(self.frame())
        self.assertEqual(result.point_count, 3)
        np.testing.assert_allclose(result.points_xyz_m[0], [0.0, -0.25, 1.0])
        np.testing.assert_allclose(result.colors_rgb[0], [30/255, 20/255, 10/255])
        np.testing.assert_array_equal(result.source_pixels_uv[0], [1, 0])

    def test_roi_is_applied_before_projection(self):
        result = PointCloudPipeline(PointCloudConfig(
            min_depth_m=0.1, max_depth_m=2.1,
            roi_xyxy=(0.0, 0.0, 2/3, 1.0), voxel_size_m=0,
            remove_outliers=False)).process(self.frame())
        self.assertEqual(result.point_count, 2)
        self.assertTrue(np.all(result.source_pixels_uv[:, 0] < 2))

    def test_voxel_downsampling_and_ply_export(self):
        frame = RGBDFrame(
            np.zeros((1, 3, 3), dtype=np.uint8),
            np.ones((1, 3), dtype=np.float32),
            CameraIntrinsics(3, 1, 10.0, 10.0, -1.0, -1.0),
        )
        result = PointCloudPipeline(PointCloudConfig(
            min_depth_m=0.4, max_depth_m=1.6, voxel_size_m=10,
            remove_outliers=False)).process(frame)
        self.assertEqual(result.point_count, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = save_ply(Path(directory)/"cloud.ply", result)
            text = path.read_text(encoding="ascii")
            self.assertIn("element vertex 1", text)
            self.assertTrue(path.with_suffix(".json").is_file())

    def test_rejects_unaligned_rgb_and_depth(self):
        frame = self.frame()
        frame.color_bgr = np.zeros((3, 3, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "pixel-aligned"):
            PointCloudPipeline().process(frame)


if __name__ == "__main__":
    unittest.main()
