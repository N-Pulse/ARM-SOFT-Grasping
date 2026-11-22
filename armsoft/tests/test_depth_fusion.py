"""Tests for robust 2-D keypoint to metric 3-D fusion."""

from __future__ import annotations

import json
import unittest

import numpy as np

from ..mechanical_hand import KEYPOINT_NAMES, fuse_pose_3d, sample_depth
from ..sources import CameraIntrinsics, RGBDFrame


class DepthSamplingTest(unittest.TestCase):
    def test_recovers_depth_from_local_hole(self):
        depth = np.full((11, 11), 0.42, dtype=np.float32)
        depth[5, 5] = 0.0
        sample = sample_depth(depth, 5, 5, max_radius_px=4)
        self.assertTrue(sample.ok)
        self.assertAlmostEqual(sample.depth_m, 0.42, places=5)
        self.assertGreater(sample.confidence, 0.5)

    def test_workspace_mask_is_respected(self):
        depth = np.full((9, 9), 0.8, dtype=np.float32)
        depth[3:6, 3:6] = 0.35
        valid = np.zeros_like(depth, dtype=bool)
        valid[3:6, 3:6] = True
        sample = sample_depth(depth, 4, 4, valid_mask=valid)
        self.assertTrue(sample.ok)
        self.assertAlmostEqual(sample.depth_m, 0.35, places=5)

    def test_missing_depth_has_reason(self):
        sample = sample_depth(np.zeros((10, 10), np.float32), 5, 5)
        self.assertFalse(sample.ok)
        self.assertEqual(sample.reason, "no_reliable_depth")


class PoseFusionTest(unittest.TestCase):
    def frame(self) -> RGBDFrame:
        intr = CameraIntrinsics(100, 80, 100.0, 100.0, 50.0, 40.0)
        return RGBDFrame(
            color_bgr=np.zeros((80, 100, 3), np.uint8),
            depth_m=np.full((80, 100), 0.5, np.float32), intrinsics=intr,
            index=7, timestamp=12.5, source="test",
        )

    def test_deprojects_center_and_serializes(self):
        xy = np.tile(np.array([[50.0, 40.0]]), (len(KEYPOINT_NAMES), 1))
        confidence = np.full(len(KEYPOINT_NAMES), 0.9)
        pose = fuse_pose_3d(
            self.frame(), xy, confidence, instance_confidence=0.8,
            bbox_xyxy=[10, 10, 90, 70],
        )
        self.assertEqual(pose.keypoints[0].xyz_m, (0.0, 0.0, 0.5))
        self.assertAlmostEqual(pose.keypoints[0].distance_m, 0.5)
        json.dumps(pose.to_dict())

    def test_low_pose_confidence_does_not_claim_3d(self):
        xy = np.tile(np.array([[50.0, 40.0]]), (len(KEYPOINT_NAMES), 1))
        confidence = np.zeros(len(KEYPOINT_NAMES))
        pose = fuse_pose_3d(
            self.frame(), xy, confidence, instance_confidence=0.8,
            bbox_xyxy=[10, 10, 90, 70], min_pose_confidence=0.25,
        )
        self.assertTrue(all(point.xyz_m is None for point in pose.keypoints))
        self.assertTrue(all(point.reason == "low_pose_confidence"
                            for point in pose.keypoints))

    def test_nonfinite_model_output_remains_strict_json(self):
        xy = np.tile(np.array([[50.0, 40.0]]), (len(KEYPOINT_NAMES), 1))
        confidence = np.full(len(KEYPOINT_NAMES), 0.9)
        xy[3, 0] = np.nan
        pose = fuse_pose_3d(
            self.frame(), xy, confidence, instance_confidence=0.8,
            bbox_xyxy=[10, 10, 90, 70],
        )
        self.assertEqual(pose.keypoints[3].reason, "invalid_pose_output")
        json.dumps(pose.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
