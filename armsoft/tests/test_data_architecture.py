"""Tests for the shared capture contract and depth preprocessing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ..data import CaptureSessionWriter, SESSION_SCHEMA_VERSION, read_capture_session
from ..preprocessing import DepthFilterConfig, apply_depth_filter
from ..sources import CameraIntrinsics, SimulatedSource


class DepthFilterTests(unittest.TestCase):
    def test_filter_keeps_only_requested_distance_without_mutating_input(self):
        color = np.full((2, 3, 3), 90, dtype=np.uint8)
        depth = np.array([[0.0, 0.2, 0.9], [np.nan, 0.5, 0.8]], dtype=np.float32)
        original = color.copy()
        filtered, mask = apply_depth_filter(
            color, depth, DepthFilterConfig(0.15, 0.75)
        )
        np.testing.assert_array_equal(mask, [[False, True, False],
                                             [False, True, False]])
        np.testing.assert_array_equal(color, original)
        self.assertTrue(np.all(filtered[~mask] == 0))
        self.assertTrue(np.all(filtered[mask] == 90))

    def test_normalized_roi(self):
        color = np.full((4, 4, 3), 255, dtype=np.uint8)
        depth = np.full((4, 4), 0.4, dtype=np.float32)
        _, mask = apply_depth_filter(
            color, depth, DepthFilterConfig(0.1, 0.8, roi_xyxy=(0.25, 0.25, 0.75, 0.75))
        )
        expected = np.zeros((4, 4), dtype=bool)
        expected[1:3, 1:3] = True
        np.testing.assert_array_equal(mask, expected)

    def test_table_plane_removes_surface_but_keeps_object_above_it(self):
        # +Y points down. Plane y=0.1 has upward normal (0,-1,0), d=0.1.
        intrinsics = CameraIntrinsics(2, 2, 1.0, 1.0, 0.0, 0.0)
        depth = np.array([[0.4, 0.4], [0.1, 0.4]], dtype=np.float32)
        color = np.full((2, 2, 3), 255, dtype=np.uint8)
        config = DepthFilterConfig(
            0.05, 0.8, table_plane=(0.0, -1.0, 0.0, 0.1),
            min_height_above_table_m=0.01,
        )
        _, mask = apply_depth_filter(color, depth, config, intrinsics)
        # Top row y=0 is above the table; bottom-left lies on it.
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[1, 0])


class CaptureSessionTests(unittest.TestCase):
    def test_simulated_frame_round_trips_to_session(self):
        frame = SimulatedSource(n_frames=1, noise_m=0.0).read()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "session"
            writer = CaptureSessionWriter(root, purpose="general", notes="test")
            record = writer.write(frame)
            metadata = writer.close()

            self.assertEqual(metadata["schema"], SESSION_SCHEMA_VERSION)
            self.assertEqual(metadata["frame_count"], 1)
            saved_color = cv2.imread(str(root / record["color"]))
            saved_depth = np.load(root / record["depth"], allow_pickle=False)
            np.testing.assert_array_equal(saved_color, frame.color_bgr)
            np.testing.assert_array_equal(saved_depth, frame.depth_m)
            lines = (root / "manifest.jsonl").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["id"], "frame_000000")
            rows = list(read_capture_session(root))
            self.assertEqual(len(rows), 1)
            np.testing.assert_array_equal(rows[0][1].depth_m, frame.depth_m)


if __name__ == "__main__":
    unittest.main()
