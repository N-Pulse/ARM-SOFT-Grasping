"""Tests for the shared capture contract and depth preprocessing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from ..data import CaptureSessionWriter, SESSION_SCHEMA_VERSION
from ..preprocessing import DepthFilterConfig, apply_depth_filter
from ..sources import SimulatedSource


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


if __name__ == "__main__":
    unittest.main()
