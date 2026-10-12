"""Tests for orange fingertip segmentation and stable tracking IDs."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from armsoft.mechanical_hand import OrangeFingertipTracker


def synthetic_hand(offset: int = 0) -> np.ndarray:
    image = np.full((480, 640, 3), 60, dtype=np.uint8)
    orange = (0, 90, 255)
    for index, x in enumerate((100, 200, 300, 400, 500)):
        cv2.rectangle(image, (x + offset, 90 + index % 2 * 12),
                      (x + 50 + offset, 190 + index % 2 * 12), orange, -1)
    cv2.circle(image, (30, 30), 3, orange, -1)  # noise must be rejected
    return image


class OrangeFingertipTrackerTest(unittest.TestCase):
    def test_detects_five_orange_shells(self):
        tracker = OrangeFingertipTracker()
        tips, mask = tracker.detect(synthetic_hand())
        self.assertEqual(len(tips), 5)
        self.assertEqual(mask.shape, (480, 640))
        self.assertEqual([tip.track_id for tip in tips], [0, 1, 2, 3, 4])

    def test_ids_survive_small_motion(self):
        tracker = OrangeFingertipTracker(smoothing=0.0)
        first, _ = tracker.detect(synthetic_hand())
        second, _ = tracker.detect(synthetic_hand(offset=8))
        self.assertEqual([tip.track_id for tip in first],
                         [tip.track_id for tip in second])
        for before, after in zip(first, second):
            self.assertAlmostEqual(after.center_px[0] - before.center_px[0], 8.0)


if __name__ == "__main__":
    unittest.main()
