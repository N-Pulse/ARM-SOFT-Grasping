"""Unit tests for the rule-based pose labels; no model download required."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from armsoft.actions import analyze_arms, classify_pose, parse_hands


@dataclass
class Point:
    x: float = 0.5
    y: float = 0.5
    visibility: float = 1.0


@dataclass
class WorldPoint:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


def standing_pose():
    points = [Point() for _ in range(33)]
    coords = {
        11: (0.4, 0.3), 12: (0.6, 0.3), 13: (0.4, 0.48),
        14: (0.6, 0.48), 15: (0.4, 0.65), 16: (0.6, 0.65),
        23: (0.45, 0.58), 24: (0.55, 0.58), 25: (0.45, 0.76),
        26: (0.55, 0.76), 27: (0.45, 0.94), 28: (0.55, 0.94),
    }
    for index, (x, y) in coords.items():
        points[index] = Point(x, y)
    return points


class ClassifyPoseTest(unittest.TestCase):
    def test_standing(self):
        label, _ = classify_pose(standing_pose())
        self.assertEqual(label, "standing")

    def test_both_hands_up(self):
        points = standing_pose()
        points[13], points[15] = Point(0.35, 0.2), Point(0.35, 0.08)
        points[14], points[16] = Point(0.65, 0.2), Point(0.65, 0.08)
        label, _ = classify_pose(points)
        self.assertEqual(label, "both_hands_up")

    def test_left_hand_up(self):
        points = standing_pose()
        points[13], points[15] = Point(0.35, 0.2), Point(0.35, 0.08)
        label, _ = classify_pose(points)
        self.assertEqual(label, "left_hand_up")

    def test_right_hand_up(self):
        points = standing_pose()
        points[14], points[16] = Point(0.65, 0.2), Point(0.65, 0.08)
        label, _ = classify_pose(points)
        self.assertEqual(label, "right_hand_up")

    def test_t_pose(self):
        points = standing_pose()
        points[13], points[15] = Point(0.25, 0.3), Point(0.1, 0.3)
        points[14], points[16] = Point(0.75, 0.3), Point(0.9, 0.3)
        label, _ = classify_pose(points)
        self.assertEqual(label, "t_pose")

    def test_squat(self):
        points = standing_pose()
        points[25], points[27] = Point(0.35, 0.70), Point(0.45, 0.70)
        points[26], points[28] = Point(0.65, 0.70), Point(0.55, 0.70)
        label, _ = classify_pose(points)
        self.assertEqual(label, "squat")

    def test_arm_analysis(self):
        arms = analyze_arms(standing_pose())
        self.assertEqual([arm.side for arm in arms], ["left", "right"])
        self.assertTrue(all(arm.elbow_deg is not None for arm in arms))
        self.assertTrue(all(arm.state.endswith("straight") for arm in arms))

    def test_arm_angle_prefers_3d_world_landmarks(self):
        image_points = standing_pose()
        world_points = [WorldPoint() for _ in range(33)]
        world_points[11] = WorldPoint(0.0, 1.0, 0.0)
        world_points[13] = WorldPoint(0.0, 0.0, 0.0)
        world_points[15] = WorldPoint(1.0, 0.0, 0.0)
        world_points[12] = WorldPoint(0.0, 1.0, 0.0)
        world_points[14] = WorldPoint(0.0, 0.0, 0.0)
        world_points[16] = WorldPoint(1.0, 0.0, 0.0)
        arms = analyze_arms(image_points, world_points)
        self.assertAlmostEqual(arms[0].elbow_deg, 90.0)
        self.assertAlmostEqual(arms[1].elbow_deg, 90.0)

    def test_hand_result_parsing(self):
        @dataclass
        class Category:
            category_name: str
            score: float

        @dataclass
        class Result:
            hand_landmarks: list
            handedness: list
            gestures: list

        result = Result(
            hand_landmarks=[[Point() for _ in range(21)]],
            handedness=[[Category("Left", 0.99)]],
            gestures=[[Category("Open_Palm", 0.91)]],
        )
        hand = parse_hands(result)[0]
        self.assertEqual(hand.side, "left")
        self.assertEqual(hand.gesture, "open_palm")
        self.assertAlmostEqual(hand.confidence, 0.91)


if __name__ == "__main__":
    unittest.main()
