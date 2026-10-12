"""MediaPipe pose landmarks and lightweight, explainable action labels."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import acos, degrees
from pathlib import Path
from typing import Sequence
from urllib.request import urlretrieve

import cv2
import numpy as np


POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# MediaPipe's 33-landmark pose topology. Keeping it here avoids depending on
# private drawing helpers that have changed between MediaPipe releases.
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19),
    (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20),
    (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
    (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31),
    (30, 32), (27, 31), (28, 32),
)


@dataclass(frozen=True)
class ActionPrediction:
    """One frame's action label and landmark quality."""

    label: str
    confidence: float
    landmarks: Sequence


def _visible(point, threshold: float = 0.45) -> bool:
    return getattr(point, "visibility", 1.0) >= threshold


def _angle(a, b, c) -> float:
    """Angle ABC in degrees, using normalized image coordinates."""
    ba = np.array([a.x - b.x, a.y - b.y], dtype=np.float64)
    bc = np.array([c.x - b.x, c.y - b.y], dtype=np.float64)
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return degrees(acos(cosine))


def classify_pose(landmarks: Sequence) -> tuple[str, float]:
    """Classify a few intentional poses from MediaPipe's 33 landmarks.

    This is deliberately rule-based: the first prototype is easy to inspect,
    tune and replace with a temporal classifier once real recordings exist.
    """
    if len(landmarks) < 33:
        return "no_pose", 0.0

    ls, rs = landmarks[11], landmarks[12]
    le, re = landmarks[13], landmarks[14]
    lw, rw = landmarks[15], landmarks[16]
    lh, rh = landmarks[23], landmarks[24]
    lk, rk = landmarks[25], landmarks[26]
    la, ra = landmarks[27], landmarks[28]
    required = (ls, rs, le, re, lw, rw, lh, rh, lk, rk)
    if sum(_visible(p) for p in required) < 8:
        return "no_pose", 0.0

    scale = max(abs(ls.x - rs.x), 0.08)
    left_up = lw.y < ls.y - 0.35 * scale
    right_up = rw.y < rs.y - 0.35 * scale
    left_horizontal = abs(lw.y - ls.y) < 0.55 * scale
    right_horizontal = abs(rw.y - rs.y) < 0.55 * scale
    left_extended = _angle(ls, le, lw) > 150.0
    right_extended = _angle(rs, re, rw) > 150.0

    left_knee = _angle(lh, lk, la) if _visible(la) else 180.0
    right_knee = _angle(rh, rk, ra) if _visible(ra) else 180.0
    knee_bend = min(left_knee, right_knee)

    if left_up and right_up:
        return "both_hands_up", 0.95
    if left_horizontal and right_horizontal and left_extended and right_extended:
        return "t_pose", 0.92
    if knee_bend < 125.0:
        return "squat", 0.85
    if left_up:
        return "left_hand_up", 0.88
    if right_up:
        return "right_hand_up", 0.88
    return "standing", 0.72


class PoseActionRecognizer:
    """MediaPipe Pose Landmarker plus a short majority-vote stabilizer."""

    def __init__(self, model_path: str | Path, smoothing: int = 5):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed; run: pip install mediapipe"
            ) from exc

        model_path = Path(model_path)
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[actions] downloading pose model to {model_path}")
            urlretrieve(POSE_MODEL_URL, model_path)

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._labels: deque[str] = deque(maxlen=max(1, smoothing))
        self._last_timestamp = -1

    def process(self, bgr: np.ndarray, timestamp_ms: int) -> ActionPrediction:
        timestamp_ms = max(timestamp_ms, self._last_timestamp + 1)
        self._last_timestamp = timestamp_ms
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            self._labels.clear()
            return ActionPrediction("no_pose", 0.0, ())

        landmarks = result.pose_landmarks[0]
        label, confidence = classify_pose(landmarks)
        self._labels.append(label)
        stable_label, votes = Counter(self._labels).most_common(1)[0]
        stability = votes / len(self._labels)
        return ActionPrediction(stable_label, confidence * stability, landmarks)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def draw_prediction(image: np.ndarray, prediction: ActionPrediction) -> np.ndarray:
    """Draw pose skeleton and label on a BGR image."""
    canvas = image.copy()
    h, w = canvas.shape[:2]
    landmarks = prediction.landmarks
    if landmarks:
        for a, b in POSE_CONNECTIONS:
            pa, pb = landmarks[a], landmarks[b]
            if _visible(pa) and _visible(pb):
                cv2.line(canvas, (int(pa.x * w), int(pa.y * h)),
                         (int(pb.x * w), int(pb.y * h)), (50, 220, 50), 2)
        for point in landmarks:
            if _visible(point):
                cv2.circle(canvas, (int(point.x * w), int(point.y * h)),
                           3, (0, 180, 255), -1)
    text = f"{prediction.label}  {prediction.confidence:.0%}"
    cv2.rectangle(canvas, (10, 10), (420, 55), (0, 0, 0), -1)
    cv2.putText(canvas, text, (20, 43), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas
