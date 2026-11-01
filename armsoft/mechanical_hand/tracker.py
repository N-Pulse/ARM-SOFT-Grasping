"""Colour-based tracking for the orange fingertips of the ARM-SOFT hand."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Fingertip:
    """One detected orange fingertip in image coordinates."""

    track_id: int
    center_px: tuple[float, float]
    area_px: float
    box: tuple[int, int, int, int]
    confidence: float


class OrangeFingertipTracker:
    """Detect and temporally track up to five orange fingertip shells."""

    def __init__(
        self,
        hue_min: int = 3,
        hue_max: int = 18,
        saturation_min: int = 150,
        value_min: int = 140,
        min_area_ratio: float = 0.00025,
        max_area_ratio: float = 0.12,
        smoothing: float = 0.65,
        max_tips: int = 5,
    ):
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.saturation_min = saturation_min
        self.value_min = value_min
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self.max_tips = max_tips
        self._previous: dict[int, np.ndarray] = {}

    def _mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower = np.array(
            [self.hue_min, self.saturation_min, self.value_min], dtype=np.uint8
        )
        upper = np.array([self.hue_max, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        scale = max(3, int(round(min(bgr.shape[:2]) / 240)))
        if scale % 2 == 0:
            scale += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def detect(self, bgr: np.ndarray) -> tuple[list[Fingertip], np.ndarray]:
        """Return tracked fingertips and the cleaned binary orange mask."""
        mask = self._mask(bgr)
        image_area = float(bgr.shape[0] * bgr.shape[1])
        min_area = self.min_area_ratio * image_area
        max_area = self.max_area_ratio * image_area
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not min_area <= area <= max_area:
                continue
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < 0.70:
                continue
            moment = cv2.moments(contour)
            if abs(moment["m00"]) < 1e-6:
                continue
            center = np.array(
                [moment["m10"] / moment["m00"], moment["m01"] / moment["m00"]],
                dtype=np.float64,
            )
            box = cv2.boundingRect(contour)
            confidence = float(np.clip(0.55 + 0.45 * solidity, 0.0, 1.0))
            candidates.append((center, area, box, confidence))

        # Larger components are much more likely to be the five printed shells.
        candidates.sort(key=lambda item: item[1], reverse=True)
        candidates = candidates[:self.max_tips]
        assigned = self._assign_tracks([item[0] for item in candidates], bgr.shape)

        tips = []
        for (center, area, box, confidence), (track_id, smooth_center) in zip(
            candidates, assigned
        ):
            tips.append(Fingertip(
                track_id=track_id,
                center_px=(float(smooth_center[0]), float(smooth_center[1])),
                area_px=area,
                box=box,
                confidence=confidence,
            ))
        tips.sort(key=lambda tip: tip.track_id)
        return tips, mask

    def _assign_tracks(self, centers: list[np.ndarray], image_shape) -> list[tuple[int, np.ndarray]]:
        if not centers:
            self._previous.clear()
            return []
        if not self._previous:
            ordered = sorted(enumerate(centers), key=lambda item: item[1][0])
            ids = {candidate_index: track_id for track_id, (candidate_index, _) in enumerate(ordered)}
            result = [(ids[i], center) for i, center in enumerate(centers)]
            self._previous = {track_id: center.copy() for track_id, center in result}
            return result

        previous_ids = sorted(self._previous)
        previous = np.stack([self._previous[i] for i in previous_ids])
        current = np.stack(centers)
        distances = np.linalg.norm(previous[:, None, :] - current[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(distances)
        max_distance = 0.18 * np.hypot(image_shape[0], image_shape[1])
        matches = {}
        used_ids = set()
        for row, col in zip(rows, cols):
            if distances[row, col] <= max_distance:
                track_id = previous_ids[row]
                matches[col] = track_id
                used_ids.add(track_id)

        available_ids = [i for i in range(self.max_tips) if i not in used_ids]
        result = []
        next_previous = {}
        for index, center in enumerate(centers):
            track_id = matches.get(index)
            if track_id is None:
                track_id = available_ids.pop(0) if available_ids else index
                smooth = center
            else:
                smooth = (
                    self.smoothing * self._previous[track_id]
                    + (1.0 - self.smoothing) * center
                )
            result.append((track_id, smooth))
            next_previous[track_id] = smooth.copy()
        self._previous = next_previous
        return result


def draw_fingertips(bgr: np.ndarray, fingertips: list[Fingertip],
                    mask: np.ndarray | None = None) -> np.ndarray:
    """Draw tracked IDs, centres and optional mask inset."""
    canvas = bgr.copy()
    colours = ((30, 255, 30), (255, 80, 30), (255, 30, 220),
               (30, 220, 255), (220, 255, 30))
    for tip in fingertips:
        colour = colours[tip.track_id % len(colours)]
        x, y, w, h = tip.box
        cx, cy = (int(round(v)) for v in tip.center_px)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 3)
        cv2.circle(canvas, (cx, cy), 7, colour, -1)
        cv2.putText(canvas, f"tip {tip.track_id + 1}", (x, max(24, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"orange fingertips: {len(fingertips)}/5", (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (50, 230, 50) if len(fingertips) == 5 else (0, 180, 255),
                2, cv2.LINE_AA)

    if mask is not None:
        inset_width = min(320, canvas.shape[1] // 3)
        inset_height = int(mask.shape[0] * inset_width / mask.shape[1])
        inset = cv2.cvtColor(
            cv2.resize(mask, (inset_width, inset_height)), cv2.COLOR_GRAY2BGR
        )
        canvas[0:inset_height, canvas.shape[1] - inset_width:canvas.shape[1]] = inset
    return canvas
