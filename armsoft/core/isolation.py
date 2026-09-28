"""
armsoft/isolation.py
=====================
Camera-independent object isolation.

Given one :class:`~armsoft.frame_source.RGBDFrame` it

  1. removes depth discontinuities and near-white background,
  2. segments red pixels in HSV and keeps the blob nearest the frame centre,
  3. deprojects that blob into a point cloud with plain numpy.

It never touches a camera SDK and runs no background thread, so the same code
serves live frames, recordings and the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..sources.base import RGBDFrame

# ─── Depth gating ─────────────────────────────────────────────────────────────
MIN_DEPTH_M = 0.07
MAX_DEPTH_M = 0.70
SUBSAMPLE   = 2      # pixel stride in both axes

# ─── Foreground filtering ─────────────────────────────────────────────────────
DEPTH_GAP_M          = 0.030   # 30 raw D405 units ≈ 3 cm
DEPTH_KERNEL_SIZE    = 5
WHITE_BRIGHTNESS_MIN = 170
WHITE_SAT_MAX        = 30
MAX_MASK_FILL        = 0.80

# ─── Red-object HSV segmentation ──────────────────────────────────────────────
RED_HUE_HIGH1 = 10
RED_HUE_LOW2  = 160
RED_SAT_MIN   = 80
RED_VAL_MIN   = 50
RED_MIN_AREA  = 500


@dataclass
class Isolation:
    """Result of isolating one frame."""
    object_points: np.ndarray      # (N, 3) float32, metres, camera frame
    object_colors: np.ndarray      # (N, 3) float32 RGB in [0, 1]
    scene_points:  np.ndarray      # (M, 3) whole visible scene
    scene_colors:  np.ndarray      # (M, 3)
    box: np.ndarray | None         # [x1, y1, x2, y2] pixels, or None
    mask: np.ndarray | None        # (H, W) bool object mask, or None
    crop_bgr: np.ndarray | None    # BGR crop inside `box` (classifier input)
    preview_bgr: np.ndarray        # annotated image for display / debugging

    @property
    def found(self) -> bool:
        return len(self.object_points) > 0


def _foreground_mask(bgr: np.ndarray, depth_m: np.ndarray,
                     exclude_box=None) -> np.ndarray:
    """Bool mask — True where a pixel is plausibly foreground."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (DEPTH_KERNEL_SIZE, DEPTH_KERNEL_SIZE))

    depth   = np.ascontiguousarray(depth_m, dtype=np.float32)
    invalid = depth <= 0

    depth_max = cv2.dilate(depth, kernel)
    erode_in  = depth.copy()
    erode_in[invalid] = 1e6
    depth_min = cv2.erode(erode_in, kernel)
    depth_min[depth_min >= 1e6] = 0.0

    gap_mask = (depth_max - depth_min > DEPTH_GAP_M) | invalid

    brightness = bgr.max(axis=2).astype(np.float32)
    saturation = (bgr.max(axis=2) - bgr.min(axis=2)).astype(np.float32)
    white_mask = (brightness > WHITE_BRIGHTNESS_MIN) & (saturation < WHITE_SAT_MAX)
    if exclude_box is not None:
        x1, y1, x2, y2 = exclude_box
        white_mask[y1:y2, x1:x2] = False

    return ~(gap_mask | white_mask)


def detect_red_mask(bgr: np.ndarray):
    """Red blob closest to the image centre → (bool mask, box) or (None, None)."""
    h, w = bgr.shape[:2]
    center = np.array([w / 2.0, h / 2.0])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask_lo = cv2.inRange(hsv, (0, RED_SAT_MIN, RED_VAL_MIN),
                          (RED_HUE_HIGH1, 255, 255))
    mask_hi = cv2.inRange(hsv, (RED_HUE_LOW2, RED_SAT_MIN, RED_VAL_MIN),
                          (179, 255, 255))
    red = cv2.bitwise_or(mask_lo, mask_hi)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN,  kernel, iterations=2)
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_mask = best_box = None
    best_dist = float("inf")
    for cnt in contours:
        if cv2.contourArea(cnt) < RED_MIN_AREA:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if (cw * ch) / float(w * h) > MAX_MASK_FILL:
            continue
        dist = np.linalg.norm(np.array([x + cw / 2.0, y + ch / 2.0]) - center)
        if dist < best_dist:
            m = np.zeros((h, w), np.uint8)
            cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
            best_mask, best_box, best_dist = m.astype(bool), \
                np.array([x, y, x + cw, y + ch]), dist

    return best_mask, best_box


class ObjectIsolatorRGBD:
    """
    Stateless-per-frame isolator working on :class:`RGBDFrame`.

    Parameters
    ----------
    min_points : int
        Frames yielding fewer isolated points are reported as "not found".
    """

    def __init__(self, min_points: int = 50):
        self.min_points = int(min_points)
        self._last_box = None

    def isolate(self, frame: RGBDFrame, annotate: bool = True) -> Isolation:
        bgr   = frame.color_bgr
        depth = frame.depth_m
        h, w  = depth.shape

        fg_mask = _foreground_mask(bgr, depth, exclude_box=self._last_box)
        mask, box = detect_red_mask(bgr)
        self._last_box = box

        # ── point map + validity, subsampled ─────────────────────────────────
        pts = frame.points()
        s   = SUBSAMPLE
        sub_pts   = pts[::s, ::s]
        sub_depth = depth[::s, ::s]
        sub_bgr   = bgr[::s, ::s]
        valid = (sub_depth > MIN_DEPTH_M) & (sub_depth < MAX_DEPTH_M)

        scene_points = sub_pts[valid].astype(np.float32)
        scene_colors = (sub_bgr[valid][:, ::-1] / 255.0).astype(np.float32)

        if mask is not None:
            sub_mask = mask[::s, ::s] & valid
            obj_points = sub_pts[sub_mask].astype(np.float32)
            obj_colors = (sub_bgr[sub_mask][:, ::-1] / 255.0).astype(np.float32)
        else:
            obj_points = np.zeros((0, 3), np.float32)
            obj_colors = np.zeros((0, 3), np.float32)

        if len(obj_points) < self.min_points:
            obj_points = np.zeros((0, 3), np.float32)
            obj_colors = np.zeros((0, 3), np.float32)

        crop = None
        if box is not None:
            x1, y1, x2, y2 = box
            c = bgr[y1:y2, x1:x2]
            crop = c if c.size else None

        preview = self._preview(bgr, fg_mask, mask, box) if annotate else bgr

        return Isolation(object_points=obj_points, object_colors=obj_colors,
                         scene_points=scene_points, scene_colors=scene_colors,
                         box=box, mask=mask, crop_bgr=crop, preview_bgr=preview)

    @staticmethod
    def _preview(bgr, fg_mask, mask, box) -> np.ndarray:
        preview = bgr.copy()
        preview[~fg_mask] = (preview[~fg_mask] * 0.25).astype(np.uint8)
        if box is not None and mask is not None:
            overlay = np.zeros_like(preview)
            overlay[mask] = (0, 255, 0)
            preview = cv2.addWeighted(preview, 0.7, overlay, 0.3, 0)
            x1, y1, x2, y2 = box
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return preview
