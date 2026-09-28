"""
armsoft/sources/artifacts.py
============================
Structured image corruptions for the simulator.

Gaussian depth noise perturbs every pixel a little and averages out; it is the
easy case. Real depth cameras fail in *shaped* ways — thin dark streaks where
the projected pattern is lost, ragged holes at grazing angles, speckle along
edges — and those survive averaging, because they are correlated in space and
persist for many frames.

``add_worms`` draws exactly that: short wandering dark curves ("worms") over
the colour image, optionally punching the depth out underneath them, so the
isolator sees a chewed-up mask and the fitter sees a cloud with bites missing.

All functions modify ``bgr`` / ``depth`` in place and return the mask they
touched, so a caller can report how much of the object was hit.
"""

from __future__ import annotations

import cv2
import numpy as np


def worm_mask(shape: tuple[int, int],
              rng: np.random.Generator,
              count: int,
              length_px: float = 60.0,
              thickness_px: int = 4,
              wiggle: float = 0.5,
              focus_mask: np.ndarray | None = None,
              focus_fraction: float = 0.6) -> np.ndarray:
    """
    Draw ``count`` worm-shaped strokes and return them as a bool mask.

    Each worm is a random walk whose heading drifts by ``wiggle`` radians per
    segment, which gives a curved, organic stroke rather than a straight line.

    focus_mask / focus_fraction
        Roughly ``focus_fraction`` of the worms start on a pixel of
        ``focus_mask`` (normally the object), so the corruption actually lands
        where it matters instead of mostly on the background.
    """
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    if count <= 0:
        return mask.astype(bool)

    focus_px = None
    if focus_mask is not None and focus_mask.any():
        ys, xs = np.nonzero(focus_mask)
        focus_px = (xs, ys)

    n_seg = max(int(length_px / 6.0), 3)
    for _ in range(int(count)):
        if focus_px is not None and rng.random() < focus_fraction:
            i = rng.integers(len(focus_px[0]))
            x, y = float(focus_px[0][i]), float(focus_px[1][i])
        else:
            x, y = float(rng.integers(w)), float(rng.integers(h))

        ang = rng.uniform(0.0, 2.0 * np.pi)
        step = length_px / n_seg
        pts = [(x, y)]
        for _ in range(n_seg):
            ang += rng.normal(0.0, wiggle)
            x += step * np.cos(ang)
            y += step * np.sin(ang)
            pts.append((x, y))

        thick = max(1, int(round(rng.normal(thickness_px, thickness_px * 0.3))))
        cv2.polylines(mask, [np.asarray(pts, np.int32).reshape(-1, 1, 2)],
                      False, 255, thick, cv2.LINE_AA)

    return mask > 0


def add_worms(bgr: np.ndarray,
              depth: np.ndarray,
              rng: np.random.Generator,
              count: int,
              length_px: float = 60.0,
              thickness_px: int = 4,
              darkness: float = 0.15,
              drop_depth: bool = True,
              focus_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Paint dark worms onto ``bgr`` and (by default) drop the depth under them.

    darkness
        What fraction of the original brightness survives — 0.0 is pure black,
        0.15 leaves the faint shading a real dropout usually keeps.
    drop_depth
        True sets the depth to 0 under each worm, i.e. "no reading", which is
        how a real sensor reports these regions. False corrupts only the
        colour image, which is useful for testing the classifier alone.
    """
    mask = worm_mask(bgr.shape[:2], rng, count, length_px, thickness_px,
                     focus_mask=focus_mask)
    if not mask.any():
        return mask
    bgr[mask] = (bgr[mask] * float(darkness)).astype(np.uint8)
    if drop_depth:
        depth[mask] = 0.0
    return mask


def add_depth_holes(depth: np.ndarray,
                    rng: np.random.Generator,
                    count: int,
                    radius_px: float = 6.0,
                    focus_mask: np.ndarray | None = None) -> np.ndarray:
    """Blob-shaped depth dropouts that leave the colour image untouched."""
    h, w = depth.shape
    mask = np.zeros((h, w), np.uint8)
    if count <= 0:
        return mask.astype(bool)

    if focus_mask is not None and focus_mask.any():
        ys, xs = np.nonzero(focus_mask)
    else:
        ys = xs = None

    for _ in range(int(count)):
        if xs is not None:
            i = rng.integers(len(xs))
            cx, cy = int(xs[i]), int(ys[i])
        else:
            cx, cy = int(rng.integers(w)), int(rng.integers(h))
        r = max(1, int(rng.normal(radius_px, radius_px * 0.4)))
        cv2.circle(mask, (cx, cy), r, 255, -1)

    mask = mask > 0
    depth[mask] = 0.0
    return mask
