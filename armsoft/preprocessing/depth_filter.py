"""Depth-based region filtering shared by annotation and inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthFilterConfig:
    min_depth_m: float = 0.10
    max_depth_m: float = 0.80
    background_bgr: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        if self.min_depth_m < 0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("expected 0 <= min_depth_m < max_depth_m")


def valid_depth_mask(depth_m: np.ndarray, config: DepthFilterConfig) -> np.ndarray:
    """Return pixels with finite, non-zero depth inside the requested range."""
    if depth_m.ndim != 2:
        raise ValueError("depth_m must have shape (H, W)")
    return (np.isfinite(depth_m) & (depth_m >= config.min_depth_m)
            & (depth_m <= config.max_depth_m))


def apply_depth_filter(color_bgr: np.ndarray, depth_m: np.ndarray,
                       config: DepthFilterConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return a filtered colour copy and the boolean mask; inputs stay intact."""
    if color_bgr.shape[:2] != depth_m.shape:
        raise ValueError("colour and depth must be pixel-aligned")
    mask = valid_depth_mask(depth_m, config)
    filtered = color_bgr.copy()
    filtered[~mask] = config.background_bgr
    return filtered, mask
