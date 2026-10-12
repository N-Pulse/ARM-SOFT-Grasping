"""Reusable RGB-D preprocessing; never modifies raw captures in place."""

from .depth_filter import DepthFilterConfig, apply_depth_filter, valid_depth_mask

__all__ = ["DepthFilterConfig", "apply_depth_filter", "valid_depth_mask"]
