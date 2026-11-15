"""Common model-output and RGB-D fusion contracts."""

from .depth_fusion import box_core_depth, deproject_pixel, mask_depth
from .results import PerceptionInstance
from .ultralytics import adapt_ultralytics_result

__all__ = [
    "PerceptionInstance",
    "adapt_ultralytics_result",
    "box_core_depth",
    "deproject_pixel",
    "mask_depth",
]
