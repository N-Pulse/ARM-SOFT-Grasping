"""Synchronized RGB-D data augmentation for object-segmentation training."""

from .rgbd import AugmentationConfig, AugmentedSample, RGBDAugmentor

__all__ = ["AugmentationConfig", "AugmentedSample", "RGBDAugmentor"]
