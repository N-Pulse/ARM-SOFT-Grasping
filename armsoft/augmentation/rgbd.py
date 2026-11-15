"""Application-aware augmentation for aligned colour, depth and masks.

Geometric operations are sampled once and applied to every modality. Colour
uses linear interpolation; depth and masks use nearest-neighbour interpolation
so no synthetic depth values or fractional labels are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import cv2
import numpy as np

from ..object_segmentation.annotations import ObjectAnnotation
from ..sources.base import CameraIntrinsics, RGBDFrame


@dataclass(frozen=True)
class AugmentationConfig:
    """Conservative defaults for a moving RGB-D prosthesis camera."""

    horizontal_flip_p: float = 0.5
    rotation_deg: float = 12.0
    translation_fraction: float = 0.08
    scale_range: tuple[float, float] = (0.90, 1.15)
    perspective_p: float = 0.15
    perspective_jitter: float = 0.03
    brightness_delta: float = 0.15
    contrast_range: tuple[float, float] = (0.85, 1.15)
    gamma_range: tuple[float, float] = (0.90, 1.10)
    gaussian_noise_std: float = 3.0
    motion_blur_p: float = 0.10
    depth_dropout_p: float = 0.02
    depth_noise_std_at_1m: float = 0.002
    depth_scale_range: tuple[float, float] = (0.99, 1.01)
    depth_offset_m: float = 0.003


@dataclass
class AugmentedSample:
    """Augmented frame data and labels.

    ``annotations`` retain the original class IDs/names. Their polygons and
    boxes are transformed with the image; objects whose transformed polygons
    leave the image are clipped rather than silently discarded.
    """

    frame: RGBDFrame
    masks: list[np.ndarray]
    annotations: list[ObjectAnnotation]


class RGBDAugmentor:
    """Apply synchronized, reproducible RGB-D augmentations."""

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or AugmentationConfig()
        self.rng = np.random.default_rng(seed)

    def augment(
        self,
        frame: RGBDFrame,
        masks: Sequence[np.ndarray] = (),
        annotations: Sequence[ObjectAnnotation] = (),
    ) -> AugmentedSample:
        """Return one augmented sample without mutating the input arrays."""
        self._validate_frame(frame)
        height, width = frame.depth_m.shape
        mask_array = [
            self._validate_mask(mask, height, width).copy() for mask in masks
        ]
        matrix = self._geometric_matrix(width, height)
        color = cv2.warpPerspective(
            frame.color_bgr, matrix, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )
        depth = cv2.warpPerspective(
            frame.depth_m.astype(np.float32), matrix, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        transformed_masks = [
            cv2.warpPerspective(
                mask.astype(np.uint8), matrix, (width, height),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
            ).astype(bool)
            for mask in mask_array
        ]
        transformed_annotations = [
            self._transform_annotation(annotation, matrix, width, height)
            for annotation in annotations
        ]
        color = self._augment_color(color)
        depth = self._augment_depth(depth)
        intrinsics = self._update_intrinsics(frame.intrinsics, matrix)
        augmented_frame = replace(
            frame, color_bgr=color, depth_m=depth, intrinsics=intrinsics,
        )
        return AugmentedSample(
            frame=augmented_frame,
            masks=transformed_masks,
            annotations=transformed_annotations,
        )

    def _geometric_matrix(self, width: int, height: int) -> np.ndarray:
        config = self.config
        center = (width / 2.0, height / 2.0)
        angle = float(self.rng.uniform(-config.rotation_deg, config.rotation_deg))
        scale = float(self.rng.uniform(*config.scale_range))
        affine = cv2.getRotationMatrix2D(center, angle, scale)
        affine[0, 2] += self.rng.uniform(
            -config.translation_fraction * width,
            config.translation_fraction * width,
        )
        affine[1, 2] += self.rng.uniform(
            -config.translation_fraction * height,
            config.translation_fraction * height,
        )
        matrix = np.vstack([affine, [0.0, 0.0, 1.0]])
        if self.rng.random() < config.perspective_p:
            jitter_x = config.perspective_jitter * width
            jitter_y = config.perspective_jitter * height
            source = np.float32([
                [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1],
            ])
            target = source + self.rng.uniform(
                [-jitter_x, -jitter_y], [jitter_x, jitter_y], source.shape
            ).astype(np.float32)
            matrix = cv2.getPerspectiveTransform(source, target) @ matrix
        if self.rng.random() < config.horizontal_flip_p:
            matrix = np.array([[-1.0, 0.0, width - 1.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]]) @ matrix
        return matrix

    def _augment_color(self, image: np.ndarray) -> np.ndarray:
        config = self.config
        result = image.astype(np.float32)
        result = result * float(self.rng.uniform(*config.contrast_range))
        result += float(self.rng.uniform(
            -config.brightness_delta, config.brightness_delta,
        )) * 255.0
        gamma = float(self.rng.uniform(*config.gamma_range))
        result = 255.0 * np.power(np.clip(result / 255.0, 0.0, 1.0), gamma)
        if config.gaussian_noise_std:
            result += self.rng.normal(0.0, config.gaussian_noise_std, result.shape)
        result = np.clip(result, 0.0, 255.0).astype(np.uint8)
        if self.rng.random() < config.motion_blur_p:
            kernel_size = int(self.rng.choice([3, 5]))
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
            result = cv2.filter2D(result, -1, kernel)
        return result

    def _augment_depth(self, depth: np.ndarray) -> np.ndarray:
        config = self.config
        result = depth.astype(np.float32, copy=True)
        valid = result > 0
        scale = float(self.rng.uniform(*config.depth_scale_range))
        result[valid] = result[valid] * scale + float(
            self.rng.uniform(-config.depth_offset_m, config.depth_offset_m)
        )
        if config.depth_noise_std_at_1m:
            std = config.depth_noise_std_at_1m * np.square(result)
            result[valid] += self.rng.normal(0.0, std[valid])
        if config.depth_dropout_p:
            dropout = self.rng.random(result.shape) < config.depth_dropout_p
            result[dropout] = 0.0
        result[result < 0] = 0.0
        return result

    @staticmethod
    def _update_intrinsics(
        intrinsics: CameraIntrinsics, matrix: np.ndarray,
    ) -> CameraIntrinsics:
        """Update the pinhole approximation for image scale/translation/flip."""
        if not np.allclose(matrix[2], [0.0, 0.0, 1.0]) or not np.isclose(matrix[0, 1], 0):
            # Rotation/perspective changes the effective camera orientation;
            # retain the original calibration rather than inventing skew.
            return intrinsics
        sx, sy = matrix[0, 0], matrix[1, 1]
        return replace(
            intrinsics,
            fx=abs(float(sx)) * intrinsics.fx,
            fy=abs(float(sy)) * intrinsics.fy,
            cx=float(sx * intrinsics.cx + matrix[0, 2]),
            cy=float(sy * intrinsics.cy + matrix[1, 2]),
        )

    @staticmethod
    def _transform_annotation(
        annotation: ObjectAnnotation,
        matrix: np.ndarray,
        width: int,
        height: int,
    ) -> ObjectAnnotation:
        points = np.asarray(annotation.polygon_xy, dtype=np.float32)[None, :, :]
        transformed = cv2.perspectiveTransform(points, matrix)[0]
        transformed[:, 0] = np.clip(transformed[:, 0], 0, width - 1)
        transformed[:, 1] = np.clip(transformed[:, 1], 0, height - 1)
        x1, y1 = transformed.min(axis=0)
        x2, y2 = transformed.max(axis=0)
        return replace(
            annotation,
            polygon_xy=transformed.astype(float).tolist(),
            bbox_xyxy=[float(x1), float(y1), float(x2), float(y2)],
        )

    @staticmethod
    def _validate_frame(frame: RGBDFrame) -> None:
        if frame.color_bgr.ndim != 3 or frame.color_bgr.shape[2] != 3:
            raise ValueError("color_bgr must have shape (height, width, 3)")
        if frame.depth_m.ndim != 2 or frame.depth_m.shape != frame.color_bgr.shape[:2]:
            raise ValueError("depth_m must align with color_bgr")

    @staticmethod
    def _validate_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
        if mask.shape != (height, width):
            raise ValueError("every mask must align with the RGB-D frame")
        return mask
