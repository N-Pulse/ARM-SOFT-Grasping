"""Assisted annotation and YOLO-Seg dataset utilities."""

from .annotations import (
    ANNOTATION_SCHEMA_VERSION,
    ObjectAnnotation,
    FrameAnnotation,
    masks_from_components,
    mask_to_polygon,
)

__all__ = [
    "ANNOTATION_SCHEMA_VERSION", "ObjectAnnotation", "FrameAnnotation",
    "masks_from_components", "mask_to_polygon",
]
