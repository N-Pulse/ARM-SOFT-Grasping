#!/usr/bin/env python3
"""Generate reproducible RGB-D augmentation variants for approved annotations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from armsoft.augmentation import (
    AugmentationConfig, AugmentedSample, RGBDAugmentor,
)
from armsoft.object_segmentation.annotations import FrameAnnotation
from armsoft.sources.base import CameraIntrinsics, RGBDFrame


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _base_config(**changes: object) -> AugmentationConfig:
    values = {
        "horizontal_flip_p": 0.0, "rotation_deg": 0.0,
        "translation_fraction": 0.0, "scale_range": (1.0, 1.0),
        "perspective_p": 0.0, "brightness_delta": 0.0,
        "contrast_range": (1.0, 1.0), "gamma_range": (1.0, 1.0),
        "gaussian_noise_std": 0.0, "motion_blur_p": 0.0,
        "depth_dropout_p": 0.0, "depth_noise_std_at_1m": 0.0,
        "depth_scale_range": (1.0, 1.0), "depth_offset_m": 0.0,
    }
    values.update(changes)
    return AugmentationConfig(**values)


def _load_frame(annotation: FrameAnnotation) -> RGBDFrame:
    root = Path(annotation.source_session)
    metadata = json.loads((root / "session.json").read_text(encoding="utf-8"))
    intrinsics = CameraIntrinsics(**metadata["intrinsics"])
    color = cv2.imread(
        str(root / annotation.source_color), cv2.IMREAD_COLOR,
    )
    if color is None:
        raise FileNotFoundError(root / annotation.source_color)
    depth = np.load(
        root / "depth" / f"{annotation.frame_id}.npy", allow_pickle=False,
    ).astype(np.float32)
    return RGBDFrame(color, depth, intrinsics, source=f"session:{root.name}")


def _masks(frame: RGBDFrame, annotation: FrameAnnotation) -> list[np.ndarray]:
    result = []
    for obj in annotation.objects:
        mask = np.zeros(frame.depth_m.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(obj.polygon_xy, dtype=np.int32)], 255)
        result.append(mask.astype(bool))
    return result


def _save(
    destination: Path,
    sample,
    source: str,
    variant: str,
    physical_rgbd: bool = True,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination / "color.png"), sample.frame.color_bgr)
    np.save(destination / "depth.npy", sample.frame.depth_m.astype(np.float32))
    for index, mask in enumerate(sample.masks):
        cv2.imwrite(str(destination / f"mask_{index:02d}.png"),
                    (mask.astype(np.uint8) * 255))
    metadata = {
        "schema": "armsoft.rgbd-augmentation.v1",
        "source_annotation": source,
        "variant": variant,
        "physical_rgbd": physical_rgbd,
        "intrinsics": sample.frame.intrinsics.to_dict(),
        "objects": [replace(obj).__dict__ for obj in sample.annotations],
    }
    (destination / "annotations.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )


def _background_sample(augmentor, frame, masks, objects, style):
    union = np.logical_or.reduce(masks)
    background = augmentor._synthetic_background(
        frame.color_bgr.shape[0], frame.color_bgr.shape[1], style,
    )
    color = background
    color[union] = frame.color_bgr[union]
    return AugmentedSample(
        frame=replace(frame, color_bgr=color),
        masks=list(masks),
        annotations=list(objects),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    variants = [
        ("01_horizontal_flip", _base_config(horizontal_flip_p=1.0)),
        ("02_rotation", _base_config(rotation_deg=12.0)),
        ("03_scale", _base_config(scale_range=(1.15, 1.15))),
        ("04_translation", _base_config(translation_fraction=0.08)),
        ("05_perspective", _base_config(perspective_p=1.0)),
        ("06_brightness", _base_config(brightness_delta=0.15)),
        ("07_contrast", _base_config(contrast_range=(1.15, 1.15))),
        ("08_gamma", _base_config(gamma_range=(1.10, 1.10))),
        ("09_gaussian_noise", _base_config(gaussian_noise_std=12.0)),
        ("10_motion_blur", _base_config(motion_blur_p=1.0)),
        ("11_depth_dropout", _base_config(depth_dropout_p=0.08)),
        ("12_depth_noise", _base_config(depth_noise_std_at_1m=0.01)),
        ("13_depth_scale", _base_config(depth_scale_range=(1.05, 1.05))),
        ("14_depth_offset", _base_config(depth_offset_m=0.03)),
    ]
    rotations = [-90, -45, -30, 30, 45, 90]
    annotations = sorted(
        path for path in args.annotations.glob("*/annotations/*.json")
    )
    manifest = {
        "schema": "armsoft.rgbd-augmentation-manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "approved_only": True,
        "seed": args.seed,
        "source_frames": 0,
        "outputs": 0,
    }
    for annotation_path in annotations:
        annotation = FrameAnnotation.load(annotation_path)
        if annotation.review_status != "approved" or not annotation.objects:
            continue
        frame = _load_frame(annotation)
        masks = _masks(frame, annotation)
        key = f"{_safe(Path(annotation.source_session).name)}_{annotation.frame_id}"
        for variant, config in variants:
            sample = RGBDAugmentor(config, seed=args.seed).augment(
                frame, masks, annotation.objects,
            )
            _save(args.out / key / variant, sample, str(annotation_path), variant)
            manifest["outputs"] += 1
        for angle in rotations:
            variant = f"15_full_rotation_{angle:+d}".replace("+", "plus").replace("-", "minus")
            sample = RGBDAugmentor(
                _base_config(rotation_choices=(float(angle),)), seed=args.seed,
            ).augment(frame, masks, annotation.objects)
            _save(args.out / key / variant, sample, str(annotation_path), variant)
            manifest["outputs"] += 1
        for object_index, obj in enumerate(annotation.objects):
            variant = f"16_object_rotation_{object_index:02d}_90"
            sample = RGBDAugmentor(seed=args.seed).rotate_object(
                frame, masks[object_index], obj, 90, cleanup_margin_px=18,
            )
            sample.masks = [
                sample.masks[0] if index == object_index else mask
                for index, mask in enumerate(masks)
            ]
            sample.annotations = [
                sample.annotations[0] if index == object_index else other
                for index, other in enumerate(annotation.objects)
            ]
            _save(args.out / key / variant, sample, str(annotation_path), variant)
            manifest["outputs"] += 1
        for style in ("solid", "gradient", "noise"):
            variant = f"17_background_{style}"
            augmentor = RGBDAugmentor(seed=args.seed)
            sample = _background_sample(
                augmentor, frame, masks, annotation.objects, style,
            )
            _save(
                args.out / key / variant, sample, str(annotation_path), variant,
                physical_rgbd=False,
            )
            manifest["outputs"] += 1
        manifest["source_frames"] += 1
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
