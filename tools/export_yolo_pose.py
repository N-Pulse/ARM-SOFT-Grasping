#!/usr/bin/env python3
"""Export reviewed 17-point annotations into a YOLO Pose dataset split."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armsoft.mechanical_hand import KEYPOINT_NAMES, PoseAnnotation, to_yolo_pose_line  # noqa: E402


def write_data_yaml(root: Path) -> None:
    keypoint_names = "\n".join(f"    - {json.dumps(name)}" for name in KEYPOINT_NAMES)
    uniform_sigma = 1.0 / len(KEYPOINT_NAMES)
    text = (
        f"path: {root.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: mechanical_hand\n"
        f"kpt_shape: [{len(KEYPOINT_NAMES)}, 3]\n"
        f"flip_idx: [{', '.join(str(i) for i in range(len(KEYPOINT_NAMES)))}]\n"
        f"kpt_oks_sigmas: [{', '.join(f'{uniform_sigma:.6f}' for _ in KEYPOINT_NAMES)}]\n"
        f"kpt_names:\n  0:\n{keypoint_names}\n"
    )
    (root / "data.yaml").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--include-pending", action="store_true")
    args = parser.parse_args()

    image_dir = args.out / "images" / args.split
    label_dir = args.out / "labels" / args.split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    skipped = 0
    for path in sorted((args.project / "annotations").glob("*.json")):
        annotation = PoseAnnotation.load(path)
        allowed = annotation.review_status == "approved" or (
            args.include_pending and annotation.review_status == "pending")
        if not allowed:
            skipped += 1
            continue
        source = Path(annotation.source_session) / annotation.source_color
        image_name = f"{Path(annotation.source_session).name}__{annotation.frame_id}.png"
        shutil.copy2(source, image_dir / image_name)
        (label_dir / f"{Path(image_name).stem}.txt").write_text(
            to_yolo_pose_line(annotation) + "\n", encoding="utf-8"
        )
        manifest.append({"image": image_name, "annotation": str(path.resolve())})
    write_data_yaml(args.out)
    (args.out / f"manifest_{args.split}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[pose-export] exported {len(manifest)}; skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
