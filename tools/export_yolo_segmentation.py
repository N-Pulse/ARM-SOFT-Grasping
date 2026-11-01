#!/usr/bin/env python3
"""Export approved editable annotations to a YOLO segmentation split."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armsoft.object_segmentation import FrameAnnotation  # noqa: E402
from armsoft.object_segmentation.export import to_yolo_lines  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--classes", type=Path, default=Path("configs/object_classes.json"))
    parser.add_argument("--include-pending", action="store_true")
    args = parser.parse_args()

    image_dir = args.out / "images" / args.split
    label_dir = args.out / "labels" / args.split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    class_config = json.loads(args.classes.read_text(encoding="utf-8"))
    classes = {int(item["id"]): item["name"] for item in class_config["classes"]}
    exported = skipped = 0
    manifest = []
    for path in sorted((args.project / "annotations").glob("*.json")):
        annotation = FrameAnnotation.load(path)
        allowed = annotation.review_status == "approved" or (
            args.include_pending and annotation.review_status == "pending")
        if not allowed:
            skipped += 1
            continue
        for obj in annotation.objects:
            if classes.get(obj.class_id) != obj.class_name:
                raise ValueError(f"object class {obj.class_id}:{obj.class_name} is not in {args.classes}")
        source = Path(annotation.source_session) / annotation.source_color
        image_name = f"{Path(annotation.source_session).name}__{annotation.frame_id}.png"
        shutil.copy2(source, image_dir / image_name)
        lines = to_yolo_lines(annotation)
        (label_dir / f"{Path(image_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        manifest.append({"image": image_name, "annotation": str(path.resolve())})
        exported += 1

    names = "\n".join(f"  {class_id}: {json.dumps(name)}"
                      for class_id, name in sorted(classes.items()))
    yaml = f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n"
    (args.out / "data.yaml").write_text(yaml, encoding="utf-8")
    (args.out / f"manifest_{args.split}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[export] {exported} {args.split} image(s); skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
