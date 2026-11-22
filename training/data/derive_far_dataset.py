"""
training/data/derive_far_dataset.py
===================================
Build a copy of a YOLO dataset where images are assembled into 640x480 grids, so
objects have the apparent size expected at 50 cm-1 m on the D405.  No image moves
between splits, so val and test still hold the same frozen source images.

    python training/data/derive_far_dataset.py
        training/datasets/openimages6 -> training/datasets/openimages6_far
        + training/data/openimages6_far.yaml

Each val/test image is used once; each train image TRAIN_PASSES times.  This only
simulates scale: the photos are still web photos, not D405 scenes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.data.far_scale import (CANVAS, fit_in_cell, parse_yolo,  # noqa: E402
                                     plan_canvases, shrink_boxes)
from training.data.yolo_format import class_counts  # noqa: E402
from training.paths import DATASETS_DIR, TRAINING_DIR  # noqa: E402

PAD_GREY = 114                          # same padding value as Ultralytics letterbox
TRAIN_PASSES = 3
PASSES = {"train": TRAIN_PASSES, "val": 1, "test": 1}


def build_canvas(base: Path, split: str, grid, ids) -> tuple[np.ndarray, list[str]]:
    canvas = np.full((CANVAS[1], CANVAS[0], 3), PAD_GREY, np.uint8)
    lines = []
    for index, image_id in enumerate(ids):
        src = base / "images" / split / f"{image_id}.jpg"
        img = cv2.imread(str(src))
        if img is None:
            raise ValueError(f"cannot read {src}")
        h, w = img.shape[:2]
        new_w, new_h, dx, dy = placement = fit_in_cell((w, h), grid, index)
        canvas[dy:dy + new_h, dx:dx + new_w] = cv2.resize(img, (new_w, new_h),
                                                          interpolation=cv2.INTER_AREA)
        label = base / "labels" / split / f"{image_id}.txt"
        lines += shrink_boxes(parse_yolo(label.read_text().splitlines()), placement)
    return canvas, lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="openimages6")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    base, name = DATASETS_DIR / args.base, f"{args.base}_far"
    out = DATASETS_DIR / name
    if out.exists():
        shutil.rmtree(out)

    stats = {}
    for split, passes in PASSES.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        ids = [p.stem for p in (base / "images" / split).glob("*.jpg")]
        plan = plan_canvases(ids, passes, args.seed)
        all_lines = []
        for number, (grid, cell_ids) in enumerate(plan):
            canvas, lines = build_canvas(base, split, grid, cell_ids)
            cv2.imwrite(str(out / "images" / split / f"grid{number:05d}.jpg"), canvas,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / split / f"grid{number:05d}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            all_lines += lines
        stats[split] = {"images": len(plan), "source_images": len(ids),
                        "instances": class_counts(all_lines)}
        print(f"[{split}] {stats[split]}")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    cfg = yaml.safe_load((TRAINING_DIR / "data" / f"{args.base}.yaml").read_text())
    cfg["path"] = f"../datasets/{name}"
    (TRAINING_DIR / "data" / f"{name}.yaml").write_text(
        f"# {args.base} assembled into grids: apparent size of objects at 50 cm-1 m "
        f"(derive_far_dataset.py).\n" + yaml.safe_dump(cfg, sort_keys=False))
    print(f"YOLO dataset written to {out}")


if __name__ == "__main__":
    main()
