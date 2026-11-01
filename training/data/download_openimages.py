"""
training/data/download_openimages.py
====================================
Download a 4-class subset of Open Images V7 (ball, can, box, bottle) with
FiftyOne and write it in YOLO format:

    training/datasets/openimages4/
        images/{train,val,test}/<id>.jpg
        labels/{train,val,test}/<id>.txt
        stats.json

    python training/data/download_openimages.py                 # ~500 img/class
    python training/data/download_openimages.py --per-class 50  # quick smoke test

Images are picked per class (so every class gets ~--per-class images), but ALL
boxes of our four classes in each picked image are kept: a "Ball" image that
also shows a bottle gets both labels.  Detections flagged IsGroupOf or
IsDepiction are skipped (training/decisions.md); the image is kept, and an
image left with no box becomes a background-only example.  The first run downloads Open
Images metadata CSVs (~2 GB for the train split) into the FiftyOne cache below.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.data.yolo_format import (CLASSES, OI_TO_CLASS, class_counts,  # noqa: E402
                                       filter_detections, oi_labels_for,
                                       split_ids, yolo_lines)
from training.paths import DATASETS_DIR  # noqa: E402

FLAGS = ("IsGroupOf", "IsDepiction")


def check_classes() -> None:
    from fiftyone.utils.openimages import get_classes
    available = set(get_classes(version="v7"))
    missing = [oi for oi in OI_TO_CLASS if oi not in available]
    if missing:
        sys.exit(f"Open Images V7 has no boxable class {missing}; "
                 f"edit OI_TO_CLASS in training/data/yolo_format.py")


def _flags(det) -> dict:
    """OI box attributes as stored by FiftyOne (absent field → not set)."""
    return {k: bool(det[k]) for k in FLAGS if det.has_field(k) and det[k] is not None}


def collect(per_class: int, oi_split: str, seed: int, cache: Path):
    """→ ({image_id: (filepath, [(oi_label, [x, y, w, h]), ...])}, skipped box counts)"""
    import fiftyone as fo
    import fiftyone.zoo as foz
    # FiftyOne >= 1.x takes the download location from its config, not from a
    # `dataset_dir` argument (that one is forwarded to the importer and clashes).
    fo.config.dataset_zoo_dir = str(cache)
    images, skipped = {}, {"group_of": 0, "depiction": 0}
    for cls in CLASSES:
        ds = foz.load_zoo_dataset(
            "open-images-v7", split=oi_split, label_types=["detections"],
            classes=oi_labels_for(cls), only_matching=False,  # keep every box
            max_samples=per_class, shuffle=True, seed=seed,
            dataset_name=f"oi4-{cls}", drop_existing_dataset=True)
        field = "ground_truth" if ds.has_sample_field("ground_truth") else "detections"
        for sample in ds:
            image_id = Path(sample.filepath).stem
            if image_id in images:
                continue                      # already seen via another class
            dets = sample[field].detections if sample[field] else []
            kept, n_skipped = filter_detections(
                (d.label, d.bounding_box, _flags(d)) for d in dets)
            for k, n in n_skipped.items():
                skipped[k] += n
            images[image_id] = (sample.filepath, kept)
        print(f"[{cls}] {len(ds)} images (unique so far: {len(images)}, "
              f"skipped boxes so far: {skipped})")
    return images, skipped


def write_yolo(images: dict, skipped: dict, out: Path, seed: int) -> dict:
    if out.exists():
        shutil.rmtree(out)
    stats = {}
    for split, ids in split_ids(images, seed=seed).items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        all_lines = []
        for image_id in ids:
            src, dets = images[image_id]
            lines = yolo_lines(dets)
            shutil.copy2(src, out / "images" / split / f"{image_id}.jpg")
            (out / "labels" / split / f"{image_id}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            all_lines += lines
        stats[split] = {"images": len(ids), "instances": class_counts(all_lines)}
    stats["skipped_boxes"] = skipped
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--oi-split", default="train",
                    choices=["train", "validation", "test"],
                    help="Open Images split to sample from (we re-split ourselves)")
    ap.add_argument("--out", type=Path, default=DATASETS_DIR / "openimages4")
    ap.add_argument("--cache", type=Path, default=DATASETS_DIR / "_fiftyone_cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    check_classes()
    images, skipped = collect(args.per_class, args.oi_split, args.seed, args.cache)
    stats = write_yolo(images, skipped, args.out, args.seed)
    print(json.dumps(stats, indent=2))
    print(f"YOLO dataset written to {args.out}")


if __name__ == "__main__":
    main()
