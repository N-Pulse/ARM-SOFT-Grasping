"""
training/data/download_lvis.py
==============================
Download a subset of LVIS v1 for our six classes and write it in
YOLO format, like download_openimages.py does for Open Images:

    training/datasets/lvis6/
        images/{train,val,test}/lvis_<id>.jpg
        labels/{train,val,test}/lvis_<id>.txt
        stats.json          counts, rejected images and the selection rules

    python training/data/download_lvis.py                  # 500 img/class
    python training/data/download_lvis.py --per-class 20   # quick smoke test

Only the LVIS annotation files (~1.1 GB unzipped) and the picked COCO images are
downloaded, into --cache. No account and no FiftyOne are needed.
Images are picked per class, but ALL boxes of our classes in a picked image are kept.
Rejected: images not exhaustively labelled for one of their classes, with more than
--max-objects boxes, with two boxes overlapping more than --max-overlap IoU, or whose
object is smaller than --min-area of the image (rules in lvis_format.py).
LVIS annotations are CC BY 4.0; the images keep their COCO / Flickr licences.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.data.lvis_format import (DEFAULT_RULES, LVIS_TO_CLASS, category_map,  # noqa: E402
                                       check_image, image_stem, select_per_class,
                                       visible_classes, yolo_lines)
from training.data.yolo_format import CLASSES, class_counts, split_ids  # noqa: E402
from training.paths import DATASETS_DIR  # noqa: E402

ANNOTATION_URL = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_{split}.json.zip"


def fetch(url: str, dest: Path, attempts: int = 3) -> Path:
    """Download ``url`` to ``dest`` atomically (``.part`` then rename), with retries."""
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:
                shutil.copyfileobj(response, out)
            part.replace(dest)
            return dest
        except OSError:
            part.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)
    return dest


def annotation_file(split: str, cache: Path) -> Path:
    path = cache / f"lvis_v1_{split}.json"
    if not path.is_file():
        archive = fetch(ANNOTATION_URL.format(split=split), cache / f"lvis_v1_{split}.json.zip")
        print(f"[unzip] {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            zf.extract(path.name, cache)
    return path


def collect(source_splits, cache: Path, rules: dict, mapping=LVIS_TO_CLASS):
    """→ (images {id: record}, candidates {class: [id]}, rejected Counter)"""
    images, candidates, rejected = {}, defaultdict(list), Counter()
    for split in source_splits:
        path = annotation_file(split, cache)
        print(f"[read] {path.name}")
        data = json.loads(path.read_text(encoding="utf-8"))
        cat_map = category_map(data["categories"], mapping)
        cat_ids_of = defaultdict(set)
        for cat_id, cls in cat_map.items():
            cat_ids_of[cls].add(cat_id)
        boxes = defaultdict(list)
        for ann in data["annotations"]:
            cls = cat_map.get(ann["category_id"])
            if cls is not None:
                boxes[ann["image_id"]].append((cls, ann["bbox"]))
        for image in data["images"]:
            found = boxes.get(image["id"], [])
            if not found:
                continue
            reason = check_image(image, found, cat_ids_of, rules)
            visible = visible_classes(image, found, rules["min_area"]) if reason is None else set()
            if reason is None and not visible:
                reason = "too_small"
            if reason:
                rejected[reason] += 1
                continue
            images[image["id"]] = {"url": image["coco_url"], "width": image["width"],
                                   "height": image["height"], "boxes": found}
            for cls in visible:
                candidates[cls].append(image["id"])
        del data
        print(f"[{split}] usable images so far: {len(images)}, rejected: {dict(rejected)}")
    return images, candidates, rejected


def download_images(picked, images: dict, cache: Path, workers: int) -> None:
    folder = cache / "images"
    jobs = [(images[i]["url"], folder / f"{image_stem(i)}.jpg") for i in picked]
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(lambda job: fetch(*job), jobs):
            done += 1
            if done % 200 == 0 or done == len(jobs):
                print(f"[images] {done}/{len(jobs)}")


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_yolo(picked: dict, images: dict, cache: Path, out: Path, seed: int,
               settings: dict, rejected: Counter, candidates: dict) -> dict:
    if out.exists():
        shutil.rmtree(out)
    stems = {image_stem(i): i for i in picked}
    stats = {}
    for split, split_stems in split_ids(stems, seed=seed).items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        all_lines = []
        for stem in split_stems:
            image_id = stems[stem]
            record = images[image_id]
            lines = yolo_lines(record["boxes"], record["width"], record["height"])
            link_or_copy(cache / "images" / f"{stem}.jpg", out / "images" / split / f"{stem}.jpg")
            (out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            all_lines += lines
        stats[split] = {"images": len(split_stems), "instances": class_counts(all_lines)}
    stats["picked_per_class"] = dict(Counter(picked.values()))
    stats["candidates_per_class"] = {cls: len(set(candidates.get(cls, []))) for cls in CLASSES}
    stats["rejected_images"] = dict(rejected)
    stats["settings"] = settings
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DATASETS_DIR / "lvis6")
    ap.add_argument("--cache", type=Path, default=DATASETS_DIR / "_lvis_cache")
    ap.add_argument("--source-splits", nargs="+", default=["train", "val"],
                    choices=["train", "val"], help="LVIS splits to pick from (we re-split ourselves)")
    ap.add_argument("--max-objects", type=int, default=DEFAULT_RULES["max_objects"])
    ap.add_argument("--max-overlap", type=float, default=DEFAULT_RULES["max_overlap"])
    ap.add_argument("--min-area", type=float, default=DEFAULT_RULES["min_area"])
    ap.add_argument("--exclude", type=int, nargs="*", default=[],
                    help="LVIS image ids rejected after visual control")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    if args.per_class < 1:
        ap.error("--per-class must be at least 1")

    rules = {"max_objects": args.max_objects, "max_overlap": args.max_overlap,
             "min_area": args.min_area}
    images, candidates, rejected = collect(args.source_splits, args.cache, rules)
    excluded = set(args.exclude)
    if excluded:
        rejected["excluded_by_config"] = len(excluded & set(images))
        candidates = {cls: [i for i in ids if i not in excluded] for cls, ids in candidates.items()}
    picked = select_per_class(candidates, args.per_class, args.seed)
    print(f"[pick] {len(picked)} images: {dict(Counter(picked.values()))}")
    download_images(picked, images, args.cache, args.workers)
    settings = {"per_class": args.per_class, "seed": args.seed, "rules": rules,
                "source_splits": args.source_splits, "exclude": sorted(excluded),
                "class_map": LVIS_TO_CLASS}
    stats = write_yolo(picked, images, args.cache, args.out, args.seed, settings, rejected,
                       candidates)
    print(json.dumps({k: stats[k] for k in ("train", "val", "test", "picked_per_class",
                                            "rejected_images")}, indent=2))
    print(f"YOLO dataset written to {args.out}")


if __name__ == "__main__":
    main()
