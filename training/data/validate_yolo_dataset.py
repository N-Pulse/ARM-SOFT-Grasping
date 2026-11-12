"""Validate a YOLO detection dataset before it is used for training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.data.yolo_format import CLASSES  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def _image_files(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES} if folder.is_dir() else {}


def _sha256(path: Path, cache: dict[tuple[int, int, int], str]) -> str:
    stat = path.stat()
    key = (stat.st_dev, stat.st_ino, stat.st_size)
    if key not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        cache[key] = digest.hexdigest()
    return cache[key]


def _image_decoder():
    """Return a function path -> bool that fully decodes an image, or None if unavailable."""
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        cv2 = None
    if cv2 is not None:
        def decode_cv2(path: Path) -> bool:
            data = np.fromfile(str(path), dtype=np.uint8)
            return data.size > 0 and cv2.imdecode(data, cv2.IMREAD_UNCHANGED) is not None
        return decode_cv2
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        return None

    def decode_pil(path: Path) -> bool:
        try:
            with Image.open(path) as image:
                image.load()
            return True
        except Exception:  # noqa: BLE001 — any decoder failure means a corrupt image
            return False
    return decode_pil


def validate_dataset(root: Path, classes: list[str] | None = None,
                     check_images: bool = True) -> dict:
    """Check labels, images, class ids, coordinates and cross-split duplicates.

    With ``check_images`` every distinct image file is fully decoded once (hard links
    to the same file are decoded only once).
    """
    classes = classes or CLASSES
    errors: list[str] = []
    warnings: list[str] = []
    report: dict = {"dataset": str(root.resolve()), "classes": classes, "splits": {}}
    hashes: dict[str, set[str]] = defaultdict(set)
    hash_cache: dict[tuple[int, int, int], str] = {}
    decode = _image_decoder() if check_images else None
    if check_images and decode is None:
        warnings.append("image decoding skipped: neither OpenCV nor Pillow is installed")
    decoded: dict[tuple[int, int, int], bool] = {}
    corrupt: list[str] = []

    for split in SPLITS:
        images = _image_files(root / "images" / split)
        label_dir = root / "labels" / split
        labels = {p.stem: p for p in sorted(label_dir.glob("*.txt"))} if label_dir.is_dir() else {}
        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        if missing_labels:
            errors.append(f"{split}: {len(missing_labels)} images have no label file")
        if missing_images:
            errors.append(f"{split}: {len(missing_images)} label files have no image")

        counts = Counter()
        boxes = 0
        for stem, label_path in labels.items():
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    errors.append(f"{label_path}:{line_number}: expected 5 fields")
                    continue
                try:
                    class_id = int(fields[0])
                    coords = [float(value) for value in fields[1:]]
                except ValueError:
                    errors.append(f"{label_path}:{line_number}: non-numeric YOLO label")
                    continue
                if not 0 <= class_id < len(classes):
                    errors.append(f"{label_path}:{line_number}: invalid class id {class_id}")
                    continue
                if any(value < 0 or value > 1 for value in coords):
                    errors.append(f"{label_path}:{line_number}: coordinates outside [0, 1]")
                    continue
                if coords[2] <= 0 or coords[3] <= 0:
                    errors.append(f"{label_path}:{line_number}: box has zero area")
                    continue
                counts[classes[class_id]] += 1
                boxes += 1

        for image_path in images.values():
            try:
                digest = _sha256(image_path, hash_cache)
            except OSError as exc:
                errors.append(f"cannot read image {image_path}: {exc}")
                continue
            hashes[digest].add(split)
            if decode is not None:
                stat = image_path.stat()
                key = (stat.st_dev, stat.st_ino, stat.st_size)
                if key not in decoded:
                    decoded[key] = decode(image_path)
                if not decoded[key]:
                    corrupt.append(str(image_path))

        report["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": boxes,
            "instances": {name: counts.get(name, 0) for name in classes},
            "missing_labels": missing_labels,
            "missing_images": missing_images,
        }

    if corrupt:
        errors.append(f"{len(corrupt)} images cannot be decoded")
    report["corrupt_images"] = corrupt
    report["images_decoded"] = len(decoded) if decode is not None else None
    leakage = sorted(digest for digest, splits in hashes.items() if len(splits) > 1)
    if leakage:
        errors.append(f"{len(leakage)} image contents occur in more than one split")
    report["cross_split_duplicate_hashes"] = leakage
    report["errors"] = errors
    report["warnings"] = warnings
    report["valid"] = not errors
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-image-decode", action="store_true",
                        help="only hash images; do not decode them")
    args = parser.parse_args(argv)
    report = validate_dataset(args.dataset, check_images=not args.skip_image_decode)
    output = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
