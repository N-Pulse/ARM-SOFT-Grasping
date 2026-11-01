"""
training/data/preview_dataset.py
================================
Visual check of a YOLO dataset before training: one row per class, N random
images of the split that contain that class, with ALL their boxes drawn
(each class in its own colour, so wrong / missing labels stand out).

    python training/data/preview_dataset.py                       # openimages4, train, 6/class
    python training/data/preview_dataset.py --dataset training/datasets/oi4_smoke

Writes training/results/dataset_preview.jpg (JPEG q=85, deterministic for a given --seed).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.data.yolo_format import CLASSES  # noqa: E402
from training.paths import DATASETS_DIR, RESULTS_DIR  # noqa: E402

COLORS = [(0, 165, 255), (255, 80, 0), (0, 200, 0), (200, 0, 200)]  # BGR per class
TILE = 260
LABEL_W = 110


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            c, *xywh = line.split()
            rows.append((int(c), *map(float, xywh)))
    return rows


def tile(img_path: Path, labels) -> np.ndarray:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    for c, xc, yc, bw, bh in labels:
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        th = max(2, round(min(h, w) / 150))
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[c], th)
        cv2.putText(img, CLASSES[c], (x1 + 3, max(y1 + 3 * th + 12, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, min(h, w) / 600), COLORS[c], th)
    s = TILE / max(h, w)                                     # letterbox into a square
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    out = np.full((TILE, TILE, 3), 40, np.uint8)
    y0, x0 = (TILE - img.shape[0]) // 2, (TILE - img.shape[1]) // 2
    out[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    cv2.putText(out, img_path.stem[:12], (4, TILE - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (220, 220, 220), 1)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATASETS_DIR / "openimages4")
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "dataset_preview.jpg")
    args = ap.parse_args(argv)

    label_dir = args.dataset / "labels" / args.split
    image_dir = args.dataset / "images" / args.split
    labels = {p.stem: read_labels(p) for p in sorted(label_dir.glob("*.txt"))}
    rng = random.Random(args.seed)

    rows = []
    for c, name in enumerate(CLASSES):
        ids = [i for i, rows_ in labels.items() if any(r[0] == c for r in rows_)]
        picked = rng.sample(ids, min(args.per_class, len(ids)))
        tiles = [tile(image_dir / f"{i}.jpg", labels[i]) for i in picked]
        tiles += [np.full((TILE, TILE, 3), 40, np.uint8)] * (args.per_class - len(tiles))
        head = np.full((TILE, LABEL_W, 3), 255, np.uint8)
        cv2.putText(head, name, (8, TILE // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[c], 2)
        cv2.putText(head, f"{len(ids)} img", (8, TILE // 2 + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (60, 60, 60), 1)
        rows.append(np.hstack([head, *tiles]))
        print(f"{name}: {len(ids)} {args.split} images contain it, showing {len(picked)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
