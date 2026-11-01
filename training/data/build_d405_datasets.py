"""
training/data/build_d405_datasets.py
====================================
Build two YOLO datasets from a downloaded Dataset Studio data root:

    python training/data/build_d405_datasets.py --root ../ARM-SOFT-data-douaa

    training/datasets/d405              approved D405 frames only, split by session
    training/datasets/openimages6_d405  openimages6 + D405 train (repeated) and val

Only approved frames of the sessions in studio_format.SESSION_SPLITS are used.
The mixed dataset keeps every class present while the model adapts to D405 images.
The D405 test split stays out of it and is used for evaluation only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.data.studio_format import (SESSION_SPLITS, frame_lines,  # noqa: E402
                                         raw_session_name, sessions_are_disjoint)
from training.data.yolo_format import class_counts  # noqa: E402
from training.paths import DATASETS_DIR  # noqa: E402

TRAIN_REPEAT = 5                        # D405 is about 6 % of the train images otherwise


def approved_frames(root: Path, session: str):
    folder = root / "annotations" / "object_segmentation" / session / "annotations"
    paths = sorted(folder.glob("*.json"))
    if not paths:
        raise SystemExit(f"no annotation for session {session} under {folder}")
    for path in paths:
        annotation = json.loads(path.read_text(encoding="utf-8"))
        if annotation["review_status"] == "approved":
            raw = raw_session_name(annotation["source_session"])
            yield root / "raw" / "object" / raw / annotation["source_color"], annotation


def write_frame(src: Path, lines: list[str], out: Path, split: str, stem: str) -> None:
    shutil.copy2(src, out / "images" / split / f"{stem}{src.suffix}")
    (out / "labels" / split / f"{stem}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""))


def make_tree(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    for split in SESSION_SPLITS:
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="Dataset Studio data root")
    ap.add_argument("--base", default="openimages6")
    args = ap.parse_args(argv)
    if not sessions_are_disjoint(SESSION_SPLITS):
        raise SystemExit("a session appears in two splits")

    d405, mixed = DATASETS_DIR / "d405", DATASETS_DIR / f"{args.base}_d405"
    make_tree(d405)
    if mixed.exists():
        shutil.rmtree(mixed)
    shutil.copytree(DATASETS_DIR / args.base, mixed,
                    ignore=shutil.ignore_patterns("*.cache", "stats.json"))

    stats = {}
    for split, sessions in SESSION_SPLITS.items():
        all_lines, frames = [], 0
        for session in sessions:
            for src, annotation in approved_frames(args.root, session):
                lines = frame_lines(annotation)
                stem = f"{session}__{annotation['frame_id']}"
                write_frame(src, lines, d405, split, stem)
                all_lines += lines
                frames += 1
                if split == "train":
                    for n in range(TRAIN_REPEAT):
                        write_frame(src, lines, mixed, split, f"{stem}__r{n}")
                elif split == "val":
                    write_frame(src, lines, mixed, split, stem)
        stats[split] = {"images": frames, "sessions": sessions,
                        "instances": class_counts(all_lines)}
        print(f"[{split}] {stats[split]}")
    download = args.root / "DOWNLOAD.json"
    if download.exists():
        stats["source"] = json.loads(download.read_text(encoding="utf-8"))
    (d405 / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"written: {d405} and {mixed}")


if __name__ == "__main__":
    main()
