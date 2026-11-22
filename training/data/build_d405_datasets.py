"""
training/data/build_d405_datasets.py
====================================
Build two YOLO datasets from a downloaded Dataset Studio data root:

    python training/data/build_d405_datasets.py --root ../ARM-SOFT-data-douaa

    training/datasets/d405              approved D405 frames only, split by session
    training/datasets/openimages6_d405  openimages6 + D405 train (repeated) and val

Only approved frames of the sessions in studio_format.SESSION_SPLITS are used,
minus studio_format.EXCLUDED_FRAMES (rejected after visual control).
The mixed dataset keeps every class present while the model adapts to D405 images.
The D405 test split stays out of it and is used for evaluation only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.data.studio_format import (EXCLUDED_FRAMES, SESSION_SPLITS, frame_lines,  # noqa: E402
                                         is_excluded, raw_session_name, sessions_are_disjoint)
from training.data.yolo_format import class_counts  # noqa: E402
from training.paths import DATASETS_DIR  # noqa: E402

TRAIN_REPEAT = 5                        # D405 is about 6 % of the train images otherwise


def approved_frames(root: Path, session: str, excluded=None):
    folder = root / "annotations" / "object_segmentation" / session / "annotations"
    paths = sorted(folder.glob("*.json"))
    if not paths:
        raise SystemExit(f"no annotation for session {session} under {folder}")
    for path in paths:
        annotation = json.loads(path.read_text(encoding="utf-8"))
        if (annotation["review_status"] == "approved"
                and not is_excluded(session, annotation["frame_id"], excluded)):
            raw = raw_session_name(annotation["source_session"])
            yield root / "raw" / "object" / raw / annotation["source_color"], annotation


def copy_or_link(src: Path, dst: Path, link_mode: str = "copy") -> str:
    """Copy an image, or hard-link it with a portable copy fallback."""
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return str(dst)
        except OSError:
            pass
    shutil.copy2(src, dst)
    return str(dst)


def write_frame(src: Path, lines: list[str], out: Path, split: str, stem: str,
                link_mode: str = "copy") -> None:
    if not src.is_file():
        raise FileNotFoundError(f"source image does not exist: {src}")
    copy_or_link(src, out / "images" / split / f"{stem}{src.suffix}", link_mode)
    (out / "labels" / split / f"{stem}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""))


def make_tree(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    for split in SESSION_SPLITS:
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)


def build_datasets(root: Path, base: Path, d405: Path, mixed: Path,
                   train_repeat: int = TRAIN_REPEAT, link_mode: str = "copy",
                   session_splits=None, excluded_frames=None, class_map=None) -> dict:
    """Build D405-only and mixed YOLO datasets at explicit output paths.

    ``session_splits``, ``excluded_frames`` and ``class_map`` default to the values in
    ``studio_format``; the versioned pipeline passes them from its config file.
    """
    session_splits = SESSION_SPLITS if session_splits is None else session_splits
    excluded_frames = EXCLUDED_FRAMES if excluded_frames is None else excluded_frames
    if set(session_splits) != {"train", "val", "test"}:
        raise ValueError("session_splits must define exactly train, val and test")
    if train_repeat < 1:
        raise ValueError("train_repeat must be at least 1")
    if link_mode not in {"copy", "hardlink"}:
        raise ValueError("link_mode must be copy or hardlink")
    if not base.is_dir():
        raise FileNotFoundError(f"base YOLO dataset does not exist: {base}")
    if not sessions_are_disjoint(session_splits):
        raise ValueError("a session appears in two splits")

    make_tree(d405)
    if mixed.exists():
        shutil.rmtree(mixed)
    shutil.copytree(base, mixed, ignore=shutil.ignore_patterns("*.cache", "stats.json"),
                    copy_function=(lambda src, dst: copy_or_link(Path(src), Path(dst), link_mode)))

    stats = {}
    for split, sessions in session_splits.items():
        all_lines, frames = [], 0
        for session in sessions:
            for src, annotation in approved_frames(root, session, excluded_frames):
                lines = frame_lines(annotation, class_map)
                stem = f"{session}__{annotation['frame_id']}"
                write_frame(src, lines, d405, split, stem, link_mode)
                all_lines += lines
                frames += 1
                if split == "train":
                    for n in range(train_repeat):
                        write_frame(src, lines, mixed, split, f"{stem}__r{n}", link_mode)
                elif split == "val":
                    write_frame(src, lines, mixed, split, stem, link_mode)
        stats[split] = {"images": frames, "sessions": sessions,
                        "instances": class_counts(all_lines)}
        print(f"[{split}] {stats[split]}")
    stats["excluded_frames"] = excluded_frames
    stats["train_repeat"] = train_repeat
    stats["link_mode"] = link_mode
    download = root / "DOWNLOAD.json"
    if download.exists():
        stats["source"] = json.loads(download.read_text(encoding="utf-8"))
    (d405 / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"written: {d405} and {mixed}")
    return stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="Dataset Studio data root")
    ap.add_argument("--base", default="openimages6")
    ap.add_argument("--d405-name", default="d405")
    ap.add_argument("--mixed-name", help="default: <base>_d405")
    ap.add_argument("--train-repeat", type=int, default=TRAIN_REPEAT)
    ap.add_argument("--link-mode", choices=["copy", "hardlink"], default="copy")
    args = ap.parse_args(argv)
    build_datasets(args.root, DATASETS_DIR / args.base, DATASETS_DIR / args.d405_name,
                   DATASETS_DIR / (args.mixed_name or f"{args.base}_d405"),
                   args.train_repeat, args.link_mode)


if __name__ == "__main__":
    main()
