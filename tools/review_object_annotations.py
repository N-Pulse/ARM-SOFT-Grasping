#!/usr/bin/env python3
"""Keyboard review of proposal previews: A approve, R reject, S skip, Q quit."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armsoft.object_segmentation import FrameAnnotation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path,
                        help="directory containing annotations/ and previews/")
    parser.add_argument("--all", action="store_true", help="also revisit reviewed frames")
    args = parser.parse_args()

    files = sorted((args.project / "annotations").glob("*.json"))
    for path in files:
        annotation = FrameAnnotation.load(path)
        if not args.all and annotation.review_status != "pending":
            continue
        preview = cv2.imread(str(args.project / "previews" / f"{annotation.frame_id}.png"))
        if preview is None:
            raise FileNotFoundError(f"preview missing for {annotation.frame_id}")
        title = (f"{annotation.frame_id} | {len(annotation.objects)} object(s) | "
                 "A approve, R reject, S skip, Q quit")
        cv2.imshow(title, preview)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(title)
        if key in (ord("q"), 27):
            break
        if key == ord("a"):
            annotation.review_status = "approved"
            annotation.save(path)
        elif key == ord("r"):
            annotation.review_status = "rejected"
            annotation.save(path)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
