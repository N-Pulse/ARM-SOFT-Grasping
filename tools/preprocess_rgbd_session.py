#!/usr/bin/env python3
"""Create reproducible depth-filter masks and previews from a raw session."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armsoft.data import read_capture_session  # noqa: E402
from armsoft.preprocessing import (DepthFilterConfig, apply_depth_filter,  # noqa: E402
                                   estimate_table_plane)


def floats(value: str, count: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain numbers") from exc
    if len(result) != count:
        raise argparse.ArgumentTypeError(f"{name} needs {count} comma-separated values")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-depth", type=float, default=0.10)
    parser.add_argument("--max-depth", type=float, default=0.80)
    parser.add_argument("--roi", type=lambda s: floats(s, 4, "ROI"),
                        help="normalized x1,y1,x2,y2, for example 0.1,0.1,0.9,0.9")
    parser.add_argument("--table-plane", type=lambda s: floats(s, 4, "table plane"),
                        help="nx,ny,nz,d for n·X+d=0 in camera coordinates")
    parser.add_argument("--auto-table", action="store_true",
                        help="fit the dominant table plane independently for every frame")
    parser.add_argument("--table-roi", type=lambda s: floats(s, 4, "table ROI"),
                        default=(0.03, 0.35, 0.97, 0.97),
                        help="normalized ROI used only to fit the table plane")
    parser.add_argument("--min-table-height", type=float, default=0.005)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output already exists and is not empty: {args.out}")
    masks = args.out / "masks"
    previews = args.out / "previews"
    masks.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)
    if args.auto_table and args.table_plane is not None:
        raise SystemExit("choose either --auto-table or --table-plane, not both")
    rows = iter(read_capture_session(args.session))
    try:
        first = next(rows)
    except StopIteration:
        raise SystemExit("capture session contains no frames")
    table_plane = args.table_plane
    config = DepthFilterConfig(
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        roi_xyxy=args.roi,
        table_plane=table_plane,
        min_height_above_table_m=args.min_table_height,
    )
    (args.out / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    records = []
    for source_record, frame in (first, *rows):
        frame_config = config
        fitted_plane = table_plane
        if args.auto_table:
            fitted_plane = estimate_table_plane(
                frame.depth_m, frame.intrinsics, args.table_roi,
                min_depth_m=args.min_depth, max_depth_m=args.max_depth,
                seed=int(source_record["source_index"]),
            )
            frame_config = DepthFilterConfig(
                min_depth_m=args.min_depth, max_depth_m=args.max_depth,
                roi_xyxy=args.roi, table_plane=fitted_plane,
                min_height_above_table_m=args.min_table_height,
            )
        filtered, mask = apply_depth_filter(
            frame.color_bgr, frame.depth_m, frame_config, frame.intrinsics,
            dim_background=True,
        )
        frame_id = source_record["id"]
        mask_rel = Path("masks") / f"{frame_id}.png"
        preview_rel = Path("previews") / f"{frame_id}.png"
        if not cv2.imwrite(str(args.out / mask_rel), mask.astype(np.uint8) * 255,
                           [cv2.IMWRITE_PNG_COMPRESSION, 9]):
            raise OSError(f"could not write {mask_rel}")
        if not cv2.imwrite(str(args.out / preview_rel), filtered,
                           [cv2.IMWRITE_PNG_COMPRESSION, 9]):
            raise OSError(f"could not write {preview_rel}")
        records.append({
            "id": frame_id,
            "source_session": str(args.session),
            "source_color": source_record["color"],
            "source_depth": source_record["depth"],
            "mask": mask_rel.as_posix(),
            "preview": preview_rel.as_posix(),
            "kept_fraction": float(mask.mean()),
            "table_plane": fitted_plane,
        })

    with (args.out / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"[preprocess] wrote {len(records)} derived frame(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
