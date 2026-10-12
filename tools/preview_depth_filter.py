#!/usr/bin/env python3
"""Live visual check of depth/ROI/table filtering before batch export."""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armsoft.preprocessing import DepthFilterConfig, apply_depth_filter  # noqa: E402
from armsoft.sources import create_frame_source  # noqa: E402


def values(text: str, count: int) -> tuple[float, ...]:
    result = tuple(float(item) for item in text.split(","))
    if len(result) != count:
        raise argparse.ArgumentTypeError(f"expected {count} comma-separated values")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("camera", "sim", "replay"), default="camera")
    parser.add_argument("--replay-path")
    parser.add_argument("--min-depth", type=float, default=0.10)
    parser.add_argument("--max-depth", type=float, default=0.80)
    parser.add_argument("--roi", type=lambda s: values(s, 4))
    parser.add_argument("--table-plane", type=lambda s: values(s, 4))
    parser.add_argument("--min-table-height", type=float, default=0.005)
    args = parser.parse_args()
    if args.source == "replay" and not args.replay_path:
        raise SystemExit("--replay-path is required with --source replay")

    config = DepthFilterConfig(
        args.min_depth, args.max_depth, args.roi, args.table_plane,
        args.min_table_height,
    )
    kwargs = {"path": args.replay_path, "loop": True} if args.source == "replay" else {}
    with create_frame_source(args.source, **kwargs) as source:
        for frame in source.frames():
            filtered, mask = apply_depth_filter(
                frame.color_bgr, frame.depth_m, config, frame.intrinsics,
                dim_background=True,
            )
            mask_bgr = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
            canvas = np.hstack((frame.color_bgr, filtered, mask_bgr))
            cv2.putText(canvas, "RAW", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            width = frame.color_bgr.shape[1]
            cv2.putText(canvas, "FILTERED", (width + 12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.putText(canvas, "MASK", (2 * width + 12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.imshow("Depth preprocessing - Q/Esc to quit", canvas)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
