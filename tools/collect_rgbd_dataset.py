#!/usr/bin/env python3
"""Capture a reproducible RGB-D session from a camera, replay, or simulator."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armsoft.data import CaptureSessionWriter
from armsoft.sources import create_frame_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("auto", "camera", "sim", "replay"),
                        default="auto")
    parser.add_argument("--replay-path", help="directory of legacy .npz frames")
    parser.add_argument("--out", type=Path, help="session directory")
    parser.add_argument("--purpose", choices=("object-segmentation", "mechanical-hand-pose",
                                                "calibration", "general"),
                        default="general")
    parser.add_argument("--frames", type=int, default=100,
                        help="number of frames; 0 means until interrupted")
    parser.add_argument("--every", type=int, default=1,
                        help="save one frame out of every N source frames")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames < 0 or args.every < 1:
        raise SystemExit("--frames must be >= 0 and --every must be >= 1")
    if args.source == "replay" and not args.replay_path:
        raise SystemExit("--replay-path is required with --source replay")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.out or Path("datasets/captures") / f"{stamp}_{args.purpose}"
    kwargs = {"path": args.replay_path} if args.source == "replay" else {}
    source_limit = None if args.frames == 0 else args.frames * args.every

    saved = 0
    writer = CaptureSessionWriter(output, purpose=args.purpose, notes=args.notes)
    try:
        with create_frame_source(args.source, **kwargs) as source:
            for seen, frame in enumerate(source.frames(limit=source_limit)):
                if seen % args.every:
                    continue
                writer.write(frame)
                saved += 1
                print(f"\r[capture] saved {saved} frame(s) to {output}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n[capture] stopped by user")
    finally:
        metadata = writer.close()
    print(f"\n[capture] complete: {metadata['frame_count']} frame(s); {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
