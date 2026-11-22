#!/usr/bin/env python3
"""Track the ARM-SOFT mechanical hand's five orange fingertips."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from armsoft.mechanical_hand import OrangeFingertipTracker, draw_fingertips  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track orange mechanical fingertips")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--image", help="process one image and exit")
    source.add_argument("--video", help="process a video instead of a camera")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--hue", default="3,18", help="OpenCV hue range, default 3,18")
    p.add_argument("--saturation-min", type=int, default=150)
    p.add_argument("--value-min", type=int, default=140)
    p.add_argument("--min-area", type=float, default=0.00025,
                   help="minimum component area / image area")
    p.add_argument("--output", help="save annotated image/video")
    p.add_argument("--show-mask", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    hue_min, hue_max = (int(v) for v in args.hue.split(","))
    tracker = OrangeFingertipTracker(
        hue_min=hue_min, hue_max=hue_max,
        saturation_min=args.saturation_min, value_min=args.value_min,
        min_area_ratio=args.min_area,
    )

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"[mechanical-hand] cannot read image: {args.image}")
            return 2
        tips, mask = tracker.detect(image)
        annotated = draw_fingertips(image, tips, mask if args.show_mask else None)
        output = args.output or "mechanical_hand_preview.png"
        cv2.imwrite(output, annotated)
        _print_tips(tips)
        print(f"[mechanical-hand] preview written to {output}")
        return 0 if tips else 1

    source = args.video if args.video else args.camera
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"[mechanical-hand] cannot open camera/video: {source}")
        return 2
    if not args.video:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps > 240:
        fps = 30.0
    writer = None
    frame_index = 0
    try:
        while args.max_frames <= 0 or frame_index < args.max_frames:
            ok, image = capture.read()
            if not ok:
                break
            tips, mask = tracker.detect(image)
            annotated = draw_fingertips(image, tips, mask if args.show_mask else None)
            if frame_index % 30 == 0:
                _print_tips(tips)
            if args.output:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                    )
                writer.write(annotated)
            if not args.headless:
                cv2.imshow("ARM-SOFT orange fingertips — Q to quit", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    print(f"[mechanical-hand] processed {frame_index} frame(s)")
    return 0 if frame_index else 1


def _print_tips(tips) -> None:
    values = "  ".join(
        f"tip{tip.track_id + 1}=({tip.center_px[0]:.0f},{tip.center_px[1]:.0f})"
        for tip in tips
    )
    print(f"[mechanical-hand] {len(tips)}/5  {values}")


if __name__ == "__main__":
    raise SystemExit(main())
