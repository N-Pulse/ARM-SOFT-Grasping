#!/usr/bin/env python3
"""Live/video smoke test for MediaPipe-based human action recognition."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from armsoft.actions import PoseActionRecognizer, draw_prediction  # noqa: E402


DEFAULT_MODEL = ROOT / "models" / "pose_landmarker_full.task"
DEFAULT_GESTURE_MODEL = ROOT / "models" / "gesture_recognizer.task"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recognize simple human poses")
    parser.add_argument("--camera", type=int, default=0,
                        help="OpenCV camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280,
                        help="requested camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="requested camera height (default: 720)")
    parser.add_argument("--video", help="video file instead of a live camera")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--gesture-model", default=str(DEFAULT_GESTURE_MODEL))
    parser.add_argument("--output", help="optional annotated output video")
    parser.add_argument("--headless", action="store_true",
                        help="do not open a window (use with --video)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="0 = run until video ends or Q is pressed")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    source = args.video if args.video else args.camera
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"[actions] cannot open camera/video: {source}")
        return 2

    if not args.video:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[actions] camera resolution: {actual_width}x{actual_height}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps > 240:
        fps = 30.0
    writer = None
    frame_index = 0
    previous = None

    try:
        with PoseActionRecognizer(args.model, args.gesture_model) as recognizer:
            while args.max_frames <= 0 or frame_index < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp_ms = int(frame_index * 1000.0 / fps) if args.video else int(time.monotonic() * 1000)
                prediction = recognizer.process(frame, timestamp_ms)
                annotated = draw_prediction(frame, prediction)
                if prediction.label != previous:
                    print(f"[{frame_index:05d}] {prediction.label} ({prediction.confidence:.0%})")
                    previous = prediction.label
                details = []
                for arm in prediction.arms:
                    angle = "?" if arm.elbow_deg is None else f"{arm.elbow_deg:.0f}deg"
                    details.append(f"{arm.side}_arm={arm.state}/{angle}")
                for hand in prediction.hands:
                    details.append(
                        f"{hand.side}_hand={hand.gesture}/{hand.confidence:.0%}"
                    )
                if details and frame_index % 30 == 0:
                    print("           " + "  ".join(details))

                if args.output:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    writer.write(annotated)
                if not args.headless:
                    cv2.imshow("ARM-SOFT MediaPipe actions — Q to quit", annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                        break
                frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"[actions] processed {frame_index} frame(s)")
    return 0 if frame_index else 1


if __name__ == "__main__":
    raise SystemExit(main())
