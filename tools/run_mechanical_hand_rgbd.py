#!/usr/bin/env python3
"""Run YOLO Pose on aligned RGB-D frames and emit metric 3-D keypoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armsoft.data import read_capture_session  # noqa: E402
from armsoft.mechanical_hand import KEYPOINT_NAMES, SKELETON, fuse_pose_3d  # noqa: E402
from armsoft.preprocessing import DepthFilterConfig, valid_depth_mask  # noqa: E402
from armsoft.sources import create_frame_source  # noqa: E402


def values(text: str, count: int) -> tuple[float, ...]:
    result = tuple(float(item) for item in text.split(","))
    if len(result) != count:
        raise argparse.ArgumentTypeError(f"expected {count} comma-separated values")
    return result


def draw_pose(image: np.ndarray, poses) -> np.ndarray:
    canvas = image.copy()
    for pose in poses:
        for start, end in SKELETON:
            a, b = pose.keypoints[int(start)], pose.keypoints[int(end)]
            if a.xyz_m is not None and b.xyz_m is not None:
                cv2.line(canvas, tuple(int(v) for v in a.pixel_xy),
                         tuple(int(v) for v in b.pixel_xy), (40, 220, 40), 2)
        for point in pose.keypoints:
            xy = tuple(int(v) for v in point.pixel_xy)
            if point.xyz_m is None:
                cv2.circle(canvas, xy, 4, (0, 0, 255), 1)
                continue
            cv2.circle(canvas, xy, 5, (0, 255, 0), -1)
            cv2.putText(canvas, f"{point.name}:{point.xyz_m[2]:.3f}m", (xy[0] + 5, xy[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", choices=("camera", "sim", "replay"), default="camera")
    parser.add_argument("--replay-path", help="legacy .npz replay directory")
    parser.add_argument("--session", type=Path, help="new RGB-D capture session; overrides --source")
    parser.add_argument("--frames", type=int, default=0, help="0 means until source ends/Q")
    parser.add_argument("--model-conf", type=float, default=0.25)
    parser.add_argument("--keypoint-conf", type=float, default=0.25)
    parser.add_argument("--max-depth-radius", type=int, default=8)
    parser.add_argument("--min-depth", type=float, default=0.07)
    parser.add_argument("--max-depth", type=float, default=1.50)
    parser.add_argument("--roi", type=lambda s: values(s, 4))
    parser.add_argument("--table-plane", type=lambda s: values(s, 4))
    parser.add_argument("--min-table-height", type=float, default=0.005)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-preview", type=Path)
    args = parser.parse_args()
    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")
    if args.jsonl and args.jsonl.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.jsonl}")
    if args.source == "replay" and not args.replay_path and args.session is None:
        raise SystemExit("--replay-path is required with --source replay")

    from ultralytics import YOLO  # late import keeps --help lightweight
    model = YOLO(str(args.weights))
    config = DepthFilterConfig(
        args.min_depth, args.max_depth, args.roi, args.table_plane,
        args.min_table_height,
    )
    json_stream = args.jsonl.open("w", encoding="utf-8") if args.jsonl else None
    processed = 0
    last_preview = None
    try:
        with ExitStack() as stack:
            if args.session is not None:
                iterator = (frame for _, frame in read_capture_session(args.session))
            else:
                kwargs = {"path": args.replay_path} if args.source == "replay" else {}
                source = stack.enter_context(create_frame_source(args.source, **kwargs))
                iterator = source.frames(limit=None if args.frames == 0 else args.frames)
            for frame in iterator:
                workspace = valid_depth_mask(frame.depth_m, config, frame.intrinsics)
                result = model.predict(frame.color_bgr, conf=args.model_conf, verbose=False)[0]
                poses = []
                if result.keypoints is not None and result.keypoints.xy is not None:
                    all_xy = result.keypoints.xy.cpu().numpy()
                    all_conf = result.keypoints.conf
                    if all_conf is None:
                        all_conf = np.ones(all_xy.shape[:2], dtype=np.float32)
                    else:
                        all_conf = all_conf.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((len(all_xy), 4))
                    box_conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(all_xy))
                    for xy, confidence, box, instance_confidence in zip(
                            all_xy, all_conf, boxes, box_conf):
                        if xy.shape[0] != len(KEYPOINT_NAMES):
                            raise ValueError(
                                f"model outputs {xy.shape[0]} keypoints; expected {len(KEYPOINT_NAMES)}"
                            )
                        poses.append(fuse_pose_3d(
                            frame, xy, confidence,
                            instance_confidence=float(instance_confidence),
                            bbox_xyxy=[float(value) for value in box],
                            valid_mask=workspace,
                            min_pose_confidence=args.keypoint_conf,
                            max_radius_px=args.max_depth_radius,
                        ))
                payload = {
                    "frame_index": frame.index, "timestamp": frame.timestamp,
                    "source": frame.source, "instances": [pose.to_dict() for pose in poses],
                }
                if json_stream:
                    json_stream.write(json.dumps(payload, allow_nan=False) + "\n")
                    json_stream.flush()
                else:
                    valid = sum(point.xyz_m is not None for pose in poses for point in pose.keypoints)
                    print(f"[{frame.index:06d}] instances={len(poses)} valid_3d_keypoints={valid}")
                last_preview = draw_pose(frame.color_bgr, poses)
                processed += 1
                if args.show:
                    cv2.imshow("Mechanical hand RGB-D pose - Q/Esc to quit", last_preview)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                if args.frames and processed >= args.frames:
                    break
    finally:
        if json_stream:
            json_stream.close()
        cv2.destroyAllWindows()
    if args.save_preview and last_preview is not None:
        args.save_preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_preview), last_preview)
    print(f"[rgbd-pose] processed {processed} frame(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
