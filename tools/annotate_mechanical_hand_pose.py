#!/usr/bin/env python3
"""17-point editor with orange-tip snapping for RGB-D capture sessions."""

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
from armsoft.mechanical_hand import (  # noqa: E402
    KEYPOINT_NAMES, SKELETON, OrangeFingertipTracker, PoseAnnotation,
    PosePoint, bbox_from_mask, empty_keypoints,
)

TIP_INDICES = {4, 7, 10, 13, 16}


def derived_masks(path: Path) -> dict[str, Path]:
    result = {}
    with (path / "manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            result[record["id"]] = path / record["mask"]
    return result


def render(image: np.ndarray, annotation: PoseAnnotation, current: int) -> np.ndarray:
    canvas = image.copy()
    x1, y1, x2, y2 = (int(v) for v in annotation.bbox_xyxy)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 180, 0), 2)
    for x, y in annotation.fingertip_candidates_xy:
        cv2.circle(canvas, (int(x), int(y)), 11, (0, 140, 255), 2)
    for start, end in SKELETON:
        a, b = annotation.keypoints[int(start)], annotation.keypoints[int(end)]
        if a.visibility and b.visibility:
            cv2.line(canvas, (int(a.x), int(a.y)), (int(b.x), int(b.y)),
                     (60, 220, 60), 2)
    for index, point in enumerate(annotation.keypoints):
        if not point.visibility:
            continue
        color = (0, 255, 0) if point.visibility == 2 else (0, 255, 255)
        cv2.circle(canvas, (int(point.x), int(point.y)), 5, color, -1)
        cv2.putText(canvas, str(index), (int(point.x) + 5, int(point.y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (0, 0, 0), -1)
    cv2.putText(canvas, f"{current + 1}/17: {KEYPOINT_NAMES[current]}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(canvas, "left=visible right=occluded x=absent u=back a=approve s=save q=quit",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1)
    return canvas


def edit(image: np.ndarray, annotation: PoseAnnotation, snap_radius: float) -> str:
    state = {"current": next((i for i, p in enumerate(annotation.keypoints)
                              if p.visibility == 0), 0)}
    window = "Mechanical hand pose annotation"

    def place(event, x, y, _flags, _param):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        index = state["current"]
        px, py, source = float(x), float(y), "manual"
        if index in TIP_INDICES and annotation.fingertip_candidates_xy:
            candidates = np.asarray(annotation.fingertip_candidates_xy, dtype=float)
            distances = np.linalg.norm(candidates - np.array([x, y]), axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= snap_radius:
                px, py = candidates[nearest]
                source = "orange-snap"
        visibility = 2 if event == cv2.EVENT_LBUTTONDOWN else 1
        annotation.keypoints[index] = PosePoint(px, py, visibility, source)
        state["current"] = min(index + 1, len(KEYPOINT_NAMES) - 1)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, place)
    while True:
        cv2.imshow(window, render(image, annotation, state["current"]))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window)
            return "quit"
        if key == ord("x"):
            annotation.keypoints[state["current"]] = PosePoint()
            state["current"] = min(state["current"] + 1, len(KEYPOINT_NAMES) - 1)
        elif key in (ord("u"), 8):
            state["current"] = max(0, state["current"] - 1)
        elif key == ord("a"):
            annotation.review_status = "approved"
            cv2.destroyWindow(window)
            return "save"
        elif key == ord("r"):
            annotation.review_status = "rejected"
            cv2.destroyWindow(window)
            return "save"
        elif key == ord("s"):
            annotation.review_status = "pending"
            cv2.destroyWindow(window)
            return "save"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snap-radius", type=float, default=45.0)
    parser.add_argument("--bootstrap-only", action="store_true",
                        help="create templates/candidates without opening windows")
    parser.add_argument("--all", action="store_true", help="revisit approved/rejected frames")
    args = parser.parse_args()

    output = args.out / "annotations"
    output.mkdir(parents=True, exist_ok=True)
    masks = derived_masks(args.derived)
    tracker = OrangeFingertipTracker(smoothing=0.0)
    created = edited = 0
    for source_record, frame in read_capture_session(args.session):
        frame_id = source_record["id"]
        path = output / f"{frame_id}.json"
        if path.exists():
            annotation = PoseAnnotation.load(path)
        else:
            mask_path = masks.get(frame_id)
            if mask_path is None:
                raise KeyError(f"derived mask missing for {frame_id}")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
            tips, _ = tracker.detect(frame.color_bgr)
            annotation = PoseAnnotation(
                frame_id=frame_id,
                source_session=str(args.session.resolve()),
                source_color=source_record["color"],
                width=frame.color_bgr.shape[1], height=frame.color_bgr.shape[0],
                bbox_xyxy=bbox_from_mask(mask > 0, frame.color_bgr.shape[1],
                                         frame.color_bgr.shape[0]),
                keypoints=empty_keypoints(),
                fingertip_candidates_xy=[[tip.center_px[0], tip.center_px[1]] for tip in tips],
            )
            annotation.save(path)
            created += 1
        if args.bootstrap_only or (not args.all and annotation.review_status != "pending"):
            continue
        action = edit(frame.color_bgr, annotation, args.snap_radius)
        if action == "quit":
            break
        annotation.save(path)
        edited += 1
    print(f"[pose-annotation] created {created} template(s), edited {edited}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
