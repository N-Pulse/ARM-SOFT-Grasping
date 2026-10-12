#!/usr/bin/env python3
"""Create object-mask proposals from depth components, optionally refined by SAM."""

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
from armsoft.object_segmentation.annotations import (  # noqa: E402
    FrameAnnotation, ObjectAnnotation, mask_bbox, mask_to_polygon,
    masks_from_components,
)


def load_derived_masks(path: Path) -> dict[str, Path]:
    result = {}
    with (path / "manifest.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            result[record["id"]] = path / record["mask"]
    return result


def sam_refine(model, image: np.ndarray, candidates: list[np.ndarray],
               valid_workspace: np.ndarray) -> list[np.ndarray]:
    boxes = [mask_bbox(mask) for mask in candidates]
    results = model.predict(image, bboxes=boxes, verbose=False)
    if not results or results[0].masks is None:
        return candidates
    refined = results[0].masks.data.cpu().numpy()
    output = []
    for mask in refined:
        if mask.shape != valid_workspace.shape:
            mask = cv2.resize(mask.astype(np.uint8),
                              (valid_workspace.shape[1], valid_workspace.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        output.append(mask.astype(bool) & valid_workspace)
    return output


def draw_preview(image: np.ndarray, objects: list[ObjectAnnotation]) -> np.ndarray:
    preview = image.copy()
    overlay = image.copy()
    for index, obj in enumerate(objects):
        polygon = np.asarray(obj.polygon_xy, dtype=np.int32)
        color = ((37 * index + 20) % 255, (97 * index + 180) % 255,
                 (53 * index + 80) % 255)
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(preview, [polygon], True, color, 2)
        x, y = polygon.min(axis=0)
        cv2.putText(preview, obj.class_name, (int(x), max(15, int(y))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return cv2.addWeighted(preview, 0.65, overlay, 0.35, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--derived", type=Path, required=True,
                        help="output from preprocess_rgbd_session.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--class-name", default="object")
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--max-instances", type=int, default=20)
    parser.add_argument("--sam-model",
                        help="optional Ultralytics SAM weights, e.g. mobile_sam.pt")
    parser.add_argument("--force", action="store_true",
                        help="replace existing pending annotations")
    args = parser.parse_args()

    annotation_dir = args.out / "annotations"
    preview_dir = args.out / "previews"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    mask_paths = load_derived_masks(args.derived)
    sam = None
    if args.sam_model:
        from ultralytics import SAM  # noqa: PLC0415
        sam = SAM(args.sam_model)

    written = skipped = 0
    for source_record, frame in read_capture_session(args.session):
        frame_id = source_record["id"]
        destination = annotation_dir / f"{frame_id}.json"
        if destination.exists() and not args.force:
            skipped += 1
            continue
        mask_path = mask_paths.get(frame_id)
        if mask_path is None:
            raise KeyError(f"derived mask missing for {frame_id}")
        workspace = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if workspace is None:
            raise FileNotFoundError(mask_path)
        workspace = workspace > 0
        candidate_masks = masks_from_components(
            workspace, min_area_px=args.min_area, max_instances=args.max_instances
        )
        source = "depth-components"
        if sam is not None and candidate_masks:
            candidate_masks = sam_refine(sam, frame.color_bgr, candidate_masks, workspace)
            source = f"sam-box:{args.sam_model}"

        objects = []
        for mask in candidate_masks:
            polygon = mask_to_polygon(mask)
            if len(polygon) < 3:
                continue
            objects.append(ObjectAnnotation(
                class_id=args.class_id,
                class_name=args.class_name,
                polygon_xy=polygon,
                bbox_xyxy=mask_bbox(mask),
                source=source,
            ))
        annotation = FrameAnnotation(
            frame_id=frame_id,
            source_session=str(args.session.resolve()),
            source_color=source_record["color"],
            width=frame.color_bgr.shape[1],
            height=frame.color_bgr.shape[0],
            objects=objects,
        )
        annotation.save(destination)
        cv2.imwrite(str(preview_dir / f"{frame_id}.png"),
                    draw_preview(frame.color_bgr, objects))
        written += 1

    run = {
        "source_session": str(args.session.resolve()),
        "derived_masks": str(args.derived.resolve()),
        "proposal_method": f"sam-box:{args.sam_model}" if sam else "depth-components",
        "class_id": args.class_id,
        "class_name": args.class_name,
        "min_area_px": args.min_area,
        "written": written,
        "skipped_existing": skipped,
    }
    (args.out / "annotation_run.json").write_text(
        json.dumps(run, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[annotate] wrote {written}, preserved {skipped} existing annotation(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
