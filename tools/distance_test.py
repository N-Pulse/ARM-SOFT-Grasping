"""
tools/distance_test.py
======================
Guided camera range test: one object, several distances, one summary table.

For each target distance the live view is shown so the object can be placed;
SPACE then records N frames, runs the model on each, and stores everything:

    <out>/<object>_<distance>cm/
        frames/          raw .npz frames (replayable with --source replay)
        results.jsonl    one JSON object per frame: model result + raw metrics
        preview.png      annotated view of the last frame
        meta.json        object, target distance and true dimensions

After the last distance, every run under <out> is summarised into
<out>/summary.csv and printed.

Distances are measured from the camera's front glass to the FRONT FACE of
the object.  The model's `distance_m` is to the grasp point (the object's
centre), so it is compared with  front distance + depth/2.

    python tools/distance_test.py --object cylinder --width-mm 60 --height-mm 100
    python tools/distance_test.py --object cuboid --width-mm 50 --height-mm 120 --depth-mm 50
    python tools/distance_test.py --summarize-only          # just rebuild summary.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armsoft.core import isolation                          # noqa: E402
from armsoft.core.classifier import build_classifier        # noqa: E402
from armsoft.core.pipeline import GraspPipeline             # noqa: E402
from armsoft.sources import create_frame_source, save_frame  # noqa: E402
from live_preview import depth_view, text                    # noqa: E402

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "models", "shape_classifier.pt")
WIN = "ARM-SOFT distance test"


def process_quiet(pipeline, frame):
    """Run the model without its per-frame debug prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return pipeline.process(frame)


# ── per-frame raw metrics (independent of the shape fit) ─────────────────────

def raw_metrics(frame, iso) -> dict:
    """Sensor-level numbers for the red object: depth, size in pixels, fill."""
    h, w = frame.depth_m.shape
    out = {"mask_px": 0, "mask_depth_m": None, "depth_fill": None,
           "bbox": None, "cut_off": None}
    if iso is None or iso.mask is None or iso.box is None:
        return out
    mask = iso.mask
    d = frame.depth_m[mask]
    good = d[d > 0]
    x1, y1, x2, y2 = (int(v) for v in iso.box)
    out.update(
        mask_px=int(mask.sum()),
        mask_depth_m=float(np.median(good)) if good.size else None,
        depth_fill=float(good.size / max(d.size, 1)),
        bbox=[x1, y1, x2, y2],
        cut_off=bool(x1 <= 1 or y1 <= 1 or x2 >= w - 2 or y2 >= h - 2),
    )
    return out


# ── capture ──────────────────────────────────────────────────────────────────

def overlay(frame, pipeline, result, lines, max_depth) -> np.ndarray:
    iso = pipeline.last_isolation
    left = (iso.preview_bgr if iso is not None and iso.preview_bgr is not None
            else frame.color_bgr).copy()
    right = depth_view(frame.depth_m, max_depth)
    h, w = frame.depth_m.shape
    patch = frame.depth_m[h // 2 - 3:h // 2 + 4, w // 2 - 3:w // 2 + 4]
    valid = patch[patch > 0]
    for img in (left, right):
        cv2.drawMarker(img, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
    text(right, "centre depth: " + (f"{np.median(valid):.3f} m" if valid.size
                                    else "no reading"), 20)
    for i, s in enumerate(lines):
        text(left, s, 20 + 20 * i, (0, 255, 255))
    if result is not None:
        if result.ok:
            text(left, f"{result.shape}  d={result.distance_m:.3f}m  "
                       f"w={result.object_width_m * 1e3:.0f}mm  "
                       f"h={result.object_height_m * 1e3:.0f}mm  "
                       f"stable={result.stable}", h - 10, (0, 255, 0))
        else:
            text(left, result.status, h - 10, (60, 60, 255))
    return np.hstack([left, right])


def run_distance(source, classifier, args, dist_cm, frames_iter) -> str:
    """Position, then record one distance.  Returns 'done', 'skip' or 'quit'."""
    pipeline = GraspPipeline(table_normal=args.normal, classifier=classifier)
    prompt = [f"{args.object}: place FRONT FACE at {dist_cm} cm",
              "SPACE = record   N = skip   Q = quit"]

    # 1. positioning
    print(f"[test] {dist_cm} cm: place the object, then click the camera window "
          f"and press SPACE (N = skip, Q = quit)")
    while True:
        frame = next(frames_iter)
        result = process_quiet(pipeline, frame)
        cv2.imshow(WIN, overlay(frame, pipeline, result, prompt, args.depth_view))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return "quit"
        if key == ord("n"):
            return "skip"
        if key in (32, 13):
            break

    # 2. recording — a fresh pipeline so the stability gate starts empty
    run_dir = os.path.join(args.out, f"{args.object}_{dist_cm:03d}cm")
    os.makedirs(os.path.join(run_dir, "frames"), exist_ok=True)
    meta = {"object": args.object, "distance_front_cm": dist_cm,
            "width_mm": args.width_mm, "height_mm": args.height_mm,
            "depth_mm": args.depth_mm, "frames": args.frames,
            "max_depth_m": args.max_depth, "table_normal": list(args.normal)}
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    pipeline = GraspPipeline(table_normal=args.normal, classifier=classifier)
    with open(os.path.join(run_dir, "results.jsonl"), "w", encoding="utf-8") as f:
        for i in range(args.frames):
            frame = next(frames_iter)
            save_frame(frame, os.path.join(run_dir, "frames", f"frame_{i:05d}.npz"))
            result = process_quiet(pipeline, frame)
            row = result.to_dict()
            row.pop("gripper_points", None)
            row["test"] = raw_metrics(frame, pipeline.last_isolation)
            row["test"]["confidence"] = getattr(classifier, "last_confidence", None)
            f.write(json.dumps(row) + "\n")
            cv2.imshow(WIN, overlay(frame, pipeline, result,
                                    [f"RECORDING {dist_cm} cm  {i + 1}/{args.frames}",
                                     "do not move anything"], args.depth_view))
            cv2.waitKey(1)
    if pipeline.last_isolation is not None:
        cv2.imwrite(os.path.join(run_dir, "preview.png"),
                    pipeline.last_isolation.preview_bgr)
    s = summarize_run(run_dir)
    print(f"[test] {dist_cm} cm recorded -> {run_dir}  "
          f"(detected {s['detected_frames_%']}%, stable {s['stable_frames_%']}%)")
    return "done"


# ── summary ──────────────────────────────────────────────────────────────────

def _stat(vals, fn):
    vals = [v for v in vals if v is not None]
    return fn(vals) if vals else None


def summarize_run(run_dir: str) -> dict | None:
    meta_path = os.path.join(run_dir, "meta.json")
    res_path = os.path.join(run_dir, "results.jsonl")
    if not (os.path.exists(meta_path) and os.path.exists(res_path)):
        return None
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    with open(res_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        return None

    ok = [r for r in rows if r["status"] == "ok"]
    front_m = meta["distance_front_cm"] / 100.0
    centre_m = front_m + (meta["depth_mm"] or 0) / 2000.0
    w_true = meta["width_mm"]
    h_true = meta["height_mm"]
    mm = lambda v: None if v is None else round(v * 1e3, 1)   # noqa: E731

    seen = [r for r in rows if r["test"]["mask_px"] > 0]
    raw_depth = [r["test"]["mask_depth_m"] for r in seen]
    widths = [r["object_width_m"] for r in ok]
    heights = [r["object_height_m"] for r in ok]
    dists = [r["distance_m"] for r in ok]
    pos = np.array([r["position_m"] for r in ok]) if ok else np.zeros((0, 3))
    times = [r["timing_ms"].get("total") for r in rows if r.get("timing_ms")]

    mean = lambda v: float(np.mean(v))   # noqa: E731
    out = {
        "object": meta["object"],
        "true_distance_cm": meta["distance_front_cm"],
        # estimates — sensor depth of the red pixels, model width / height
        "est_distance_cm": _stat(raw_depth, lambda v: round(mean(v) * 100, 1)),
        "est_width_mm": _stat(widths, lambda v: mm(mean(v))),
        "est_height_mm": _stat(heights, lambda v: mm(mean(v))),
        # classification accuracy
        "class_accuracy_%": (round(100 * sum(r["shape"] == meta["object"] for r in ok)
                             / len(ok), 1) if ok else None),
        # depth and distance errors
        "depth_error_mm": _stat(raw_depth, lambda v: mm(mean(v) - front_m)),
        "depth_noise_mm": _stat(raw_depth, lambda v: mm(float(np.std(v)))),
        "grasp_distance_error_mm": _stat(dists, lambda v: mm(mean(v) - centre_m)),
        # object-dimension errors
        "width_error_mm": (_stat(widths, lambda v: round(mean(v) * 1e3 - w_true, 1))
                           if w_true else None),
        "height_error_mm": (_stat(heights, lambda v: round(mean(v) * 1e3 - h_true, 1))
                            if h_true else None),
        # grasp-position stability
        "grasp_position_std_mm": (mm(float(np.std(pos, axis=0).max()))
                                  if len(pos) > 1 else None),
        "stable_frames_%": round(100 * sum(bool(r["stable"]) for r in rows) / len(rows), 1),
        # field of view and occlusion
        "detected_frames_%": round(100 * len(ok) / len(rows), 1),
        "object_size_px": _stat([r["test"]["mask_px"] for r in seen],
                                lambda v: int(np.median(v))),
        "cut_off_by_edge_%": _stat([r["test"]["cut_off"] for r in seen],
                                   lambda v: round(100 * mean(v), 1)),
        # processing time
        "time_mean_ms": _stat(times, lambda v: round(mean(v), 1)),
        "time_max_ms": _stat(times, lambda v: round(max(v), 1)),
    }
    return out


def summarize(out_dir: str) -> None:
    rows = [s for d in sorted(glob.glob(os.path.join(out_dir, "*_*cm")))
            if (s := summarize_run(d)) is not None]
    if not rows:
        print(f"[test] no runs found under {out_dir}")
        return
    rows.sort(key=lambda r: (r["object"], r["true_distance_cm"]))
    path = os.path.join(out_dir, "summary.csv")
    try:
        f = open(path, "w", newline="", encoding="utf-8")
    except PermissionError:          # usually: summary.csv is open in Excel
        path = os.path.join(out_dir, f"summary_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        print("[test] summary.csv is locked (open in Excel?) - writing a new file")
        f = open(path, "w", newline="", encoding="utf-8")
    with f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    cols = list(rows[0])
    fmt = lambda v: "-" if v is None else str(v)   # noqa: E731
    widths = [max(len(c), *(len(fmt(r[c])) for r in rows)) for c in cols]
    print("\n" + "  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(fmt(r[c]).rjust(w) for c, w in zip(cols, widths)))
    print(f"\n[test] summary written to {path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Guided camera range test.")
    p.add_argument("--object", choices=["cylinder", "cuboid"])
    p.add_argument("--width-mm", type=float, default=50.0, help="true object width")
    p.add_argument("--height-mm", type=float, default=50.0, help="true object height")
    p.add_argument("--depth-mm", type=float, default=None,
                   help="true size along the camera axis (default: = width)")
    p.add_argument("--distances", default="10,30,50,75,100,150,200",
                   help="comma-separated distances in cm, to the object's front face")
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--table-normal", default="0,-1,0")
    p.add_argument("--max-depth", type=float, default=2.5,
                   help="model depth cut-off for this test, metres (normal runs: 0.70)")
    p.add_argument("--depth-view", type=float, default=2.0,
                   help="depth (m) at the far end of the colour scale")
    p.add_argument("--classifier", default="auto",
                   choices=["auto", "yolo", "geometric", "fixed"])
    p.add_argument("--source", default="camera", choices=["camera", "sim"])
    p.add_argument("--out", default=os.path.join("tests", "d405"))
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()

    if args.summarize_only:
        summarize(args.out)
        return 0
    if args.object is None:
        p.error("--object is required (unless --summarize-only)")
    if args.depth_mm is None:
        args.depth_mm = args.width_mm
    args.normal = np.array([float(x) for x in args.table_normal.split(",")])

    # The model ignores depth beyond 0.70 m by default; lift it for the test so
    # far distances measure the camera, not that cut-off.
    isolation.MAX_DEPTH_M = args.max_depth
    print(f"[test] model depth window: {isolation.MIN_DEPTH_M}-{args.max_depth} m")

    classifier = build_classifier(args.classifier, model_path=os.path.abspath(WEIGHTS))
    source = create_frame_source(args.source)
    source.open()
    try:
        frames_iter = source.frames(limit=None)
        for dist_cm in (int(float(x)) for x in args.distances.split(",")):
            if run_distance(source, classifier, args, dist_cm, frames_iter) == "quit":
                break
    except KeyboardInterrupt:
        print("\n[test] interrupted")
    finally:
        source.close()
        cv2.destroyAllWindows()

    summarize(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
