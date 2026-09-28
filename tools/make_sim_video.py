"""
tools/make_sim_video.py
=======================
Animated version of the visual check: a hand-held camera shaking its way in
towards the object, with the model's output updating frame by frame.

    python tools/make_sim_video.py                     # -> temp-sim-res/*.gif
    python tools/make_sim_video.py --shape cuboid --worms 8
    python tools/make_sim_video.py --frames 80 --fps 12 --mp4

Each GIF frame carries the same panels as the still images, plus a live plot of
how the reported parameters move as the camera closes in:

    ┌──────────┬──────────┬──────────┬──────────┬──────────────┐
    │ camera   │ depth    │ fit+grasp│ top-down │ history      │
    └──────────┴──────────┴──────────┴──────────┴──────────────┘

The point is to see the *sequence*: the distance falling, the jaw opening
tracking the object, the shape label flickering or holding, and the stability
gate latching only once the hand settles.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from armsoft import GraspPipeline, SimulatedSource, build_classifier
from armsoft.sources.trajectory import ShakyApproach
from panels import (label, panel_camera, panel_depth, panel_fit, panel_topdown,
                    FONT, GREEN, ORANGE, CYAN, RED, WHITE)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(HERE, "models", "shape_classifier.pt")


def history_panel(hist, n_frames, size=(480, 640)):
    """Live strip chart of distance, jaw opening and the stability flag."""
    h, w = size
    img = np.full((h, w, 3), 18, np.uint8)
    label(img, "5. parameters over time", (10, 22), WHITE, 0.55, 1)

    pad_l, pad_r, pad_t, pad_b = 58, 14, 46, 74
    x0, x1 = pad_l, w - pad_r
    y0, y1 = pad_t, h - pad_b
    cv2.rectangle(img, (x0, y0), (x1, y1), (60, 60, 60), 1)

    def draw(series, lo, hi, colour, name, row):
        if len(series) < 2:
            return
        pts = []
        for i, v in enumerate(series):
            if v is None or not np.isfinite(v):
                continue
            px = int(x0 + (x1 - x0) * i / max(n_frames - 1, 1))
            py = int(y1 - (y1 - y0) * (np.clip(v, lo, hi) - lo) / (hi - lo))
            pts.append((px, py))
        if len(pts) > 1:
            cv2.polylines(img, [np.asarray(pts, np.int32)], False, colour, 2,
                          cv2.LINE_AA)
        if pts:
            cv2.circle(img, pts[-1], 4, colour, -1, cv2.LINE_AA)
        # Current value, inside the plot so it cannot collide with the legend.
        last = next((v for v in reversed(series) if v is not None), None)
        txt = f"{name}: {last:.0f}" if last is not None else f"{name}: --"
        label(img, txt, (x0 + 10, y0 + 20 + row * 20), colour, 0.45, 1)

    draw([None if d is None else d * 1e3 for d in hist["distance"]],
         100, 450, CYAN, "distance mm", 0)
    draw([None if j is None else j * 1e3 for j in hist["jaw"]],
         0, 200, ORANGE, "jaw mm", 1)

    # shape / stability ribbon along the bottom of the plot area
    for i, (shape, stable) in enumerate(zip(hist["shape"], hist["stable"])):
        px = int(x0 + (x1 - x0) * i / max(n_frames - 1, 1))
        nxt = int(x0 + (x1 - x0) * (i + 1) / max(n_frames - 1, 1))
        if shape is None:
            colour = (70, 70, 70)
        elif stable:
            colour = GREEN
        else:
            colour = (80, 180, 255)
        cv2.rectangle(img, (px, y1 + 4), (max(nxt - 1, px), y1 + 16), colour, -1)
    label(img, "grey = no shape   blue = tracking   green = stable",
          (x0, y1 + 34), (190, 190, 190), 0.4, 1)
    label(img, "450", (8, y0 + 8), (150, 150, 150), 0.38, 1)
    label(img, "100", (8, y1), (150, 150, 150), 0.38, 1)
    return img


def frame_caption(width, i, n, result, conf, source):
    """One caption bar per video frame: where the camera is, what came out."""
    bar = np.full((70, width, 3), 28, np.uint8)
    _R, t = source.camera_pose
    label(bar, f"frame {i + 1:3d}/{n}", (12, 28), WHITE, 0.6, 2)
    label(bar, f"camera moved {t[2] * 1e3:+.0f} mm forward,  "
               f"shake {np.linalg.norm(t[:2]) * 1e3:4.1f} mm",
          (190, 26), (190, 190, 190), 0.45, 1)
    if result.ok:
        colour = GREEN if result.stable else (80, 180, 255)
        txt = (f"{result.shape}"
               f"{'' if conf is None else f' {conf:.2f}'}   "
               f"d={result.distance_m:.3f} m   "
               f"w={result.object_width_m * 1e3:.0f} mm   "
               f"jaw={result.jaw_opening_m * 1e3:.0f} mm   "
               f"roll={np.degrees(result.base_roll_rad):+.0f} deg   "
               f"{'STABLE - would publish' if result.stable else 'tracking'}")
    else:
        colour, txt = RED, f"{result.status}"
    label(bar, txt, (190, 54), colour, 0.48, 1)
    return bar


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="cylinder", choices=["cylinder", "cuboid"])
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--start-distance", type=float, default=0.42)
    ap.add_argument("--travel", type=float, default=0.20,
                    help="how far the camera closes in, metres")
    ap.add_argument("--shake-mm", type=float, default=3.5,
                    help="per-frame translation jitter std-dev")
    ap.add_argument("--shake-deg", type=float, default=0.7)
    ap.add_argument("--noise", type=float, default=0.0015,
                    help="depth noise std-dev in metres")
    ap.add_argument("--worms", type=int, default=0)
    ap.add_argument("--no-settle", action="store_true",
                    help="keep shaking to the end instead of steadying")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="output scale per panel (0.5 keeps the GIF small)")
    ap.add_argument("--mp4", action="store_true", help="also write an .mp4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classifier", default="auto",
                    choices=["auto", "yolo", "geometric"])
    ap.add_argument("--out", default=os.path.join(HERE, "temp-sim-res"))
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    name = args.name or f"approach_{args.shape}" + ("_worms" if args.worms else "")

    traj = ShakyApproach(n_frames=args.frames, travel_m=args.travel,
                         shake_m=args.shake_mm / 1000.0, shake_deg=args.shake_deg,
                         settle=not args.no_settle, seed=args.seed)
    source = SimulatedSource(shape=args.shape, distance_m=args.start_distance,
                             noise_m=args.noise, worms=args.worms,
                             n_frames=args.frames, trajectory=traj,
                             seed=args.seed)
    classifier = build_classifier(args.classifier, model_path=WEIGHTS)
    pipeline = GraspPipeline(table_normal=source.table_normal_hint,
                             classifier=classifier)

    hist = {"distance": [], "jaw": [], "shape": [], "stable": []}
    frames_out = []

    for i, frame in enumerate(source.frames()):
        # The camera tilts as it shakes, so the table plane moves in the camera
        # frame.  A real system would re-estimate this; here the simulator knows
        # it exactly, which isolates the model from calibration drift.
        pipeline.table_normal = source.table_normal_hint

        result = pipeline.process(frame)
        conf = getattr(classifier, "last_confidence", None)

        hist["distance"].append(result.distance_m if result.ok else None)
        hist["jaw"].append(result.jaw_opening_m if result.ok else None)
        hist["shape"].append(result.shape)
        hist["stable"].append(bool(result.stable))

        pred = result.shape or getattr(classifier, "last_label", None)
        strip = np.hstack([
            panel_camera(frame, pipeline.last_isolation, pred, conf,
                         result.shape is not None),
            panel_depth(frame),
            panel_fit(frame, pipeline, result),
            panel_topdown(pipeline, result, pipeline.table_normal),
            history_panel(hist, args.frames),
        ])
        img = np.vstack([strip, frame_caption(strip.shape[1], i, args.frames,
                                              result, conf, source)])
        if args.scale != 1.0:
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                             interpolation=cv2.INTER_AREA)
        frames_out.append(img)
        print(f"  frame {i + 1:3d}/{args.frames}  {result.status:9s}"
              f"  d={'' if not result.ok else f'{result.distance_m:.3f}m'}"
              f"  stable={result.stable}")

    # ── GIF ──────────────────────────────────────────────────────────────────
    from PIL import Image
    pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).convert(
        "P", palette=Image.ADAPTIVE, colors=128) for f in frames_out]
    gif = os.path.join(args.out, f"{name}.gif")
    pil[0].save(gif, save_all=True, append_images=pil[1:],
                duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"\nwrote {os.path.relpath(gif, HERE)}  "
          f"({os.path.getsize(gif) / 1e6:.1f} MB, {len(pil)} frames)")

    if args.mp4:
        h, w = frames_out[0].shape[:2]
        mp4 = os.path.join(args.out, f"{name}.mp4")
        vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        for f in frames_out:
            vw.write(f)
        vw.release()
        print(f"wrote {os.path.relpath(mp4, HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
