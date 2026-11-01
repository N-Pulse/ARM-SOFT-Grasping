"""
tools/save_sim_results.py
=========================
Render a batch of simulated scenes and save one annotated PNG per scene, so the
pipeline can be checked by eye without a camera or a 3-D window.

    python tools/save_sim_results.py                       # -> temp-sim-res/
    python tools/save_sim_results.py --out some/dir --frames 12

Each PNG is a four-panel strip:

    ┌───────────────┬───────────────┬───────────────┬───────────────┐
    │ 1. camera     │ 2. depth      │ 3. fit+grasp  │ 4. top-down   │
    │ red mask,box, │ colour-mapped │ wireframe and │ bird's-eye of │
    │ label + conf  │ depth image   │ gripper, pro- │ the footprint │
    │               │               │ jected to 2D  │ and the jaws  │
    └───────────────┴───────────────┴───────────────┴───────────────┘

plus a caption line with the numbers the model produced. A `summary.md` and a
contact sheet of every scene are written alongside.
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
from panels import (caption, corruption_note, panel_camera, panel_depth,
                    panel_fit, panel_topdown)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(HERE, "models", "shape_classifier.pt")

#: Corruption levels.  Each scene below is rendered once per variant, and the
#: suffix is appended to the file name, so the noisy images sit next to the
#: clean one in the output folder.
#:
#: "worms" are the structured artifact: short wandering black streaks that also
#: punch the depth out underneath, which is how a real depth camera fails.
#: Unlike gaussian noise they do not average away over frames.
VARIANTS = [
    ("",              dict()),
    ("__noise",       dict(noise_m=0.003)),
    ("__worms",       dict(noise_m=0.002, worms=12)),
    ("__worms_heavy", dict(noise_m=0.004, worms=28, worm_thickness_px=6,
                           depth_holes=6)),
]

#: (name, kwargs for SimulatedSource) — edit freely.
SCENES = [
    ("cylinder_default",   dict(shape="cylinder", distance_m=0.35, yaw_deg=20)),
    ("cylinder_near",      dict(shape="cylinder", distance_m=0.20, yaw_deg=20)),
    ("cylinder_far",       dict(shape="cylinder", distance_m=0.55, yaw_deg=20)),
    ("cylinder_thin_tall", dict(shape="cylinder", distance_m=0.35, diameter_m=0.04,
                                height_m=0.14)),
    ("cuboid_default",     dict(shape="cuboid",   distance_m=0.35, yaw_deg=20)),
    ("cuboid_face_on",     dict(shape="cuboid",   distance_m=0.35, yaw_deg=0)),
    ("cuboid_diagonal",    dict(shape="cuboid",   distance_m=0.35, yaw_deg=45)),
    ("cuboid_wide_flat",   dict(shape="cuboid",   distance_m=0.30, diameter_m=0.09,
                                height_m=0.06, yaw_deg=25)),
]



def run_scene(name, scene, frames, out_dir, classifier):
    src = SimulatedSource(n_frames=frames, **scene)
    pipeline = GraspPipeline(table_normal=src.table_normal_hint,
                             classifier=classifier)

    frame = result = None
    for frame in src.frames():                 # converge the EMA / stability gate
        result = pipeline.process(frame)

    conf = getattr(classifier, "last_confidence", None)
    pred = result.shape or getattr(classifier, "last_label", None)
    accepted = result.shape is not None
    iso = pipeline.last_isolation
    strip = np.hstack([panel_camera(frame, iso, pred, conf, accepted),
                       panel_depth(frame),
                       panel_fit(frame, pipeline, result),
                       panel_topdown(pipeline, result, pipeline.table_normal)])
    img = np.vstack([strip, caption(strip.shape[1], name, scene, result, conf)])

    path = os.path.join(out_dir, f"{name}.png")
    cv2.imwrite(path, img)
    return path, img, result, conf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "temp-sim-res"))
    ap.add_argument("--frames", type=int, default=12,
                    help="frames per scene; the last one is saved")
    ap.add_argument("--classifier", default="auto",
                    choices=["auto", "yolo", "geometric"])
    ap.add_argument("--variants", default="all",
                    help="comma-separated subset of: clean, noise, worms, "
                         "worms_heavy  (default: all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    classifier = build_classifier(args.classifier, model_path=WEIGHTS)

    variants = [(sfx, kw) for sfx, kw in VARIANTS
                if args.variants == "all" or sfx.lstrip("_") in
                [v.strip() for v in args.variants.split(",")] or
                (sfx == "" and "clean" in args.variants)]

    rows, thumbs = [], []
    for name, base_scene in SCENES:
        for suffix, corruption in variants:
            scene = {**base_scene, **corruption}
            path, img, result, conf = run_scene(name + suffix, scene, args.frames,
                                                args.out, classifier)
            thumbs.append(cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4)))
            rows.append((name + suffix, scene, result, conf))
            print(f"  wrote {os.path.relpath(path, HERE)}"
                  f"   {result.status}"
                  f"{'' if not result.ok else f' / {result.shape}'}")

    sheet = os.path.join(args.out, "_contact_sheet.png")
    cv2.imwrite(sheet, np.vstack(thumbs))

    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Simulated scenes\n\n")
        f.write(f"Generated by `tools/save_sim_results.py` "
                f"({classifier.name} classifier, {args.frames} frames per scene; "
                f"the last frame is shown).\n\n")
        f.write("| scene | corruption | true shape | predicted | conf | "
                "true size (mm) | fitted size (mm) | distance (m) | jaw (mm) | "
                "stable |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for name, scene, r, conf in rows:
            tw = scene.get("diameter_m", 0.06) * 1e3
            th = scene.get("height_m", 0.10) * 1e3
            if r.ok:
                pred = f"{r.shape} {'✓' if r.shape == scene['shape'] else '✗'}"
                fit = f"{r.object_width_m * 1e3:.0f} × {r.object_height_m * 1e3:.0f}"
                dist, jaw = f"{r.distance_m:.3f}", f"{r.jaw_opening_m * 1e3:.0f}"
                stable = "yes" if r.stable else "no"
            else:
                pred, fit, dist, jaw, stable = r.status, "—", "—", "—", "—"
            f.write(f"| `{name}` | {corruption_note(scene)} | "
                    f"{scene['shape']} | {pred} | "
                    f"{'—' if conf is None else f'{conf:.2f}'} | "
                    f"{tw:.0f} × {th:.0f} | {fit} | {dist} | {jaw} | {stable} |\n")
        f.write("\nTrue distance is to the object axis; the model reports the "
                "distance to the fitted centroid, so a few millimetres of "
                "difference is expected.\n")

    print(f"\n{len(rows)} scene(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
