"""
tools/noise_sweep.py
====================
Robustness check: how the model degrades as sensor noise rises.

    python tools/noise_sweep.py                    # default sweep
    python tools/noise_sweep.py --levels 0,1,2,5   # depth noise in mm
    python tools/noise_sweep.py --repeats 5 --frames 12

For every (scene, noise level) it runs `repeats` independent sequences with
different random seeds and reports, over the last frame of each:

    detect      fraction of runs that produced a grasp at all
    correct     fraction that got the shape right
    conf        mean classifier confidence
    width/height  mean fitted size and its spread, against the true size
    pos spread  std-dev of the grasp point across runs — the number that
                matters for the robot, since the arm is sent this position
    stable      fraction that passed the 8-frame stability gate

Writes `noise_sweep.md` and `noise_sweep.png` next to the scene images.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from armsoft.core import GraspPipeline, build_classifier
from armsoft.sources import SimulatedSource

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(HERE, "models", "shape_classifier.pt")

#: Noise levels in millimetres of depth std-dev.  0.8 mm is the default; a real
#: D405 is roughly 1-3 mm at these ranges, and 10 mm is well past plausible.
DEFAULT_LEVELS = [0.0, 0.5, 0.8, 1.5, 3.0, 5.0, 10.0]

SCENES = [
    ("cylinder", dict(shape="cylinder", distance_m=0.35, diameter_m=0.06,
                      height_m=0.10, yaw_deg=20)),
    ("cuboid",   dict(shape="cuboid",   distance_m=0.35, diameter_m=0.06,
                      height_m=0.10, yaw_deg=20)),
    ("cylinder_far", dict(shape="cylinder", distance_m=0.55, diameter_m=0.06,
                          height_m=0.10, yaw_deg=20)),
]


def run(scene, noise_mm, seed, frames, classifier):
    src = SimulatedSource(n_frames=frames, noise_m=noise_mm / 1000.0,
                          seed=seed, **scene)
    pipe = GraspPipeline(table_normal=src.table_normal_hint,
                         classifier=classifier)
    result = None
    for frame in src.frames():
        result = pipe.process(frame)
    return result, getattr(classifier, "last_confidence", None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default=None,
                    help="comma-separated depth-noise std-devs in mm")
    ap.add_argument("--repeats", type=int, default=5, help="seeds per level")
    ap.add_argument("--frames", type=int, default=12, help="frames per run")
    ap.add_argument("--classifier", default="auto",
                    choices=["auto", "yolo", "geometric"])
    ap.add_argument("--out", default=os.path.join(HERE, "temp-sim-res"))
    args = ap.parse_args()

    levels = ([float(x) for x in args.levels.split(",")] if args.levels
              else DEFAULT_LEVELS)
    os.makedirs(args.out, exist_ok=True)
    classifier = build_classifier(args.classifier, model_path=WEIGHTS)

    table = defaultdict(dict)
    for name, scene in SCENES:
        for noise in levels:
            rows = [run(scene, noise, seed, args.frames, classifier)
                    for seed in range(args.repeats)]
            ok = [r for r, _ in rows if r.ok]
            confs = [c for _, c in rows if c is not None]
            stat = {
                "detect":  len(ok) / len(rows),
                "correct": sum(r.shape == scene["shape"] for r in ok) / len(rows),
                "conf":    float(np.mean(confs)) if confs else float("nan"),
                "stable":  sum(r.stable for r in ok) / len(rows),
            }
            if ok:
                w = np.array([r.object_width_m for r in ok]) * 1e3
                h = np.array([r.object_height_m for r in ok]) * 1e3
                p = np.array([r.position_m for r in ok])
                stat |= {"w_mean": w.mean(), "w_std": w.std(),
                         "h_mean": h.mean(), "h_std": h.std(),
                         "pos_spread_mm": float(p.std(axis=0).max() * 1e3),
                         "d_mean": float(np.mean([r.distance_m for r in ok]))}
            table[name][noise] = stat
            print(f"  {name:14s} noise={noise:4.1f}mm  "
                  f"detect={stat['detect']:.0%}  correct={stat['correct']:.0%}  "
                  f"conf={stat['conf']:.2f}  stable={stat['stable']:.0%}"
                  + (f"  w={stat['w_mean']:.0f}±{stat['w_std']:.0f}mm"
                     f"  pos_spread={stat['pos_spread_mm']:.1f}mm" if ok else ""))

    # ── report ───────────────────────────────────────────────────────────────
    md = os.path.join(args.out, "noise_sweep.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Noise robustness\n\n")
        f.write(f"`tools/noise_sweep.py` — {args.repeats} seeds per level, "
                f"{args.frames} frames per run, {classifier.name} classifier. "
                f"Noise is the per-pixel depth std-dev; the simulator's default "
                f"is 0.8 mm and a real D405 is roughly 1-3 mm at these ranges.\n\n")
        for name, scene in SCENES:
            true_w = scene["diameter_m"] * 1e3
            true_h = scene["height_m"] * 1e3
            f.write(f"## {name} — true {true_w:.0f} x {true_h:.0f} mm "
                    f"at {scene['distance_m']:.2f} m\n\n")
            f.write("| noise (mm) | detected | shape correct | conf | "
                    "fitted w (mm) | fitted h (mm) | grasp-point spread (mm) | "
                    "stable |\n|---|---|---|---|---|---|---|---|\n")
            for noise in levels:
                s = table[name][noise]
                if "w_mean" in s:
                    w = f"{s['w_mean']:.0f} ± {s['w_std']:.0f}"
                    h = f"{s['h_mean']:.0f} ± {s['h_std']:.0f}"
                    sp = f"{s['pos_spread_mm']:.1f}"
                else:
                    w = h = sp = "—"
                f.write(f"| {noise:.1f} | {s['detect']:.0%} | {s['correct']:.0%} | "
                        f"{s['conf']:.2f} | {w} | {h} | {sp} | {s['stable']:.0%} |\n")
            f.write("\n")
    print(f"\nwrote {os.path.relpath(md, HERE)}")

    # ── plot ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    xs = levels
    for (name, scene), colour in zip(SCENES, ["C0", "C1", "C2"]):
        axes[0].plot(xs, [table[name][n]["correct"] * 100 for n in xs],
                     "o-", color=colour, label=name)
        axes[1].errorbar(xs, [table[name][n].get("w_mean", np.nan) for n in xs],
                         yerr=[table[name][n].get("w_std", 0) for n in xs],
                         fmt="o-", color=colour, capsize=3, label=name)
        axes[1].axhline(scene["diameter_m"] * 1e3, color=colour, ls=":", lw=1)
        axes[2].plot(xs, [table[name][n].get("pos_spread_mm", np.nan) for n in xs],
                     "o-", color=colour)
        axes[3].plot(xs, [table[name][n]["stable"] * 100 for n in xs],
                     "o-", color=colour)
    for ax, t, y in zip(axes,
                        ["shape classified correctly",
                         "fitted width vs. true (dotted)",
                         "grasp-point spread across seeds",
                         "passed the stability gate"],
                        ["%", "mm", "mm", "%"]):
        ax.set_title(t); ax.set_xlabel("depth noise std-dev (mm)")
        ax.set_ylabel(y); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Noise robustness — {args.repeats} seeds x {args.frames} frames "
                 f"per point; a real D405 sits around 1-3 mm", fontsize=10)
    png = os.path.join(args.out, "noise_sweep.png")
    fig.savefig(png, dpi=110)
    print(f"wrote {os.path.relpath(png, HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
