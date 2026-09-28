"""
armsoft/cli.py
==============
Command-line entry point for the grasping model.

Examples
--------
    # one inference, camera if present, otherwise simulation
    python -m armsoft --frames 1

    # force simulation with a cuboid, write JSON lines
    python -m armsoft --source sim --sim-shape cuboid \
        --frames 20 --jsonl out.jsonl

    # live camera + Open3D viewer
    python -m armsoft --source camera --frames 0 --viz

    # record frames now, replay them later without any camera
    python -m armsoft --source camera --frames 30 --record rec/
    python -m armsoft --source replay --replay-path rec/
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import closing

import numpy as np

from .core.classifier import build_classifier
from .core.pipeline import GraspPipeline
from .core.table_plane import detect_table_plane
from .sinks import results as out
from .sources import create_frame_source, save_frame

#: Trained shape-classifier weights shipped with this tree.
DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(__file__), "..", "models", "shape_classifier.pt")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the ARM-SOFT grasping model on any development machine.")

    src = p.add_argument_group("frame source")
    src.add_argument("--source", default="auto",
                     choices=["auto", "camera", "sim", "replay"],
                     help="auto = real camera when reachable, else simulation")
    src.add_argument("--frames", type=int, default=1,
                     help="number of frames to process (0 = until the source stops)")
    src.add_argument("--replay-path", default=None, help="file or dir of .npz frames")
    src.add_argument("--record", default=None,
                     help="directory to save each captured frame into (.npz)")

    sim = p.add_argument_group("simulation")
    sim.add_argument("--sim-shape", default="cylinder", choices=["cylinder", "cuboid"])
    sim.add_argument("--sim-distance", type=float, default=0.35)
    sim.add_argument("--sim-diameter", type=float, default=0.06)
    sim.add_argument("--sim-height", type=float, default=0.10)
    sim.add_argument("--sim-yaw", type=float, default=20.0)
    sim.add_argument("--sim-noise", type=float, default=0.0008,
                     help="gaussian depth-noise std-dev in metres")
    sim.add_argument("--sim-worms", type=int, default=0,
                     help="number of worm-shaped dark dropouts per frame")
    sim.add_argument("--sim-worm-length", type=float, default=60.0,
                     help="worm length in pixels")
    sim.add_argument("--sim-worm-thickness", type=int, default=4,
                     help="worm thickness in pixels")
    sim.add_argument("--sim-worm-keep-depth", action="store_true",
                     help="worms darken the colour image but leave depth intact")
    sim.add_argument("--sim-depth-holes", type=int, default=0,
                     help="round depth dropouts that do not mark the colour image")
    sim.add_argument("--sim-drift", type=float, default=0.0,
                     help="lateral object drift per frame, metres")
    sim.add_argument("--sim-seed", type=int, default=0)

    mdl = p.add_argument_group("model")
    mdl.add_argument("--classifier", default="auto",
                     choices=["auto", "yolo", "geometric", "fixed"])
    mdl.add_argument("--weights", default=DEFAULT_WEIGHTS)
    mdl.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    mdl.add_argument("--fixed-shape", default="cylinder",
                     help="label used by --classifier fixed")
    mdl.add_argument("--table-normal", default=None,
                     help="comma-separated camera-frame normal, e.g. '0,-1,0'")
    mdl.add_argument("--calibrate", action="store_true",
                     help="detect the table plane from a chessboard first")
    mdl.add_argument("--no-smooth", action="store_true",
                     help="disable ShapeEMA temporal smoothing")

    o = p.add_argument_group("output")
    o.add_argument("--jsonl", default=None, help="append one JSON object per frame")
    o.add_argument("--json", action="store_true",
                   help="pretty-print the full result of each frame")
    o.add_argument("--quiet", action="store_true")
    o.add_argument("--ros2", action="store_true",
                   help="also publish the grasp on ROS 2 topics (needs rclpy)")
    o.add_argument("--serial", default=None, help="serial port for the JSON link")
    o.add_argument("--viz", action="store_true",
                   help="live Open3D window (needs a display)")
    o.add_argument("--save-preview", default=None,
                   help="write the annotated camera preview of the last frame here")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    source = create_frame_source(
        args.source,
        path=args.replay_path, shape=args.sim_shape,
        distance_m=args.sim_distance, diameter_m=args.sim_diameter,
        height_m=args.sim_height, yaw_deg=args.sim_yaw,
        noise_m=args.sim_noise, drift_m=args.sim_drift, seed=args.sim_seed,
        worms=args.sim_worms, worm_length_px=args.sim_worm_length,
        worm_thickness_px=args.sim_worm_thickness,
        worm_drop_depth=not args.sim_worm_keep_depth,
        depth_holes=args.sim_depth_holes,
    )

    try:
        source.open()
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        print(f"\n[run] cannot open the '{args.source}' frame source:\n{exc}\n")
        return 2

    # `closing`, not `with source:` — the source is already open, and opening a
    # live camera twice starts its pipeline twice.
    with closing(source):
        # ── table plane ──────────────────────────────────────────────────────
        if args.table_normal:
            table_normal = np.array([float(x) for x in args.table_normal.split(",")])
        elif args.calibrate:
            table_normal, _ = detect_table_plane(source)
        else:
            hint = getattr(source, "table_normal_hint", None)
            table_normal = np.array([0.0, -1.0, 0.0]) if hint is None else np.asarray(hint)
        print(f"[run] table normal = {np.round(table_normal, 3)}")

        classifier = build_classifier(
            args.classifier, model_path=os.path.abspath(args.weights),
            device=args.device, label=args.fixed_shape)
        print(f"[run] shape hint from: {classifier.name}")

        pipeline = GraspPipeline(table_normal=table_normal, classifier=classifier,
                                 smooth=not args.no_smooth)

        sinks: list[out.Sink] = []
        if not args.quiet:
            sinks.append(out.StdoutSink())
        if args.jsonl:
            sinks.append(out.JsonlSink(args.jsonl))
        if args.ros2:
            sinks.append(out.Ros2Sink())
        if args.serial:
            sinks.append(out.SerialSink(args.serial))

        viewer = None
        if args.viz:
            from .sinks.viewer import LiveViewer
            viewer = LiveViewer()

        if args.record:
            os.makedirs(args.record, exist_ok=True)

        limit = None if args.frames <= 0 else args.frames
        n_ok = 0
        last = None
        t_start = time.perf_counter()
        try:
            for frame in source.frames(limit=limit):
                if args.record:
                    save_frame(frame, os.path.join(args.record,
                                                   f"frame_{frame.index:05d}.npz"))
                result = pipeline.process(frame)
                last = result
                n_ok += int(result.ok)
                for s in sinks:
                    s.write(result)
                if args.json:
                    print(result.to_json())
                if viewer is not None and not viewer.update(pipeline, result):
                    break
        except KeyboardInterrupt:
            print("\n[run] interrupted")
        finally:
            for s in sinks:
                s.close()
            if viewer is not None:
                viewer.close()

        if args.save_preview and pipeline.last_isolation is not None:
            import cv2
            cv2.imwrite(args.save_preview, pipeline.last_isolation.preview_bgr)
            print(f"[run] preview written to {args.save_preview}")

        dt = time.perf_counter() - t_start
        print(f"[run] {n_ok} grasp(s) computed in {dt:.2f}s")
        if last is not None and args.quiet and last.ok:
            print(last.to_json())
        return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
