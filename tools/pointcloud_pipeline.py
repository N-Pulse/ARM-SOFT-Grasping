#!/usr/bin/env python3
"""Run RGB-D -> filtered coloured point cloud on camera, simulator, or session."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from itertools import islice
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armsoft.data import data_root, read_capture_session  # noqa: E402
from armsoft.pointcloud import PointCloudConfig, PointCloudPipeline, save_ply  # noqa: E402
from armsoft.preprocessing import estimate_table_plane  # noqa: E402
from armsoft.sources import create_frame_source  # noqa: E402


def floats(value: str, count: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain numbers") from exc
    if len(result) != count:
        raise argparse.ArgumentTypeError(
            f"{name} needs {count} comma-separated values")
    return result


class Viewer:
    def __init__(self):
        import open3d as o3d
        self.o3d = o3d
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("ARM-SOFT Point Cloud", 1280, 720)
        self.cloud = o3d.geometry.PointCloud()
        self.added = False

    def update(self, result) -> bool:
        self.cloud.points = self.o3d.utility.Vector3dVector(result.points_xyz_m)
        self.cloud.colors = self.o3d.utility.Vector3dVector(result.colors_rgb)
        if not self.added:
            self.vis.add_geometry(self.cloud); self.added = True
        else:
            self.vis.update_geometry(self.cloud)
        return self.vis.poll_events() and (self.vis.update_renderer() or True)

    def close(self):
        self.vis.destroy_window()


def frame_iterator(args):
    if args.session:
        frames = (frame for _, frame in read_capture_session(args.session))
        yield from islice(frames, args.frames)
        return
    with create_frame_source(args.source) as source:
        yield from source.frames(limit=args.frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("auto", "camera", "sim"),
                        default="auto")
    parser.add_argument("--session", type=Path,
                        help="captured RGB-D session; overrides --source")
    parser.add_argument("--frames", type=int, default=1,
                        help="live/sim frames to process")
    parser.add_argument("--out", type=Path,
                        help="output directory (default: shared data root)")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--min-depth", type=float, default=0.10)
    parser.add_argument("--max-depth", type=float, default=1.50)
    parser.add_argument("--roi", type=lambda s: floats(s, 4, "ROI"))
    parser.add_argument("--table-plane", type=lambda s: floats(s, 4, "table plane"))
    parser.add_argument("--auto-table", action="store_true",
                        help="fit one table plane from the first usable frame")
    parser.add_argument("--table-roi", type=lambda s: floats(s, 4, "table ROI"),
                        default=(0.03, 0.35, 0.97, 0.97))
    parser.add_argument("--min-table-height", type=float, default=0.005)
    parser.add_argument("--voxel-size", type=float, default=0.005,
                        help="metres; 0 disables voxel downsampling")
    parser.add_argument("--keep-outliers", action="store_true",
                        help="disable statistical outlier removal")
    parser.add_argument("--outlier-neighbors", type=int, default=16)
    parser.add_argument("--outlier-std-ratio", type=float, default=2.0)
    parser.add_argument("--max-points", type=int, default=100_000)
    parser.add_argument("--viz", action="store_true",
                        help="show a live Open3D window")
    args = parser.parse_args()
    if args.frames < 1 or args.save_every < 1:
        parser.error("--frames and --save-every must be positive")
    if args.auto_table and args.table_plane:
        parser.error("choose --auto-table or --table-plane, not both")
    source_name = args.session.name if args.session else args.source
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (args.out or data_root()/"derived"/"pointcloud"/
              f"{source_name}_{stamp}").resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"output already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = PointCloudConfig(
        min_depth_m=args.min_depth, max_depth_m=args.max_depth,
        roi_xyxy=args.roi, table_plane=args.table_plane,
        min_height_above_table_m=args.min_table_height,
        voxel_size_m=args.voxel_size, remove_outliers=not args.keep_outliers,
        outlier_neighbors=args.outlier_neighbors,
        outlier_std_ratio=args.outlier_std_ratio, max_points=args.max_points)
    viewer = Viewer() if args.viz else None
    records = []
    try:
        for position, frame in enumerate(frame_iterator(args)):
            if args.auto_table and config.table_plane is None:
                try:
                    plane = estimate_table_plane(
                        frame.depth_m, frame.intrinsics, args.table_roi,
                        min_depth_m=args.min_depth, max_depth_m=args.max_depth,
                        seed=frame.index)
                    config = replace(config, table_plane=plane)
                    print(f"[pointcloud] table plane: {plane}")
                except ValueError as exc:
                    print(f"[pointcloud] table fit skipped: {exc}")
            started = time.perf_counter()
            result = PointCloudPipeline(config).process(frame)
            processing_ms = (time.perf_counter() - started) * 1000
            summary = {"frame_index": frame.index, "timestamp": frame.timestamp,
                       "source": frame.source,
                       "processing_ms": round(processing_ms, 3),
                       **result.summary()}
            print(f"[pointcloud] frame {frame.index}: "
                  f"{summary['valid_points']} valid -> {summary['output_points']} points "
                  f"in {processing_ms:.1f} ms")
            if position % args.save_every == 0:
                name = f"frame_{frame.index:06d}.ply"
                save_ply(output/name, result)
                summary["pointcloud"] = name
                records.append(summary)
            if viewer and not viewer.update(result):
                break
    except KeyboardInterrupt:
        print("\n[pointcloud] stopped")
    finally:
        if viewer: viewer.close()
    (output/"config.json").write_text(
        json.dumps({"schema": 1, "coordinate_frame": "+X right, +Y down, +Z forward; metres",
                    "source": str(args.session.resolve()) if args.session else args.source,
                    "config": config.to_dict()}, indent=2) + "\n", encoding="utf-8")
    with (output/"manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"[pointcloud] saved {len(records)} cloud(s) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
