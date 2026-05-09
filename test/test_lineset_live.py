"""
test_lineset_live.py

Live GraspNet grasp visualisation on the isolated object point cloud.
ObjectIsolator locks on one object; GraspInferenceThread runs GraspNet
in the background; the best grasp is drawn as a lineset fork over the
live Open3D point cloud viewer.

Usage:
    python test_lineset_live.py --checkpoint /path/to/checkpoint.tar
    python test_lineset_live.py --checkpoint /path/to/checkpoint.tar --device cpu

Controls:
    Close the Open3D window  |  Ctrl+C
"""

import sys
import os
import queue
import argparse

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "capture"))

import numpy as np
import open3d as o3d
import torch

from object_isolation import ObjectIsolator
from grasp_pipeline_graspnet import load_model, GraspInferenceThread


# ── Gripper geometry ──────────────────────────────────────────────────────────

FINGER_LENGTH = 0.085
PALM_DEPTH    = 0.06
GRIPPER_COLOR = [1.0, 0.4, 0.0]


def _gripper_lineset(rot, trans, width):
    approach  = rot[:, 0]
    closing   = rot[:, 1]
    half_w    = width / 2.0

    tip_center = trans + approach * FINGER_LENGTH
    palm_back  = trans - approach * PALM_DEPTH
    left_root  = trans      + closing * half_w
    right_root = trans      - closing * half_w
    left_tip   = tip_center + closing * half_w
    right_tip  = tip_center - closing * half_w

    pts = np.array([
        palm_back, trans, left_root, right_root, left_tip, right_tip,
    ], dtype=np.float64)

    lines = [[0,1],[1,2],[1,3],[2,4],[3,5]]

    ls        = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([GRIPPER_COLOR] * len(lines))
    return ls


def _empty_lineset():
    ls        = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.zeros((2, 3)))
    ls.lines  = o3d.utility.Vector2iVector([[0, 1]])
    ls.colors = o3d.utility.Vector3dVector([[0, 0, 0]])
    return ls


# ── Main ──────────────────────────────────────────────────────────────────────

def run(checkpoint, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — pass --device cpu")

    print("Loading GraspNet model...")
    model = load_model(checkpoint, device=device)

    isolator = ObjectIsolator(min_points=50)
    isolator.start()
    print("Waiting for YOLO to load...")
    isolator.ready.wait()
    print("YOLO ready — opening viewer.\n")
    print("Running — close the Open3D window or Ctrl+C to stop.\n")

    grasp_thread = GraspInferenceThread(model, device=device, interval=0.5)

    pcd       = o3d.geometry.PointCloud()
    lineset   = _empty_lineset()
    geom_added = False

    vis = o3d.visualization.Visualizer()
    vis.create_window("GraspNet — Live Grasp", width=1280, height=720)
    opt = vis.get_render_option()
    opt.point_size       = 2.0
    opt.line_width       = 3.0
    opt.background_color = np.array([1.0, 1.0, 1.0])

    try:
        while True:
            try:
                full_verts, full_colors, obj_verts, obj_colors, _ = \
                    isolator._frame_queue.get(timeout=0.05)
            except queue.Empty:
                if not vis.poll_events():
                    break
                vis.update_renderer()
                continue

            pts  = obj_verts  if len(obj_verts)  > 0 else full_verts
            cols = obj_colors if len(obj_colors) > 0 else full_colors

            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)

            if len(obj_verts) > 0:
                iso = o3d.geometry.PointCloud()
                iso.points = o3d.utility.Vector3dVector(obj_verts)
                iso.colors = o3d.utility.Vector3dVector(obj_colors)
                grasp_thread.update_pcd(iso)

            if not geom_added:
                vis.add_geometry(pcd)
                vis.add_geometry(lineset)
                ctr = vis.get_view_control()
                ctr.set_front([0, 0, -1])
                ctr.set_up([0, -1, 0])
                ctr.set_zoom(0.45)
                geom_added = True
            else:
                vis.update_geometry(pcd)

            grasp = grasp_thread.get_grasp()
            if grasp is not None:
                rot, trans, width = grasp
                new_ls = _gripper_lineset(rot, trans, width)
                lineset.points = new_ls.points
                lineset.lines  = new_ls.lines
                lineset.colors = new_ls.colors
                vis.update_geometry(lineset)

            pts_arr = np.asarray(pcd.points)
            if len(pts_arr):
                vis.get_view_control().set_lookat(pts_arr.mean(axis=0).tolist())

            if not vis.poll_events():
                break
            vis.update_renderer()

    except KeyboardInterrupt:
        pass
    finally:
        isolator.stop()
        vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="GraspNet checkpoint (.tar)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.checkpoint, device=args.device)
