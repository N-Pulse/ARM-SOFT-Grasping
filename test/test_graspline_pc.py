"""
test_graspline_pc.py  —  live isolated point cloud with GraspNet grasp fork.

Usage:
    python test_graspline_pc.py --checkpoint /path/to/checkpoint.tar [--device cuda]
"""

import sys
import os
import queue
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "capture"))

import numpy as np
import open3d as o3d

from object_isolation import ObjectIsolator
from grasp_pipeline_graspnet import load_model, GraspInferenceThread

# ── Gripper geometry constants ────────────────────────────────────────────────

FINGER_LENGTH = 0.04          # m — finger length along approach
PALM_DEPTH    = 0.06          # m — approach stick behind grasp center
GRIPPER_COLOR = [1.0, 0.3, 0.0]   # orange


def _gripper_lineset(rot, trans, width):
    """
    Two-finger gripper fork as an Open3D LineSet.

    GraspNet convention (graspnetAPI):
      rot[:, 0]  approach direction (toward object)
      rot[:, 1]  closing direction (between fingers)
    """
    approach = rot[:, 0]
    closing  = rot[:, 1]
    half_w   = width / 2.0

    palm_center = trans - approach * PALM_DEPTH
    left_palm   = palm_center + closing * half_w
    right_palm  = palm_center - closing * half_w
    left_tip    = trans       + closing * half_w
    right_tip   = trans       - closing * half_w

    pts = np.array([
        palm_center,  # 0
        left_palm,    # 1
        right_palm,   # 2
        left_tip,     # 3
        right_tip,    # 4
        trans,        # 5 — grasp centre (midpoint between contacts)
    ], dtype=np.float64)

    lines = [
        [0, 5],  # approach stem
        [1, 3],  # left finger
        [2, 4],  # right finger
        [1, 2],  # palm bar
    ]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([GRIPPER_COLOR] * len(lines))
    return ls


# ── Main ──────────────────────────────────────────────────────────────────────

def run(checkpoint, device="cuda"):
    isolator = ObjectIsolator(min_points=50)
    isolator.start()
    print("Waiting for YOLO to load...")
    isolator.ready.wait()
    print("YOLO ready.\n")

    print("Loading GraspNet model...")
    model = load_model(checkpoint, device=device)
    grasp_thread = GraspInferenceThread(model, device=device, interval=0.5)
    print("GraspNet ready.\n")

    vis = o3d.visualization.Visualizer()
    vis.create_window("Stage 1 — Isolated Object + Grasp", width=1280, height=720)

    pcd         = o3d.geometry.PointCloud()
    grasp_geom  = None
    last_result = None
    geom_added  = False

    print("Running — close the window or press Ctrl+C to stop.")

    try:
        while True:
            # ── Update point cloud ────────────────────────────────────────
            try:
                verts, full_colors, obj_verts, obj_colors, _ = \
                    isolator._frame_queue.get(timeout=0.05)

                pts  = obj_verts  if len(obj_verts)  > 0 else verts
                cols = obj_colors if len(obj_colors) > 0 else full_colors

                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(cols)

                if len(obj_verts) > 0:
                    iso_pcd = o3d.geometry.PointCloud()
                    iso_pcd.points = o3d.utility.Vector3dVector(obj_verts)
                    grasp_thread.update_pcd(iso_pcd)

                if not geom_added:
                    vis.add_geometry(pcd)
                    ctr = vis.get_view_control()
                    ctr.set_lookat([0, 0, 0.3])
                    ctr.set_front([0, 0, -1])
                    ctr.set_up([0, -1, 0])
                    ctr.set_zoom(0.25)
                    geom_added = True
                else:
                    vis.update_geometry(pcd)

            except queue.Empty:
                pass

            # ── Update grasp fork ─────────────────────────────────────────
            result = grasp_thread.get_grasp()
            if result is not last_result:
                if grasp_geom is not None:
                    vis.remove_geometry(grasp_geom, reset_bounding_box=False)
                    grasp_geom = None
                if result is not None:
                    grasp_geom = _gripper_lineset(*result)
                    vis.add_geometry(grasp_geom, reset_bounding_box=False)
                last_result = result

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
    parser.add_argument("--checkpoint", required=True,
                        help="Path to GraspNet-baseline checkpoint (.tar).")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.checkpoint, device=args.device)
