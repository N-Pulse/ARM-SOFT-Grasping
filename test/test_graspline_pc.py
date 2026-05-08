"""
test_graspline_pc.py  —  run GraspNet on a .ply file and display the grasp fork.

Usage:
    python test_graspline_pc.py --ply scene.ply --checkpoint /path/to/checkpoint.tar
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "capture"))

import numpy as np
import open3d as o3d

from grasp_pipeline_graspnet import load_model, infer, cluster_and_select

# ── Gripper geometry constants ────────────────────────────────────────────────

FINGER_LENGTH = 0.04
PALM_DEPTH    = 0.06
GRIPPER_COLOR = [1.0, 0.3, 0.0]


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
        trans,        # 5
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

def run(ply_path, checkpoint, device="cuda"):
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"Loaded {len(pcd.points)} points from '{ply_path}'")

    print("Loading GraspNet model...")
    model = load_model(checkpoint, device=device)

    print("Running inference...")
    trans, rot, scores, widths = infer(model, pcd, device)
    best_rot, best_trans, best_width = cluster_and_select(trans, rot, scores, widths)

    geoms = [pcd]
    if best_rot is not None:
        print(f"Best grasp — trans: {best_trans.round(3)}  width: {best_width:.3f}")
        geoms.append(_gripper_lineset(best_rot, best_trans, best_width))
    else:
        print("No valid grasp found.")

    o3d.visualization.draw_geometries(
        geoms,
        window_name="GraspNet — PLY grasp result",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply",        required=True,  help="Input .ply point cloud.")
    parser.add_argument("--checkpoint", required=True,  help="GraspNet checkpoint (.tar).")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    run(args.ply, args.checkpoint, device=args.device)
