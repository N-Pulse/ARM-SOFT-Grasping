"""
test_draw_hand.py  —  run GraspNet on a .ply file and display the grasp fork
                       with a projected dexterous hand overlaid on top.

Usage:
    python test_draw_hand.py --ply scene.ply --checkpoint /path/to/checkpoint.tar
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "capture"))

import torch
import numpy as np
import open3d as o3d

from grasp_pipeline_graspnet import load_model, infer, cluster_and_select
from grasp_projection import GraspProjection

# ── Gripper geometry constants ────────────────────────────────────────────────

FINGER_LENGTH = 0.05
PALM_DEPTH    = 0.06
GRIPPER_COLOR = [1.0, 0.4, 0.0]


def _gripper_lineset(rot, trans, width):
    """
    Two-finger gripper fork as an Open3D LineSet.

    GraspNet convention (graspnetAPI):
      rot[:, 0]  approach direction (toward object)
      rot[:, 1]  closing direction (between fingers)
      trans      palm center; finger tips extend FINGER_LENGTH toward object
    """
    approach = rot[:, 0]
    closing  = rot[:, 1]
    half_w   = width / 2.0

    # Palm at trans; tips extend forward along approach to reach the surface
    tip_center = trans + approach * FINGER_LENGTH
    palm_back  = trans - approach * PALM_DEPTH
    left_root  = trans      + closing * half_w
    right_root = trans      - closing * half_w
    left_tip   = tip_center + closing * half_w
    right_tip  = tip_center - closing * half_w

    pts = np.array([
        palm_back,    # 0
        trans,        # 1  palm center
        left_root,    # 2
        right_root,   # 3
        left_tip,     # 4
        right_tip,    # 5
    ], dtype=np.float64)

    lines = [
        [0, 1],  # approach stem (behind palm)
        [1, 2],  # palm → left root
        [1, 3],  # palm → right root
        [2, 4],  # left finger
        [3, 5],  # right finger
    ]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([GRIPPER_COLOR] * len(lines))
    return ls


# ── Hand geometry (from GraspProjection) ─────────────────────────────────────

def _hand_linesets(rot, trans, width, pcd_pts):
    """Build Open3D geometries for the projected dexterous hand."""
    proj = GraspProjection(rot=rot, trans=trans, width=width, point_cloud=pcd_pts)
    hand = proj.hand_data

    # Skeleton edges → one LineSet
    pts, lines, colors = [], [], []
    for edge in hand.skeleton_edges:
        i = len(pts)
        pts.append(edge.start)
        pts.append(edge.end)
        lines.append([i, i + 1])
        colors.append(edge.color)

    skeleton_ls        = o3d.geometry.LineSet()
    skeleton_ls.points = o3d.utility.Vector3dVector(np.array(pts))
    skeleton_ls.lines  = o3d.utility.Vector2iVector(np.array(lines))
    skeleton_ls.colors = o3d.utility.Vector3dVector(np.array(colors))

    # Contact points → small spheres
    spheres = []
    for contact, color in zip(hand.contacts, hand.finger_colors):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        s.translate(contact)
        s.paint_uniform_color(color)
        s.compute_vertex_normals()
        spheres.append(s)

    return [skeleton_ls] + spheres


# ── Main ──────────────────────────────────────────────────────────────────────

def run(ply_path, checkpoint, device="cuda"):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment. "
            "GraspNet's pointnet2 extension requires a CUDA GPU. "
            "Check that PyTorch was installed with CUDA support: "
            "python -c \"import torch; print(torch.version.cuda)\""
        )

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

        print("Projecting dexterous hand...")
        pcd_pts = np.asarray(pcd.points)
        geoms  += _hand_linesets(best_rot, best_trans, best_width, pcd_pts)
    else:
        print("No valid grasp found.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="GraspNet — hand projection  (Ctrl+C to quit)",
                      width=1280, height=720)
    for g in geoms:
        vis.add_geometry(g)
    try:
        while vis.poll_events():
            vis.update_renderer()
    finally:
        vis.destroy_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply",        required=True,  help="Input .ply point cloud.")
    parser.add_argument("--checkpoint", required=True,  help="GraspNet checkpoint (.tar).")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()
    try:
        run(args.ply, args.checkpoint, device=args.device)
    except KeyboardInterrupt:
        print("\nInterrupted.")