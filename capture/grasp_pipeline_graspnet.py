"""
GraspNet-baseline inference → cluster & select → execute.
Live D405 viewer on the main thread; press G to run grasp inference.

Drop-in replacement for grasp_pipeline.py using graspnet-baseline instead of
DexGraspNet2. Outputs 6-DoF parallel-jaw grasps (no finger joints).
"""

import os
import sys
import time

GRASPNET_ROOT = "/home/npulse-cv/Desktop/npulse-cv/graspnet-baseline"

sys.path.insert(0, GRASPNET_ROOT)
sys.path.insert(0, os.path.join(GRASPNET_ROOT, "models"))
sys.path.insert(0, os.path.join(GRASPNET_ROOT, "dataset"))
sys.path.insert(0, os.path.join(GRASPNET_ROOT, "utils"))
sys.path.insert(0, os.path.join(GRASPNET_ROOT, "pointnet2"))

import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import torch
from sklearn.cluster import DBSCAN

from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector

from object_isolation import ObjectIsolator


NUM_POINT  = 20000   # points sampled from the cloud before feeding the model
MIN_DEPTH  = 0.07    # metres — D405 close-range minimum
MAX_DEPTH  = 0.70


# ──────────────────────────────────────────────────────────────────────────────
# RealSense point cloud capture (no YOLO — full scene)
# ──────────────────────────────────────────────────────────────────────────────

def capture_snapshot(n_warmup: int = 30) -> o3d.geometry.PointCloud:
    """
    Capture one depth+colour frame from the D405 and return a full-scene PCD.
    Runs n_warmup frames first so auto-exposure and temporal filters stabilise.
    """
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile      = pipeline.start(cfg)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 4)

    align    = rs.align(rs.stream.color)
    spatial  = rs.spatial_filter()
    temporal = rs.temporal_filter()
    holes    = rs.hole_filling_filter()
    spatial.set_option(rs.option.filter_smooth_alpha,  0.5)
    spatial.set_option(rs.option.filter_smooth_delta,  20)
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)
    pc_util  = rs.pointcloud()

    try:
        for _ in range(n_warmup):
            pipeline.wait_for_frames()

        frames     = pipeline.wait_for_frames()
        aligned    = align.process(frames)
        depth_fr   = aligned.get_depth_frame()
        color_fr   = aligned.get_color_frame()

        depth_fr = spatial.process(depth_fr)
        depth_fr = temporal.process(depth_fr)
        depth_fr = holes.process(depth_fr)

        pc_util.map_to(color_fr)
        points_rs = pc_util.calculate(depth_fr)

        verts     = np.asanyarray(points_rs.get_vertices()).view(np.float32).reshape(-1, 3)
        texcoords = np.asanyarray(points_rs.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

        depth_vals = np.linalg.norm(verts, axis=1)
        valid      = (depth_vals > MIN_DEPTH) & (depth_vals < MAX_DEPTH)
        verts      = verts[valid]
        texcoords  = texcoords[valid]

        bgr = np.asanyarray(color_fr.get_data())
        h, w = bgr.shape[:2]
        u = np.clip((texcoords[:, 0] * w).astype(int), 0, w - 1)
        v = np.clip((texcoords[:, 1] * h).astype(int), 0, h - 1)
        colors = bgr[v, u, ::-1] / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    finally:
        pipeline.stop()


# ──────────────────────────────────────────────────────────────────────────────
# GraspNet-baseline inference
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = "cuda"):
    net = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    net.load_state_dict(ckpt["model_state_dict"])
    net.to(device).eval()
    return net


def _prepare_input(pcd, device):
    pts    = np.asarray(pcd.points,  dtype=np.float32)
    colors = np.asarray(pcd.colors,  dtype=np.float32)

    n = len(pts)
    if n >= NUM_POINT:
        idxs = np.random.choice(n, NUM_POINT, replace=False)
    else:
        idxs = np.concatenate([
            np.arange(n),
            np.random.choice(n, NUM_POINT - n, replace=True),
        ])

    pts_s    = pts[idxs]
    colors_s = colors[idxs] if len(colors) == n else np.zeros((NUM_POINT, 3), dtype=np.float32)

    end_points = {
        "point_clouds":  torch.from_numpy(pts_s[np.newaxis]).to(device),
        "cloud_colors":  torch.from_numpy(colors_s[np.newaxis]).to(device),
    }
    return end_points


def infer(model, pcd, device="cuda", collision_thresh=0.01):
    end_points = _prepare_input(pcd, device)
    with torch.no_grad():
        end_points = model(end_points)
        grasp_preds = pred_decode(end_points)

    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

    # collision filtering against the full cloud
    cloud_pts = np.asarray(pcd.points, dtype=np.float32)
    detector  = ModelFreeCollisionDetector(cloud_pts, voxel_size=0.01)
    collision_mask = detector.detect(gg, approach_dist=0.05,
                                     collision_thresh=collision_thresh)
    gg = gg[~collision_mask]

    if len(gg) == 0:
        return (np.zeros((0, 3)), np.zeros((0, 3, 3)),
                np.zeros((0,)),   np.zeros((0,)))

    return (
        gg.translations,
        gg.rotation_matrices,
        gg.scores,
        gg.widths,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cluster + select
# ──────────────────────────────────────────────────────────────────────────────

def cluster_and_select(trans, rot, scores, widths, tip_pos,
                       eps=0.02, min_samples=3, proximity_w=0.3):
    if len(trans) == 0:
        return None, None, None

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(trans)
    best_score, best_idx = -np.inf, None

    for label in set(labels):
        if label == -1:
            continue
        mask            = labels == label
        cluster_scores  = scores[mask]
        cluster_trans   = trans[mask]
        cluster_indices = np.where(mask)[0]

        best_graspness  = cluster_scores.max()
        centroid        = cluster_trans.mean(axis=0)
        proximity_bonus = 1.0 / (1.0 + np.linalg.norm(centroid - tip_pos))
        combined        = (1 - proximity_w) * best_graspness + proximity_w * proximity_bonus

        if combined > best_score:
            best_score = combined
            best_idx   = cluster_indices[cluster_scores.argmax()]

    if best_idx is None:
        return None, None, None
    return rot[best_idx], trans[best_idx], widths[best_idx]


# ──────────────────────────────────────────────────────────────────────────────
# Execute (placeholder — no robot yet)
# ──────────────────────────────────────────────────────────────────────────────

def execute(rot_cam, trans_cam, width, T_cam_to_base, robot):
    grasp_cam = np.eye(4, dtype=np.float32)
    grasp_cam[:3, :3] = rot_cam
    grasp_cam[:3,  3] = trans_cam
    grasp_base = T_cam_to_base @ grasp_cam
    robot.approach(grasp_base)
    robot.set_width(width)
    robot.lift()


def run_grasp_pipeline(pcd, tip_pos_camera, T_cam_to_base, robot,
                       model, device="cuda"):
    trans, rot, scores, widths = infer(model, pcd, device)
    best_rot, best_trans, best_width = cluster_and_select(
        trans, rot, scores, widths, tip_pos_camera
    )
    if best_rot is None:
        raise RuntimeError("No valid grasp cluster found.")
    execute(best_rot, best_trans, best_width, T_cam_to_base, robot)


# ──────────────────────────────────────────────────────────────────────────────
# Live viewer + on-demand grasp inference
# ──────────────────────────────────────────────────────────────────────────────

def _run_grasp_inference(latest_pcd, model, device, run_execute):
    """Called when the user presses G. Pure side-effect (prints + optional execute)."""
    if latest_pcd is None or len(latest_pcd.points) == 0:
        print("[grasp] no point cloud available yet")
        return

    print(f"[grasp] running inference on {len(latest_pcd.points)} points...")
    t0 = time.time()
    trans, rot, scores, widths = infer(model, latest_pcd, device)
    print(f"[grasp] inference done in {time.time()-t0:.2f}s  "
          f"grasps after collision filter: {len(trans)}  "
          f"scores: min={scores.min():.3f}  max={scores.max():.3f}  mean={scores.mean():.3f}"
          if len(trans) > 0 else
          f"[grasp] inference done in {time.time()-t0:.2f}s  no grasps survived collision filter")

    if len(trans) == 0:
        print("[grasp] no grasps to select from")
        return

    # Placeholder tip: cloud centroid. Swap for real BNO085 tip (camera frame).
    tip_pos_camera = np.asarray(latest_pcd.points).mean(axis=0).astype(np.float32)
    best_rot, best_trans, best_width = cluster_and_select(
        trans, rot, scores, widths, tip_pos_camera
    )
    if best_rot is None:
        print("[grasp] no valid cluster")
        return

    print(f"[grasp] best trans = {best_trans}")
    print(f"[grasp] best width = {best_width:.4f} m")

    if run_execute:
        T_cam_to_base = np.eye(4, dtype=np.float32)
        robot         = None
        execute(best_rot, best_trans, best_width, T_cam_to_base, robot)
        print("[grasp] executed")


def live_loop(model, device="cuda", run_execute=False):
    isolator = ObjectIsolator()
    isolator.start()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("N-Pulse — isolated object (press G to grasp, Q to quit)",
                      width=1280, height=720)

    pcd        = o3d.geometry.PointCloud()
    geom_added = False
    latest     = {"pcd": None}   # mutable box so the key callback can read it

    # ── key callbacks ─────────────────────────────────────────────────────────
    def on_grasp(_vis):
        _run_grasp_inference(latest["pcd"], model, device, run_execute)
        return False

    def on_quit(_vis):
        _vis.close()
        return False

    vis.register_key_callback(ord("G"), on_grasp)
    vis.register_key_callback(ord("Q"), on_quit)

    try:
        while True:
            new_pcd = isolator.get_pcd()
            if new_pcd is not None:
                latest["pcd"] = new_pcd
                pcd.points = new_pcd.points
                pcd.colors = new_pcd.colors
                if not geom_added:
                    vis.add_geometry(pcd)
                    geom_added = True
                else:
                    vis.update_geometry(pcd)

            if not vis.poll_events():
                break
            vis.update_renderer()
    finally:
        isolator.stop()
        vis.destroy_window()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to GraspNet-baseline checkpoint (.tar)")
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--execute",  action="store_true",
                        help="Actually send commands to the robot")
    parser.add_argument("--snapshot", action="store_true",
                        help="Capture one frame from D405, run inference, then exit")
    parser.add_argument("--warmup",   type=int, default=30,
                        help="Warm-up frames for snapshot mode (default 30)")
    args = parser.parse_args()

    model = load_model(args.checkpoint, device=args.device)

    if args.snapshot:
        print("[capture] capturing snapshot from D405...")
        pcd = capture_snapshot(n_warmup=args.warmup)
        print(f"[capture] {len(pcd.points)} points captured")
        _run_grasp_inference(pcd, model, args.device, args.execute)
    else:
        live_loop(model, device=args.device, run_execute=args.execute)
