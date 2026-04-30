"""
DexGraspNet2 inference → cluster & select → execute.
Live D405 viewer on the main thread; press G to run grasp inference.
"""

import sys
import time

sys.path.insert(0, "/home/npulse-cv/DexGraspNet2")

import numpy as np
import open3d as o3d
import torch
from sklearn.cluster import DBSCAN

from src.network.model import get_model
from src.utils.config import ckpt_to_config
from src.utils.dataset import get_sparse_tensor

from stage1 import ObjectIsolator


# ──────────────────────────────────────────────────────────────────────────────
# DexGraspNet2 inference
# ──────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = "cuda"):
    config = ckpt_to_config(checkpoint_path)
    model = get_model(config.model)
    model.config.voxel_size = config.data.voxel_size
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    return model, config


def _prepare_input(pcd, voxel_size, device):
    pts  = np.asarray(pcd.points, dtype=np.float32)
    pc   = torch.from_numpy(pts).unsqueeze(0)
    data = get_sparse_tensor(pc, voxel_size)
    data["seg"] = torch.ones(1, pts.shape[0], dtype=torch.long)
    return {k: v.to(device) for k, v in data.items()}


def infer(model, config, pcd, n_grasps=64, device="cuda"):
    data = _prepare_input(pcd, config.data.voxel_size, device)
    with torch.no_grad():
        rot, trans, joints, scores, _ = model.sample(
            data, n_grasps, graspness_scale=5, allow_fail=True, cate=False,
        )
    return (
        trans[0].cpu().numpy(),
        rot[0].cpu().numpy(),
        joints[0].cpu().numpy(),
        scores[0].cpu().numpy(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cluster + select
# ──────────────────────────────────────────────────────────────────────────────

def cluster_and_select(trans, rot, joints, scores, tip_pos,
                       eps=0.02, min_samples=3, proximity_w=0.3):
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
    return rot[best_idx], trans[best_idx], joints[best_idx]


# ──────────────────────────────────────────────────────────────────────────────
# Execute (placeholder — no robot yet)
# ──────────────────────────────────────────────────────────────────────────────

def execute(rot_cam, trans_cam, joints, T_cam_to_base, robot):
    grasp_cam = np.eye(4, dtype=np.float32)
    grasp_cam[:3, :3] = rot_cam
    grasp_cam[:3,  3] = trans_cam
    grasp_base = T_cam_to_base @ grasp_cam
    robot.approach(grasp_base)
    robot.set_joints(joints)
    robot.lift()


def run_grasp_pipeline(pcd, tip_pos_camera, T_cam_to_base, robot,
                       model, config, device="cuda", n_grasps=64):
    trans, rot, joints, scores = infer(model, config, pcd, n_grasps, device)
    best_rot, best_trans, best_joints = cluster_and_select(
        trans, rot, joints, scores, tip_pos_camera
    )
    if best_rot is None:
        raise RuntimeError("No valid grasp cluster found.")
    execute(best_rot, best_trans, best_joints, T_cam_to_base, robot)


# ──────────────────────────────────────────────────────────────────────────────
# Live viewer + on-demand grasp inference
# ──────────────────────────────────────────────────────────────────────────────

def _run_grasp_inference(latest_pcd, model, config, device, n_grasps, run_execute):
    """Called when the user presses G. Pure side-effect (prints + optional execute)."""
    if latest_pcd is None or len(latest_pcd.points) == 0:
        print("[grasp] no point cloud available yet")
        return

    print(f"[grasp] running inference on {len(latest_pcd.points)} points...")
    t0 = time.time()
    trans, rot, joints, scores = infer(model, config, latest_pcd,
                                       n_grasps=n_grasps, device=device)
    print(f"[grasp] inference done in {time.time()-t0:.2f}s  "
          f"scores: min={scores.min():.2f}  max={scores.max():.2f}  mean={scores.mean():.2f}")

    # Placeholder tip: cloud centroid. Swap for real BNO085 tip (camera frame).
    tip_pos_camera = np.asarray(latest_pcd.points).mean(axis=0).astype(np.float32)
    best_rot, best_trans, best_joints = cluster_and_select(
        trans, rot, joints, scores, tip_pos_camera
    )
    if best_rot is None:
        print("[grasp] no valid cluster")
        return

    print(f"[grasp] best trans  = {best_trans}")
    print(f"[grasp] best joints = {best_joints}")

    if run_execute:
        T_cam_to_base = np.eye(4, dtype=np.float32)
        robot         = None
        execute(best_rot, best_trans, best_joints, T_cam_to_base, robot)
        print("[grasp] executed")


def live_loop(model, config, device="cuda", n_grasps=64, run_execute=False):
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
        _run_grasp_inference(latest["pcd"], model, config,
                             device, n_grasps, run_execute)
        return False  # don't force redraw

    def on_quit(_vis):
        _vis.close()
        return False

    vis.register_key_callback(ord("G"), on_grasp)
    vis.register_key_callback(ord("Q"), on_quit)

    