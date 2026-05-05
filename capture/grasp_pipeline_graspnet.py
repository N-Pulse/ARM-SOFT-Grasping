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

import cv2
import numpy as np
import open3d as o3d
import torch
from sklearn.cluster import DBSCAN

from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from object_isolation import ObjectIsolator
from pointcloud_open3d import PointcloudViewer


NUM_POINT = 20000   # points sampled from the cloud before feeding the model


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

def _grasp_geometry(rot, trans, width):
    """Coordinate frame + gripper jaw lines for one grasp pose."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
    frame.rotate(rot, center=(0, 0, 0))
    frame.translate(trans)

    # GraspNet convention: approach = col 0 (X), closing = col 1 (Y)
    half_w       = width / 2.0
    left         = trans + rot @ np.array([ half_w, 0, 0], dtype=np.float32)
    right        = trans + rot @ np.array([-half_w, 0, 0], dtype=np.float32)
    approach_tip = trans + rot @ np.array([0, 0, 0.06],    dtype=np.float32)

    jaws = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([left, right, trans, approach_tip]),
        lines=o3d.utility.Vector2iVector([[0, 1], [2, 3]]),
    )
    jaws.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0]])
    return [frame, jaws]


def _run_grasp_inference(latest_pcd, model, device, run_execute):
    """Called when the user presses G. Returns list of o3d geometries, or []."""
    if latest_pcd is None or len(latest_pcd.points) == 0:
        print("[grasp] no point cloud available yet")
        return []

    print(f"[grasp] running inference on {len(latest_pcd.points)} points...")
    t0 = time.time()
    trans, rot, scores, widths = infer(model, latest_pcd, device)
    elapsed = time.time() - t0
    if len(trans) > 0:
        print(f"[grasp] {elapsed:.2f}s  grasps: {len(trans)}  "
              f"scores: min={scores.min():.3f}  max={scores.max():.3f}  mean={scores.mean():.3f}")
    else:
        print(f"[grasp] {elapsed:.2f}s  no grasps survived collision filter")
        return []

    # Placeholder tip: cloud centroid. Swap for real BNO085 tip (camera frame).
    tip_pos_camera = np.asarray(latest_pcd.points).mean(axis=0).astype(np.float32)
    best_rot, best_trans, best_width = cluster_and_select(
        trans, rot, scores, widths, tip_pos_camera
    )
    if best_rot is None:
        print("[grasp] no valid cluster")
        return []

    print(f"[grasp] best trans = {best_trans}")
    print(f"[grasp] best width = {best_width:.4f} m")

    if run_execute:
        T_cam_to_base = np.eye(4, dtype=np.float32)
        robot         = None
        execute(best_rot, best_trans, best_width, T_cam_to_base, robot)
        print("[grasp] executed")

    return _grasp_geometry(best_rot, best_trans, best_width)


def live_loop(model, device="cuda", run_execute=False):
    # ObjectIsolator handles camera + YOLO on its own background thread.
    # It applies the same central-object selection criteria as the isolation
    # algorithm, so pressing G always grasps the same object shown in the preview.
    # create pointcloud object here?
    isolator = ObjectIsolator()
    isolator.start()
    print("[capture] background thread started — waiting for first frame...")

    # ── cv2 preview window — show immediately so user knows it's running ──────
    CV2_WIN = "Camera feed (YOLO — target in green)"
    cv2.namedWindow(CV2_WIN, cv2.WINDOW_NORMAL)
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for camera...", (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    cv2.imshow(CV2_WIN, placeholder)
    cv2.waitKey(1)

    # ── Open3D viewer (main thread) ───────────────────────────────────────────
    viewer      = PointcloudViewer("N-Pulse — live pointcloud (press G to grasp, Q to quit)")
    latest      = {"pcd": None}   # latest isolated object, updated every frame
    grasp_geoms = []

    def on_grasp(_vis):
        if model is None:
            print("[grasp] no checkpoint loaded — pass --checkpoint to enable inference")
            return False
        for g in grasp_geoms:
            _vis.remove_geometry(g, reset_bounding_box=False)
        grasp_geoms.clear()
        new_geoms = _run_grasp_inference(latest["pcd"], model, device, run_execute)
        for g in new_geoms:
            _vis.add_geometry(g, reset_bounding_box=False)
        grasp_geoms.extend(new_geoms)
        return False

    def on_quit(_vis):
        _vis.close()
        return False

    viewer.register_key(ord("G"), on_grasp)
    viewer.register_key(ord("Q"), on_quit)

    try:
        while True:
            frame = isolator.get_full_frame()
            if frame is not None:
                full_pcd, iso_pcd, preview_bgr = frame

                cv2.imshow(CV2_WIN, preview_bgr)
                cv2.waitKey(1)

                viewer.update(full_pcd)

                if iso_pcd is not None:
                    latest["pcd"] = iso_pcd

            if not viewer.tick():
                break
    finally:
        isolator.stop()
        cv2.destroyAllWindows()
        viewer.destroy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Path to GraspNet-baseline checkpoint (.tar). "
                             "Omit to run live preview only (G key disabled).")
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--execute", action="store_true",
                        help="Actually send commands to the robot")
    args = parser.parse_args()

    if args.checkpoint:
        model = load_model(args.checkpoint, device=args.device)
    else:
        print("[warn] no checkpoint provided — live preview only, G key disabled")
        model = None
    live_loop(model, device=args.device, run_execute=args.execute)
