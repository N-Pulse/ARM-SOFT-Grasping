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

import queue
import threading

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import torch
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

from graspnetAPI import GraspGroup
from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector


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
    MIN_DEPTH  = 0.07
    MAX_DEPTH  = 0.70
    YOLO_MODEL = "yolo11n-seg.pt"
    YOLO_CONF  = 0.35
    YOLO_IOU   = 0.45
    IMG_CENTER = np.array([320.0, 240.0])

    frame_queue = queue.Queue(maxsize=1)
    stop_event  = threading.Event()

    # ── background capture + YOLO thread ──────────────────────────────────────
    def capture_loop():
        # RealSense setup (mirrors pointcloud_open3d.py)
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
        pc_util = rs.pointcloud()

        yolo = YOLO(YOLO_MODEL)
        yolo.fuse()

        try:
            while not stop_event.is_set():
                frames   = pipeline.wait_for_frames()
                aligned  = align.process(frames)
                depth_fr = aligned.get_depth_frame()
                color_fr = aligned.get_color_frame()
                if not depth_fr or not color_fr:
                    continue

                depth_fr = spatial.process(depth_fr)
                depth_fr = temporal.process(depth_fr)
                depth_fr = holes.process(depth_fr)

                bgr = np.asanyarray(color_fr.get_data())

                # ── YOLO segmentation ──────────────────────────────────────
                results    = yolo.predict(source=bgr, conf=YOLO_CONF,
                                          iou=YOLO_IOU, verbose=False)
                detections = []
                for r in results:
                    if r.masks is None:
                        continue
                    for mask_t, box in zip(r.masks.data, r.boxes.xyxy):
                        m = cv2.resize(mask_t.cpu().numpy(),
                                       (bgr.shape[1], bgr.shape[0]),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                        detections.append((m, box.cpu().numpy().astype(int)))

                # pick the object whose bbox centre is closest to image centre
                target_mask, target_box = None, None
                best_dist = float("inf")
                for mask, box in detections:
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    d  = np.linalg.norm(np.array([cx, cy]) - IMG_CENTER)
                    if d < best_dist:
                        best_dist, target_mask, target_box = d, mask, box

                # ── annotate 2D preview ────────────────────────────────────
                preview = bgr.copy()
                if target_mask is not None:
                    overlay = np.zeros_like(preview)
                    overlay[target_mask] = (0, 255, 0)
                    preview = cv2.addWeighted(preview, 0.7, overlay, 0.3, 0)
                    x1, y1, x2, y2 = target_box
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(preview, "target", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # ── build full-scene PCD then mask to object ───────────────
                pc_util.map_to(color_fr)
                pts_rs    = pc_util.calculate(depth_fr)
                verts     = np.asanyarray(pts_rs.get_vertices()).view(np.float32).reshape(-1, 3)
                texcoords = np.asanyarray(pts_rs.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

                depth_vals = np.linalg.norm(verts, axis=1)
                valid      = (depth_vals > MIN_DEPTH) & (depth_vals < MAX_DEPTH)
                verts      = verts[valid]
                texcoords  = texcoords[valid]

                h, w   = bgr.shape[:2]
                u      = np.clip((texcoords[:, 0] * w).astype(int), 0, w - 1)
                v      = np.clip((texcoords[:, 1] * h).astype(int), 0, h - 1)
                colors = bgr[v, u, ::-1] / 255.0

                if target_mask is not None:
                    u_f = np.clip((texcoords[:, 0] * w).astype(int), 0, w - 1)
                    v_f = np.clip((texcoords[:, 1] * h).astype(int), 0, h - 1)
                    inside     = target_mask[v_f, u_f]
                    obj_verts  = verts[inside]
                    obj_colors = colors[inside]
                else:
                    obj_verts, obj_colors = verts, colors

                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put((obj_verts, obj_colors, preview))
        finally:
            pipeline.stop()

    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    # ── Open3D visualiser (main thread) ───────────────────────────────────────
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("N-Pulse — isolated object (press G to grasp, Q to quit)",
                      width=1280, height=720)

    pcd         = o3d.geometry.PointCloud()
    geom_added  = False
    latest      = {"pcd": None}
    grasp_geoms = []

    def on_grasp(_vis):
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

    vis.register_key_callback(ord("G"), on_grasp)
    vis.register_key_callback(ord("Q"), on_quit)

    try:
        while True:
            try:
                obj_verts, obj_colors, preview = frame_queue.get_nowait()

                cv2.imshow("Camera feed (YOLO — target in green)", preview)
                cv2.waitKey(1)

                if len(obj_verts) > 0:
                    new_pcd = o3d.geometry.PointCloud()
                    new_pcd.points = o3d.utility.Vector3dVector(obj_verts)
                    new_pcd.colors = o3d.utility.Vector3dVector(obj_colors)
                    latest["pcd"] = new_pcd
                    pcd.points    = new_pcd.points
                    pcd.colors    = new_pcd.colors
                    if not geom_added:
                        vis.add_geometry(pcd)
                        geom_added = True
                    else:
                        vis.update_geometry(pcd)
            except queue.Empty:
                pass

            if not vis.poll_events():
                break
            vis.update_renderer()
    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        vis.destroy_window()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to GraspNet-baseline checkpoint (.tar)")
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--execute", action="store_true",
                        help="Actually send commands to the robot")
    args = parser.parse_args()

    model = load_model(args.checkpoint, device=args.device)
    live_loop(model, device=args.device, run_execute=args.execute)
