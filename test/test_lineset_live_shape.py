"""
test/test_lineset_live_shape.py
===============================
Live GraspNet grasp estimation based on the FITTED SHAPE instead of the raw
isolated point cloud.

Pipeline
--------
  1. Detect table plane via chessboard (same as test_shape_fit.py)
  2. ObjectIsolator  →  isolated object point cloud per frame
  3. ShapeFitThread  →  fits cylinder / cuboid (fit_and_track), samples a clean
                        point cloud from the fitted geometry mesh, feeds it to
                        GraspInferenceThread
  4. GraspInferenceThread  →  GraspNet on the shape-sampled cloud
  5. Open3D window shows: raw point cloud + shape wireframe + grasp lineset

Usage
-----
    python test/test_lineset_live_shape.py --checkpoint /path/to/checkpoint.tar
    python test/test_lineset_live_shape.py --checkpoint /path/to/checkpoint.tar --device cpu

Controls
--------
    Close Open3D window  |  Ctrl+C  |  ESC / q in the YOLO preview

Chessboard
----------
  11 columns × 8 rows  →  inner corners (10, 7),  square = 15 mm
"""

import sys
import os
import queue
import signal
import threading
import time
import argparse

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "capture"))
sys.path.insert(0, _ROOT)

import cv2
import numpy as np
import open3d as o3d
import torch

from capture.object_isolation import ObjectIsolator
from capture.shape_fitter import ShapeTracker, fit_and_track
from grasp_pipeline_graspnet import load_model, GraspInferenceThread
from helper.pcd_visualizer import auto_zoom


# ── Chessboard defaults ────────────────────────────────────────────────────────
_BOARD_COLS = 10
_BOARD_ROWS = 7
_SQUARE_M   = 0.015

# ── Sampling ───────────────────────────────────────────────────────────────────
_SHAPE_SAMPLE_N = 2000   # surface points sampled from fitted geometry for GraspNet


# ══════════════════════════════════════════════════════════════════════════════
# TABLE PLANE DETECTION  (same logic as test_shape_fit.py)
# ══════════════════════════════════════════════════════════════════════════════

def detect_table_plane(board_cols=_BOARD_COLS, board_rows=_BOARD_ROWS):
    """
    Stream RealSense frames until a chessboard is found; fit plane via SVD.
    Returns (table_normal, d)  or  (None, None) on ESC.
    """
    import pyrealsense2 as rs

    board_shape = (board_cols, board_rows)
    criteria    = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    pipe  = rs.pipeline()
    cfg   = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    align = rs.align(rs.stream.color)

    profile     = pipe.start(cfg)
    intr        = (profile.get_stream(rs.stream.color)
                          .as_video_stream_profile().get_intrinsics())
    depth_scale = (profile.get_device().first_depth_sensor().get_depth_scale())
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

    print(f"\n[table]  Board inner corners {board_cols}×{board_rows}, "
          f"square {_SQUARE_M*1000:.0f} mm")
    print("[table]  Show chessboard to camera.  ESC to skip.\n")

    try:
        while True:
            frames  = pipe.wait_for_frames()
            aligned = align.process(frames)
            cf, df  = aligned.get_color_frame(), aligned.get_depth_frame()
            if not cf or not df:
                continue

            img   = np.asarray(cf.get_data())
            depth = np.asarray(df.get_data()).astype(np.float32) * depth_scale

            gray         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, board_shape, None)

            disp = img.copy()
            if found:
                cv2.drawChessboardCorners(disp, board_shape, corners, True)
            cv2.putText(disp,
                        "FOUND — hold still" if found else "Searching …",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 220, 0) if found else (0, 80, 255), 2)
            cv2.imshow("Table plane detection  [ESC to skip]", disp)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                return None, None
            if not found:
                continue

            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            pts3 = []
            for (u, v) in corners.reshape(-1, 2):
                ui, vi = int(round(u)), int(round(v))
                if not (0 <= ui < depth.shape[1] and 0 <= vi < depth.shape[0]):
                    continue
                z = float(depth[vi, ui])
                if z < 0.05 or z > 3.0:
                    continue
                pts3.append([(u - cx)*z/fx, (v - cy)*z/fy, z])

            if len(pts3) < 6:
                continue

            pts3     = np.array(pts3)
            centroid = pts3.mean(axis=0)
            _, _, Vt = np.linalg.svd(pts3 - centroid)
            normal   = Vt[-1] / np.linalg.norm(Vt[-1])
            d        = -float(normal @ centroid)
            if float(normal @ centroid) < 0:
                normal, d = -normal, -d

            res = float(np.abs((pts3 - centroid) @ normal).mean())
            if res > 0.005:
                print(f"[table]  Residual {res*1000:.1f} mm — retry")
                continue

            print(f"[table]  ✓  normal={np.round(normal,3)}  "
                  f"residual={res*1000:.2f} mm\n")
            cv2.waitKey(400)
            cv2.destroyAllWindows()
            return normal, d
    finally:
        pipe.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SHAPE → POINT CLOUD SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def _sample_cylinder_pcd(tracker, n=_SHAPE_SAMPLE_N):
    """
    Build a cylinder mesh from the tracker's smoothed parameters and sample n
    surface points from it.  Returns an o3d.geometry.PointCloud or None.
    """
    axis    = tracker.axis
    axis_pt = tracker.axis_pt
    r       = tracker.radius
    if axis is None or axis_pt is None or r is None:
        return None

    h_min  = tracker.h_ctr - tracker.height / 2.0
    h_max  = tracker.h_ctr + tracker.height / 2.0
    height = float(np.clip(h_max - h_min, 0.005, 5.0))
    r      = float(np.clip(r, 0.005, 2.0))

    # Rotation matrix: align Z-axis to cylinder axis
    z  = np.array([0., 0., 1.])
    v  = np.cross(z, axis);  s = np.linalg.norm(v);  c = float(np.dot(z, axis))
    if s < 1e-6:
        R = np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R  = np.eye(3) + vx + vx @ vx * (1.0 - c) / (s**2)

    center = axis_pt + axis * (h_min + h_max) / 2.0
    mesh   = o3d.geometry.TriangleMesh.create_cylinder(
        radius=r, height=height, resolution=20
    )
    mesh.rotate(R, center=np.zeros(3))
    mesh.translate(center)
    return mesh.sample_points_uniformly(n)


def _sample_cuboid_pcd(shape_ls, n=_SHAPE_SAMPLE_N):
    """
    Reconstruct a closed cuboid mesh from the 8 vertices in the LineSet returned
    by shape_fitter._build_cuboid(), then sample n surface points from it.

    Vertex layout (from shape_fitter._build_cuboid):
        [bot_0, top_0, bot_1, top_1, bot_2, top_2, bot_3, top_3]
    Bottom corners: indices 0, 2, 4, 6   (CCW order)
    Top corners:    indices 1, 3, 5, 7
    """
    verts = np.asarray(shape_ls.points)
    if len(verts) < 8:
        return None

    # 12 triangles — 2 per face × 6 faces
    triangles = [
        # bottom face
        [0, 4, 2], [0, 6, 4],
        # top face
        [1, 3, 5], [1, 5, 7],
        # side 0→1  (pillars at corners 0 and 1)
        [0, 1, 3], [0, 3, 2],
        # side 1→2
        [2, 3, 5], [2, 5, 4],
        # side 2→3
        [4, 5, 7], [4, 7, 6],
        # side 3→0
        [6, 7, 1], [6, 1, 0],
    ]
    mesh           = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh.sample_points_uniformly(n)


# ══════════════════════════════════════════════════════════════════════════════
# GRIPPER LINESET
# ══════════════════════════════════════════════════════════════════════════════

FINGER_LENGTH = 0.085
PALM_DEPTH    = 0.06
GRIPPER_COLOR = [1.0, 0.4, 0.0]


def _gripper_lineset(rot, trans, width):
    approach   = rot[:, 0]
    closing    = rot[:, 1]
    half_w     = width / 2.0

    tip_center = trans + approach * FINGER_LENGTH
    palm_back  = trans - approach * PALM_DEPTH
    left_root  = trans      + closing * half_w
    right_root = trans      - closing * half_w
    left_tip   = tip_center + closing * half_w
    right_tip  = tip_center - closing * half_w

    pts   = np.array([palm_back, trans, left_root, right_root,
                      left_tip, right_tip], dtype=np.float64)
    lines = [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5]]

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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(checkpoint, device="cuda", board_cols=_BOARD_COLS, board_rows=_BOARD_ROWS):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — pass --device cpu")

    # ── Table plane ────────────────────────────────────────────────────────────
    table_normal, _ = detect_table_plane(board_cols, board_rows)

    # ── GraspNet model ─────────────────────────────────────────────────────────
    print("Loading GraspNet model...")
    model = load_model(checkpoint, device=device)

    # ── ObjectIsolator ─────────────────────────────────────────────────────────
    isolator = ObjectIsolator(min_points=50)
    isolator.start()
    print("Waiting for YOLO to load...")
    isolator.ready.wait()
    print("YOLO ready — opening viewer.\n")
    print("Running — close the Open3D window or Ctrl+C to stop.\n")

    grasp_thread = GraspInferenceThread(model, device=device, interval=0.5)

    # ── Shape-fit background thread ────────────────────────────────────────────
    tracker   = ShapeTracker()
    _fit_in   = queue.Queue(maxsize=1)
    _fit_out  = {"shape": None, "shape_ls": None}
    _fit_lock = threading.Lock()

    def _fit_worker():
        while True:
            try:
                verts = _fit_in.get(timeout=0.5)
            except queue.Empty:
                continue
            if verts is None:       # sentinel — shut down
                break

            shape, shape_ls = fit_and_track(verts, table_normal, tracker)
            if shape is None or shape_ls is None:
                continue

            # Sample a clean point cloud from the fitted shape mesh
            if shape == "cylinder":
                shape_pcd = _sample_cylinder_pcd(tracker)
            else:
                shape_pcd = _sample_cuboid_pcd(shape_ls)

            # Feed shape-sampled cloud to GraspNet (instead of raw obj_verts)
            if shape_pcd is not None and len(shape_pcd.points) >= 50:
                grasp_thread.update_pcd(shape_pcd)

            with _fit_lock:
                _fit_out["shape"]    = shape
                _fit_out["shape_ls"] = shape_ls

    fit_thread = threading.Thread(target=_fit_worker, daemon=True)
    fit_thread.start()

    # ── Clean-exit flag ────────────────────────────────────────────────────────
    _stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: _stop.set())

    # ── Open3D window ──────────────────────────────────────────────────────────
    CV2_WIN = "YOLO Detection"
    cv2.namedWindow(CV2_WIN, cv2.WINDOW_NORMAL)

    vis = o3d.visualization.Visualizer()
    vis.create_window("GraspNet — Fitted-Shape Grasp", width=1280, height=720)
    opt = vis.get_render_option()
    opt.point_size       = 2.0
    opt.line_width       = 3.0
    opt.background_color = np.array([1.0, 1.0, 1.0])

    pcd         = o3d.geometry.PointCloud()
    grasp_ls    = _empty_lineset()
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05, origin=[0.0, 0.0, 0.0]
    )

    geom_added     = False
    zoom_fitted    = False
    last_grasp     = None
    grasp_ls_ready = False
    shape_ls_geom  = None     # current shape wireframe geometry in scene
    last_shape     = None     # track shape type to detect topology changes

    RENDER_INTERVAL = 1.0 / 30.0
    _last_render    = 0.0
    frame_ready     = False

    while not _stop.is_set():
        # ── Pull latest frame ────────────────────────────────────────────────
        frame_ready = False
        try:
            verts, _raw_cols, full_colors, obj_verts, obj_colors, preview_bgr = \
                isolator._frame_queue.get_nowait()
            frame_ready = True

            pts  = obj_verts  if len(obj_verts)  > 0 else verts
            cols = obj_colors if len(obj_colors) > 0 else full_colors

            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)

            if not geom_added:
                vis.add_geometry(pcd)
                vis.add_geometry(grasp_ls)
                vis.add_geometry(coord_frame)
                geom_added = True
            else:
                vis.update_geometry(pcd)

            if not zoom_fitted and len(obj_verts) > 0:
                iso_pcd = o3d.geometry.PointCloud()
                iso_pcd.points = o3d.utility.Vector3dVector(obj_verts)
                auto_zoom(vis, iso_pcd)
                zoom_fitted = True

            if preview_bgr is not None:
                cv2.imshow(CV2_WIN, preview_bgr)

            # Feed raw obj verts to shape fitter (drop stale frame if busy)
            if len(obj_verts) > 0:
                try:
                    _fit_in.get_nowait()
                except queue.Empty:
                    pass
                _fit_in.put(obj_verts.copy())

        except queue.Empty:
            pass

        # ── Pull latest shape wireframe ──────────────────────────────────────
        with _fit_lock:
            new_shape    = _fit_out["shape"]
            new_shape_ls = _fit_out["shape_ls"]
            _fit_out["shape_ls"] = None   # consume

        if new_shape_ls is not None and geom_added:
            if new_shape != last_shape:
                # Topology changed (cylinder ↔ cuboid) — must remove + re-add
                print(f"[shape]  *** {new_shape} ***")
                if shape_ls_geom is not None:
                    vis.remove_geometry(shape_ls_geom, reset_bounding_box=False)
                vis.add_geometry(new_shape_ls, reset_bounding_box=False)
                shape_ls_geom = new_shape_ls
                last_shape    = new_shape
            else:
                # Same topology — update in place
                shape_ls_geom.points = new_shape_ls.points
                shape_ls_geom.lines  = new_shape_ls.lines
                shape_ls_geom.colors = new_shape_ls.colors
                vis.update_geometry(shape_ls_geom)

        # ── Pull latest grasp result ─────────────────────────────────────────
        grasp = grasp_thread.get_grasp()
        if grasp is not None and grasp is not last_grasp:
            last_grasp = grasp
            rot, trans, width = grasp
            new_ls = _gripper_lineset(rot, trans, width)
            if not grasp_ls_ready and geom_added:
                # Replace placeholder lineset (different topology)
                vis.remove_geometry(grasp_ls, reset_bounding_box=False)
                grasp_ls.points = new_ls.points
                grasp_ls.lines  = new_ls.lines
                grasp_ls.colors = new_ls.colors
                vis.add_geometry(grasp_ls, reset_bounding_box=False)
                grasp_ls_ready = True
            elif grasp_ls_ready:
                grasp_ls.points = new_ls.points
                grasp_ls.lines  = new_ls.lines
                grasp_ls.colors = new_ls.colors
                vis.update_geometry(grasp_ls)

        # ── Render (capped at 30 fps) ────────────────────────────────────────
        now = time.monotonic()
        if now - _last_render >= RENDER_INTERVAL:
            if not vis.poll_events():
                _stop.set()
                break
            vis.update_renderer()
            _last_render = now

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            _stop.set()
            break

        if not frame_ready:
            time.sleep(0.005)

    # ── Teardown ──────────────────────────────────────────────────────────────
    print("\n[test_lineset_live_shape] shutting down...")
    _fit_in.put(None)   # sentinel to stop fit thread
    grasp_thread.stop()
    isolator.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()
    print("[test_lineset_live_shape] done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="GraspNet checkpoint (.tar)")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--board-cols", type=int, default=_BOARD_COLS)
    parser.add_argument("--board-rows", type=int, default=_BOARD_ROWS)
    args = parser.parse_args()
    run(args.checkpoint, device=args.device,
        board_cols=args.board_cols, board_rows=args.board_rows)
