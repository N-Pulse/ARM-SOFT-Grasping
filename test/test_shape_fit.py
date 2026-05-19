"""
test/test_shape_fit.py
======================
Live shape-fitting test: classify → fit → show wireframe.

No voting, no EMA smoothing, no temporal tracker.
Every frame is classified and fitted independently via fit_once().

Pipeline per frame
------------------
  1. shape_hint from YOLO  (if classifier is loaded into ObjectIsolator)
  2. _classify_topdown      (bird's-eye projection along table_normal)
  3. fit cylinder / cuboid
  4. Update Open3D wireframe + cv2 bounding-box label

Usage
-----
    python test/test_shape_fit.py
    python test/test_shape_fit.py --debug
    python test/test_shape_fit.py --board-cols 10 --board-rows 7

Controls
--------
  Close the Open3D window or press Ctrl+C to stop.

Chessboard
----------
  11 columns × 8 rows  →  inner corners (10, 7),  square = 15 mm
"""

import queue
import sys
import os
import argparse
import threading

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from capture.object_isolation import ObjectIsolator
from capture.shape_fitter import fit_once
from helper.pcd_visualizer import show_isolated_pcd


# ── Chessboard ─────────────────────────────────────────────────────────────────
_BOARD_COLS = 10    # inner corners  (11 col board → 10)
_BOARD_ROWS = 7     # inner corners  ( 8 row board →  7)
_SQUARE_M   = 0.015 # 15 mm


# ══════════════════════════════════════════════════════════════════════════════
# TABLE PLANE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_table_plane(board_cols=_BOARD_COLS, board_rows=_BOARD_ROWS):
    """
    Stream RealSense frames until chessboard is found; fit plane via SVD.
    Returns (table_normal, d)  or  (None, None) on ESC.
    """
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
            frames      = pipe.wait_for_frames()
            aligned     = align.process(frames)
            cf, df      = aligned.get_color_frame(), aligned.get_depth_frame()
            if not cf or not df:
                continue

            img   = np.asarray(cf.get_data())
            depth = np.asarray(df.get_data()).astype(np.float32) * depth_scale

            gray           = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
# Visualizer callback
# ══════════════════════════════════════════════════════════════════════════════

def _make_overlay_callback(table_normal, isolator):
    """
    Returns an on_new_frame(obj_verts, vis, shape_hint) callback.

    A background worker calls fit_once() each frame (no tracker, no voting).
    The Open3D wireframe is updated on the main/render thread.
    isolator.last_shape is written after each fit so the cv2 preview window
    can display the current class name above the bounding box.
    """
    state = {"ls": None, "label": None}

    _fit_in   = queue.Queue(maxsize=1)
    _fit_out  = {"shape": None, "ls": None}
    _fit_lock = threading.Lock()

    def _worker():
        while True:
            try:
                item = _fit_in.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            verts, shape_hint = item
            shape, ls = fit_once(verts, table_normal, shape_hint=shape_hint)
            if shape is not None and ls is not None:
                with _fit_lock:
                    _fit_out["shape"] = shape
                    _fit_out["ls"]    = ls
                isolator.last_shape = shape   # drives cv2 bounding-box label

    _thread = threading.Thread(target=_worker, daemon=True)
    _thread.start()

    def _on_frame(obj_verts: np.ndarray,
                  vis: o3d.visualization.Visualizer,
                  shape_hint: str | None = None):

        # ── Object lost ───────────────────────────────────────────────────────
        if len(obj_verts) == 0:
            if state["ls"] is not None:
                vis.remove_geometry(state["ls"], reset_bounding_box=False)
                state["ls"]    = None
                state["label"] = None
                vis.update_renderer()
            isolator.last_shape = None
            return

        # ── Push to worker (drop stale frame if worker is still busy) ─────────
        try:
            _fit_in.get_nowait()
        except queue.Empty:
            pass
        _fit_in.put((obj_verts.copy(), shape_hint))

        # ── Pull latest result (non-blocking) ─────────────────────────────────
        with _fit_lock:
            if _fit_out["ls"] is None:
                return
            shape  = _fit_out["shape"]
            new_ls = _fit_out["ls"]
            _fit_out["ls"] = None   # consume

        if shape != state["label"]:
            print(f"[shape_fit]  {shape}")

        # ── Update wireframe (must stay on render thread) ─────────────────────
        if state["ls"] is None:
            vis.add_geometry(new_ls, reset_bounding_box=False)
            state["ls"] = new_ls
            vis.get_view_control().set_lookat(obj_verts.mean(axis=0).tolist())
        else:
            ls = state["ls"]
            ls.points = new_ls.points
            ls.lines  = new_ls.lines
            ls.colors = new_ls.colors
            vis.update_geometry(ls)

        state["label"] = shape

    return _on_frame


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def run(board_cols=_BOARD_COLS, board_rows=_BOARD_ROWS, debug=False):
    table_normal, _ = detect_table_plane(board_cols, board_rows)

    isolator = ObjectIsolator(min_points=50)
    isolator.start()
    print("Waiting for camera …")
    isolator.ready.wait()
    print("Camera ready.\n")

    try:
        show_isolated_pcd(
            isolator,
            on_new_frame=_make_overlay_callback(table_normal, isolator),
            debug=debug,
            camera_up=(0, -1, 0),
        )
    finally:
        isolator.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--board-cols", type=int, default=_BOARD_COLS)
    p.add_argument("--board-rows", type=int, default=_BOARD_ROWS)
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    run(board_cols=a.board_cols, board_rows=a.board_rows, debug=a.debug)
