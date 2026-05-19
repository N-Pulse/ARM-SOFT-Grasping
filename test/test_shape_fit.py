"""
test/test_shape_fit.py
======================
Two-thread architecture:

  ComputeWorker (background)
    · Drains ObjectIsolator's frame queue
    · Receives shape_hint from YOLO (set inside ObjectIsolator)
    · Calls fit_once — shape is determined entirely by shape_hint
    · Stores result in a lock-protected single slot

  Render loop (main thread)
    · Reads the latest pre-computed result (non-blocking)
    · Updates Open3D point cloud + wireframe geometries
    · Shows cv2 camera preview
    · Zero computation on the render thread

Usage
-----
    python test/test_shape_fit.py
    python test/test_shape_fit.py --debug
    python test/test_shape_fit.py --board-cols 10 --board-rows 7

Controls
--------
  Close the Open3D window | Ctrl+C | ESC / q in the camera preview

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
from capture.shape_classifier import ShapeClassifier


# ── Chessboard ─────────────────────────────────────────────────────────────────
_BOARD_COLS = 10
_BOARD_ROWS = 7
_SQUARE_M   = 0.015   # 15 mm

# ── YOLO classifier ────────────────────────────────────────────────────────────
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "runs", "classify", "shape-3", "weights", "best.pt"
)


# ══════════════════════════════════════════════════════════════════════════════
# TABLE PLANE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_table_plane(board_cols=_BOARD_COLS, board_rows=_BOARD_ROWS):
    """Stream RealSense frames until chessboard is found; fit plane via SVD.
    Returns (table_normal, d) or (None, None) on ESC."""
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
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
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
# COMPUTE WORKER  (background thread — all heavy work lives here)
# ══════════════════════════════════════════════════════════════════════════════

class ComputeWorker:
    """
    Drains ObjectIsolator's frame queue on a background thread.

    For each frame:
      · shape_hint comes from YOLO (set inside ObjectIsolator._loop).
      · fit_once is called with that shape_hint — the shape is determined
        entirely by YOLO; no geometric classifier overrides it.
      · The result is stored in a single lock-protected slot.  The render
        thread reads it non-blocking; stale results are silently overwritten.
    """

    def __init__(self, isolator: ObjectIsolator, table_normal):
        self._isolator     = isolator
        self._table_normal = table_normal
        self._result       = None
        self._lock         = threading.Lock()
        self._stop         = threading.Event()
        self._thread       = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def latest(self):
        """Non-blocking.  Returns and clears the latest result dict, or None."""
        with self._lock:
            r, self._result = self._result, None
        return r

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _run(self):
        while not self._stop.is_set():
            # Block until a frame is available (with timeout to check stop flag)
            try:
                (verts, _, full_colors,
                 obj_verts, obj_colors,
                 preview_bgr, shape_hint) = \
                    self._isolator._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            has_obj = len(obj_verts) > 0
            shape   = None
            ls      = None

            if has_obj and shape_hint is not None:
                # Shape is determined entirely by YOLO — no geometric fallback.
                shape, ls = fit_once(
                    obj_verts, self._table_normal, shape_hint=shape_hint)
                self._isolator.last_shape = shape
            else:
                # Either no object or YOLO hasn't produced a confident result.
                self._isolator.last_shape = None

            with self._lock:
                self._result = {
                    "obj_verts":   obj_verts,
                    "obj_colors":  obj_colors,
                    "full_verts":  verts,
                    "full_colors": full_colors,
                    "preview_bgr": preview_bgr,
                    "shape":       shape,
                    "ls":          ls,
                    "has_obj":     has_obj,
                }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run(board_cols=_BOARD_COLS, board_rows=_BOARD_ROWS, debug=False):
    table_normal, _ = detect_table_plane(board_cols, board_rows)

    # ── Load YOLO classifier ───────────────────────────────────────────────────
    classifier = ShapeClassifier(_MODEL_PATH)

    isolator = ObjectIsolator(min_points=50, classifier=classifier)
    isolator.start()
    print("Waiting for camera …")
    isolator.ready.wait()
    print("Camera ready.\n")

    worker = ComputeWorker(isolator, table_normal)

    # ── Open3D window ──────────────────────────────────────────────────────────
    vis = o3d.visualization.Visualizer()
    vis.create_window("Shape Fit — Live", width=1280, height=720)

    pcd       = o3d.geometry.PointCloud()
    shape_ls  = o3d.geometry.LineSet()
    pcd_added = False
    ls_added  = False
    first_obj = True   # trigger auto-zoom on first valid detection

    CV2_WIN = "Camera Preview"
    cv2.namedWindow(CV2_WIN, cv2.WINDOW_NORMAL)

    last_shape = None

    print("[render]  Window open — close or press Ctrl+C / q / ESC to stop.")

    try:
        while True:
            result = worker.latest()

            if result is not None:

                # ── Choose which points to display ─────────────────────────
                if debug or not result["has_obj"]:
                    pts  = result["full_verts"]
                    cols = result["full_colors"]
                else:
                    pts  = result["obj_verts"]
                    cols = result["obj_colors"]

                # ── Update point cloud ─────────────────────────────────────
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(cols)

                if not pcd_added:
                    vis.add_geometry(pcd)
                    pcd_added = True
                else:
                    vis.update_geometry(pcd)

                # ── Auto-zoom on first isolated object ─────────────────────
                if result["has_obj"] and first_obj and len(pts) > 0:
                    ctr = vis.get_view_control()
                    ctr.set_lookat(pts.mean(axis=0).tolist())
                    ctr.set_front([0, 0, -1])
                    ctr.set_up([0, -1, 0])
                    ctr.set_zoom(0.5)
                    first_obj = False

                # ── Update wireframe ───────────────────────────────────────
                new_ls = result["ls"]

                if new_ls is not None:
                    shape_ls.points = new_ls.points
                    shape_ls.lines  = new_ls.lines
                    shape_ls.colors = new_ls.colors
                    if not ls_added:
                        vis.add_geometry(shape_ls, reset_bounding_box=False)
                        ls_added = True
                    else:
                        vis.update_geometry(shape_ls)

                    if result["shape"] != last_shape:
                        print(f"[render]  shape → {result['shape']}")
                        last_shape = result["shape"]

                elif ls_added and not result["has_obj"]:
                    vis.remove_geometry(shape_ls, reset_bounding_box=False)
                    ls_added  = False
                    first_obj = True   # re-zoom when next object appears
                    last_shape = None

                # ── cv2 preview ────────────────────────────────────────────
                if result["preview_bgr"] is not None:
                    cv2.imshow(CV2_WIN, result["preview_bgr"])

            # ── Service GUI event loops (zero computation) ─────────────────
            if not vis.poll_events():
                break
            vis.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down …")
        worker.stop()
        isolator.stop()
        cv2.destroyAllWindows()
        vis.destroy_window()
        print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--board-cols", type=int, default=_BOARD_COLS)
    p.add_argument("--board-rows", type=int, default=_BOARD_ROWS)
    p.add_argument("--debug",      action="store_true")
    a = p.parse_args()
    run(board_cols=a.board_cols, board_rows=a.board_rows, debug=a.debug)
