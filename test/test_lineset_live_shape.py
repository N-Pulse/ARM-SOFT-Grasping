"""
test/test_lineset_live_shape.py
===============================
Live gripper-lineset overlay derived from the FITTED SHAPE geometry.

Instead of running GraspNet, the grasp pose is computed analytically from the
cylinder / cuboid parameters produced by shape_fitter.  Because those parameters
are already EMA-smoothed by ShapeTracker, the lineset converges smoothly and
then locks in place once stable.

Pipeline
--------
  1. Detect table plane via chessboard (reused from test_shape_fit.py)
  2. ObjectIsolator  →  isolated object point cloud per frame
  3. ShapeFitThread  →  fit_and_track → shape type + smoothed parameters
  4. _grasp_from_shape  →  approach / closing vectors from shape geometry
  5. GraspSmoother  →  EMA blend + convergence lock
  6. Open3D window shows: raw point cloud + shape wireframe + gripper lineset

Usage
-----
    python test/test_lineset_live_shape.py
    python test/test_lineset_live_shape.py --board-cols 10 --board-rows 7

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

from capture.object_isolation import ObjectIsolator
from capture.shape_fitter import ShapeTracker, fit_and_track
from helper.pcd_visualizer import auto_zoom
from test_shape_fit import detect_table_plane


# ── Chessboard defaults ────────────────────────────────────────────────────────
_BOARD_COLS = 10
_BOARD_ROWS = 7

# ── Gripper geometry ───────────────────────────────────────────────────────────
FINGER_LENGTH   = 0.085
PALM_DEPTH      = 0.06
FINGER_DISTANCE = 0.06    # fixed distance between the two finger tips (m)
GRIPPER_COLOR   = [1.0, 0.4, 0.0]


# ══════════════════════════════════════════════════════════════════════════════
# GRASP POSE FROM FITTED SHAPE
# ══════════════════════════════════════════════════════════════════════════════

def _grasp_from_shape(shape, tracker, shape_ls):
    """
    Compute (rot, trans) analytically from the fitted shape.

    Convention (matches _gripper_lineset / GraspNet):
      rot[:, 0]  approach direction — from palm toward finger tips (toward object)
      rot[:, 1]  closing direction  — between the two fingers
      trans      palm centre

    Cylinder
    --------
    The gripper approaches horizontally (perpendicular to the vertical axis)
    from the direction of the camera.  The closing direction runs along the
    cylinder axis so the fingers span its height.

    Cuboid
    ------
    Same camera-toward approach.  The closing direction is aligned with
    whichever horizontal footprint edge is most perpendicular to the approach
    (maximising grip contact with the shorter face).

    Returns (rot, trans) or (None, None) on failure.
    """
    if shape == "cylinder":
        axis    = tracker.axis
        axis_pt = tracker.axis_pt
        if axis is None or axis_pt is None or tracker.h_ctr is None:
            return None, None

        h_min = tracker.h_ctr - tracker.height / 2.0
        h_max = tracker.h_ctr + tracker.height / 2.0
        trans = axis_pt + axis * (h_min + h_max) / 2.0   # cylinder centroid

        # Approach: camera (origin) → cylinder centre, projected ⊥ to axis
        to_obj   = trans / (np.linalg.norm(trans) + 1e-9)
        approach = to_obj - (to_obj @ axis) * axis
        nrm      = np.linalg.norm(approach)
        if nrm < 1e-6:
            ref      = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 \
                       else np.array([0., 1., 0.])
            approach = np.cross(axis, ref)
            approach /= np.linalg.norm(approach)
        else:
            approach = approach / nrm

        # Closing: along axis so fingers span the cylinder height
        closing = np.cross(axis, approach)
        closing /= np.linalg.norm(closing)
        third   = np.cross(approach, closing)

    elif shape == "cuboid":
        verts = np.asarray(shape_ls.points)   # (8, 3)
        if len(verts) < 8:
            return None, None

        trans = verts.mean(axis=0)            # cuboid centroid

        # Approach: camera (origin) → cuboid centre
        approach = trans / (np.linalg.norm(trans) + 1e-9)

        # Bottom corners in CCW order: indices 0, 2, 4, 6
        bot = verts[::2]
        e1  = bot[1] - bot[0];  e1 /= (np.linalg.norm(e1) + 1e-9)
        e2  = bot[2] - bot[1];  e2 /= (np.linalg.norm(e2) + 1e-9)

        # Pick the footprint edge most perpendicular to approach for closing
        closing = e2 if abs(approach @ e1) < abs(approach @ e2) else e1
        # Orthogonalise against approach
        closing = closing - (closing @ approach) * approach
        nrm     = np.linalg.norm(closing)
        if nrm < 1e-6:
            ref     = np.array([0., -1., 0.])
            closing = np.cross(approach, ref)
            closing /= np.linalg.norm(closing)
        else:
            closing /= nrm

        third = np.cross(approach, closing)

    else:
        return None, None

    rot = np.column_stack([approach, closing, third])
    # Ensure proper rotation matrix (det = +1)
    if np.linalg.det(rot) < 0:
        rot[:, 2] = -rot[:, 2]
    return rot, trans


# ══════════════════════════════════════════════════════════════════════════════
# GRASP SMOOTHER  (EMA + convergence lock)
# ══════════════════════════════════════════════════════════════════════════════

class GraspSmoother:
    """
    Blends successive (rot, trans) estimates with EMA and locks the pose once
    it has been stable for N_LOCK consecutive frames.

    While locked, small per-frame updates are ignored entirely.
    The lock breaks automatically if the shape centre moves more than
    UNLOCK_DIST (e.g. the object was picked up or replaced).
    """
    ALPHA       = 0.25    # EMA rate  (0 = never update, 1 = no smoothing)
    LOCK_DIST   = 0.004   # m — max per-update movement to count as stable
    N_LOCK      = 10      # consecutive stable frames needed to lock
    UNLOCK_DIST = 0.025   # m — movement that breaks the lock

    def __init__(self):
        self._rot    = None
        self._trans  = None
        self._locked = False
        self._streak = 0

    def update(self, new_rot, new_trans):
        """Blend new estimate in and return current (rot, trans)."""
        if self._trans is None:
            self._trans  = new_trans.copy()
            self._rot    = new_rot.copy()
            self._streak = 0
            return self._rot, self._trans

        # Break lock if object moved significantly
        if self._locked:
            if np.linalg.norm(new_trans - self._trans) > self.UNLOCK_DIST:
                print("[grasp]  pose UNLOCKED — object moved")
                self._locked = False
                self._streak = 0

        if not self._locked:
            # EMA on translation
            sm_trans = (1 - self.ALPHA) * self._trans + self.ALPHA * new_trans

            # EMA on rotation matrix, re-orthogonalised via SVD
            avg      = (1 - self.ALPHA) * self._rot + self.ALPHA * new_rot
            U, _, Vt = np.linalg.svd(avg)
            sm_rot   = U @ Vt
            if np.linalg.det(sm_rot) < 0:
                U[:, -1] *= -1
                sm_rot = U @ Vt

            # Count toward lock if movement is small
            if np.linalg.norm(sm_trans - self._trans) < self.LOCK_DIST:
                self._streak += 1
            else:
                self._streak = 0

            self._trans = sm_trans
            self._rot   = sm_rot

            if self._streak >= self.N_LOCK:
                self._locked = True
                print(f"[grasp]  *** pose LOCKED  trans={self._trans.round(3)} ***")

        return self._rot, self._trans

    @property
    def result(self):
        if self._trans is None:
            return None
        return self._rot, self._trans


# ══════════════════════════════════════════════════════════════════════════════
# GRIPPER LINESET
# ══════════════════════════════════════════════════════════════════════════════

def _gripper_lineset(rot, trans):
    approach   = rot[:, 0]
    closing    = rot[:, 1]
    half_w     = FINGER_DISTANCE / 2.0

    tip_center = trans + approach * FINGER_LENGTH
    palm_back  = trans - approach * PALM_DEPTH
    left_root  = trans      + closing * half_w
    right_root = trans      - closing * half_w
    left_tip   = tip_center + closing * half_w
    right_tip  = tip_center - closing * half_w

    pts   = np.array([palm_back, trans, left_root, right_root,
                      left_tip,  right_tip], dtype=np.float64)
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

def run(board_cols=_BOARD_COLS, board_rows=_BOARD_ROWS):
    # ── Table plane ────────────────────────────────────────────────────────────
    table_normal, _ = detect_table_plane(board_cols, board_rows)

    # ── ObjectIsolator ─────────────────────────────────────────────────────────
    isolator = ObjectIsolator(min_points=50)
    isolator.start()
    print("Waiting for YOLO to load...")
    isolator.ready.wait()
    print("YOLO ready — opening viewer.\n")
    print("Running — close the Open3D window or Ctrl+C to stop.\n")

    # ── Shape-fit + grasp background thread ───────────────────────────────────
    tracker  = ShapeTracker()
    smoother = GraspSmoother()

    _fit_in   = queue.Queue(maxsize=1)
    _fit_out  = {"shape": None, "shape_ls": None, "grasp": None}
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

            new_rot, new_trans = _grasp_from_shape(shape, tracker, shape_ls)
            if new_rot is None:
                continue

            rot, trans = smoother.update(new_rot, new_trans)

            with _fit_lock:
                _fit_out["shape"]    = shape
                _fit_out["shape_ls"] = shape_ls
                _fit_out["grasp"]    = (rot, trans)

    fit_thread = threading.Thread(target=_fit_worker, daemon=True)
    fit_thread.start()

    # ── Clean-exit flag ────────────────────────────────────────────────────────
    _stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: _stop.set())

    # ── Open3D window ──────────────────────────────────────────────────────────
    CV2_WIN = "YOLO Detection"
    cv2.namedWindow(CV2_WIN, cv2.WINDOW_NORMAL)

    vis = o3d.visualization.Visualizer()
    vis.create_window("Shape-based Grasp — Live", width=1280, height=720)
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
    shape_ls_geom  = None
    last_shape     = None

    RENDER_INTERVAL = 1.0 / 30.0
    _last_render    = 0.0
    frame_ready     = False

    while not _stop.is_set():
        # ── Pull latest frame ────────────────────────────────────────────────
        frame_ready = False
        try:
            verts, _, full_colors, obj_verts, obj_colors, preview_bgr = \
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

            # Push to shape fitter (drop stale frame if worker is busy)
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
            new_grasp    = _fit_out["grasp"]
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
                shape_ls_geom.points = new_shape_ls.points
                shape_ls_geom.lines  = new_shape_ls.lines
                shape_ls_geom.colors = new_shape_ls.colors
                vis.update_geometry(shape_ls_geom)

        # ── Update gripper lineset from smoothed shape-based grasp ───────────
        if new_grasp is not None and new_grasp is not last_grasp:
            last_grasp = new_grasp
            rot, trans = new_grasp
            new_ls = _gripper_lineset(rot, trans)
            if not grasp_ls_ready and geom_added:
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
    _fit_in.put(None)
    isolator.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()
    print("[test_lineset_live_shape] done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-cols", type=int, default=_BOARD_COLS)
    parser.add_argument("--board-rows", type=int, default=_BOARD_ROWS)
    args = parser.parse_args()
    run(board_cols=args.board_cols, board_rows=args.board_rows)
