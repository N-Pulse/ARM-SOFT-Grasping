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
from collections import deque

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "capture"))
sys.path.insert(0, _ROOT)

import cv2
import numpy as np
import open3d as o3d

from capture.object_isolation import ObjectIsolator, keep_largest_cluster
from capture.shape_fitter import ShapeTracker, fit_and_track
from helper.pcd_visualizer import auto_zoom
from test_shape_fit import detect_table_plane

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Int8
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration

# ══════════════════════════════════════════════════════════════════════════════
# Background fit thread
# ══════════════════════════════════════════════════════════════════════════════

class ShapeFitThread:
    """
    Background thread that runs fit_and_track on incoming point-cloud frames.

    A single-slot input queue holds the most recent obj_verts; stale frames
    are dropped if the worker is busy.  The latest (shape, ls) result is
    stored internally and consumed by pop().
    """

    def __init__(self, table_normal, tracker):
        self._table_normal = table_normal
        self._tracker      = tracker
        self._in           = queue.Queue(maxsize=1)
        self._out          = {"shape": None, "ls": None}
        self._lock         = threading.Lock()
        self._thread       = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            try:
                verts = self._in.get(timeout=0.5)
            except queue.Empty:
                continue
            if verts is None:       # sentinel — shut down
                break
            shape, ls = fit_and_track(verts, self._table_normal, self._tracker)
            if ls is not None:
                with self._lock:
                    self._out["shape"] = shape
                    self._out["ls"]    = ls

    def push(self, verts: np.ndarray):
        """Push a new frame (drops the queued one if the worker is busy)."""
        try:
            self._in.get_nowait()
        except queue.Empty:
            pass
        self._in.put(verts.copy())

    def pop(self):
        """Return (shape, ls) if a new result is ready, else (None, None)."""
        with self._lock:
            shape           = self._out["shape"]
            ls              = self._out["ls"]
            self._out["ls"] = None  # consume so the same result is not returned twice
        return shape, ls

    def stop(self):
        """Send shutdown sentinel to the worker thread."""
        self._in.put(None)


# ── Chessboard defaults ────────────────────────────────────────────────────────
_BOARD_COLS = 10
_BOARD_ROWS = 7

# ── Frame accumulation (mirrors pcd_visualizer.show_isolated_pcd) ─────────────
_ACCUM_FRAMES  = 20      # number of past frames merged into the shape-fit cloud
_ACCUM_VOXEL_M = 0.003   # voxel size (m) for downsampling the accumulated cloud
_MOVE_THRESH_M = 0.015   # clear history when object moves more than this (m)

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


def _object_params(rot, trans):
    """
    Compute distance and wrist orientation from the grasp pose.

    Camera frame convention (RealSense): X right, Y down, Z forward.

    Parameters
    ----------
    rot   : (3, 3) ndarray
              rot[:, 0]  approach — palm → fingertips (toward object)
              rot[:, 1]  closing  — axis between the two fingers
              rot[:, 2]  cross(approach, closing)
    trans : (3,) ndarray — object centroid in camera frame (metres)

    Returns
    -------
    distance_m    : float
        Euclidean distance from the camera origin to the object centroid (m).

    elevation_deg : float  — UP / DOWN wrist rotation
        Elevation of the approach axis (palm → fingertips) from horizontal.
        Describes how much the wrist is tilted upward or downward.
          0°   → wrist level, hand approaches horizontally
        +90°   → wrist tilted fully up, hand approaches from below aiming up
        -90°   → wrist tilted fully down, hand approaches from above aiming down

    bearing_deg   : float  — LEFT / RIGHT wrist rotation
        Bearing of the approach axis in the horizontal (XZ) plane.
        Describes how much the wrist has rotated left or right from straight ahead.
          0°   → wrist straight forward (neutral)
        +90°   → wrist rotated 90° to the right
        -90°   → wrist rotated 90° to the left

    Neutral reference — palm flat down, arm pointing forward along camera Z:
        approach = [0, 0, 1]  →  elevation = 0°,  bearing = 0°
    """
    distance_m = float(np.linalg.norm(trans))
    approach   = rot[:, 0]

    # UP / DOWN — elevation of the approach axis from horizontal
    # Camera Y is down, so the "up" component of approach is -approach[1]
    # Positive = wrist tilted upward (hand aims above horizontal)
    # Negative = wrist tilted downward (hand aims below horizontal)
    elevation_deg = float(np.degrees(np.arcsin(np.clip(-approach[1], -1.0, 1.0))))

    # LEFT / RIGHT — bearing of the approach axis in the horizontal (XZ) plane
    # arctan2(X-component, Z-component): 0° = straight forward, +90° = right, -90° = left
    # Positive = wrist rotated to the right, negative = rotated to the left
    bearing_deg = float(np.degrees(np.arctan2(approach[0], approach[2])))

    return distance_m, elevation_deg, bearing_deg


def ros_spin(node): # for continuous publishing
    rclpy.spin(node)

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

    # ── Shape-fit background thread ────────────────────────────────────────────
    tracker  = ShapeTracker()
    smoother = GraspSmoother()
    fitter   = ShapeFitThread(table_normal, tracker)
    
    # ── ROS2 node start ────────────────────────────────────────────────────────
    rclpy.init() # initialize ROS2
    node = Node('CV_publisher_node')
    object_publisher = node.create_publisher(Float64MultiArray, '/cv/model', 10)
    trajectory_publisher = node.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
    pose_publisher = node.create_publisher(Int8, 'pose_goals', 10)

    ros_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    ros_thread.start()

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

    pcd      = o3d.geometry.PointCloud()
    grasp_ls = _empty_lineset()

    geom_added     = False
    zoom_fitted    = False
    last_grasp     = None
    grasp_ls_ready = False
    shape_ls_geom  = None
    last_shape     = None

    # ── Frame accumulation (same logic as pcd_visualizer.show_isolated_pcd) ──
    _accum_buf       = deque(maxlen=_ACCUM_FRAMES)
    _prev_centroid   = None

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

            # ── Accumulate frames and push dense cloud to shape fitter ────────
            if len(obj_verts) > 0:
                # Clear history if the object moved significantly
                curr_centroid = obj_verts.mean(axis=0)
                if _prev_centroid is not None:
                    if np.linalg.norm(curr_centroid - _prev_centroid) > _MOVE_THRESH_M:
                        _accum_buf.clear()
                _prev_centroid = curr_centroid

                _accum_buf.append((obj_verts.copy(), obj_colors.copy()))

                # Merge, cluster-filter, and voxel-downsample accumulated cloud
                all_pts  = np.concatenate([v for v, _ in _accum_buf])
                all_cols = np.concatenate([c for _, c in _accum_buf])
                all_pts, all_cols = keep_largest_cluster(all_pts, all_cols)
                tmp = o3d.geometry.PointCloud()
                tmp.points = o3d.utility.Vector3dVector(all_pts)
                if _ACCUM_VOXEL_M > 0:
                    tmp = tmp.voxel_down_sample(_ACCUM_VOXEL_M)
                fit_pts = np.asarray(tmp.points)

                fitter.push(fit_pts)
            else:
                _accum_buf.clear()
                _prev_centroid = None

        except queue.Empty:
            pass

        # ── Pull latest shape wireframe + compute grasp ───────────────────────
        new_shape, new_shape_ls = fitter.pop()
        new_grasp = None
        if new_shape_ls is not None:
            new_rot, new_trans = _grasp_from_shape(new_shape, tracker, new_shape_ls)
            if new_rot is not None:
                rot, trans = smoother.update(new_rot, new_trans)
                new_grasp = (rot, trans)
                distance_m, elevation_deg, bearing_deg = _object_params(rot, trans)
                
                # Send object type and position
                object_msg = Float64MultiArray()
                shape = 1. # 0 for cube, 1 for cylinder
                object_msg.data = [shape, distance_m, 0., 0., 0., 0., 0.] #[object_type, x, y, z, roll, pitch, yaw]
                object_publisher.publish(object_msg)

                # Send wrist rotation
                trajectory_msg = JointTrajectory()
                trajectory_msg.joint_names.append('joint_wrist_x')
                trajectory_msg.joint_names.append('joint_wrist_y')
                point = JointTrajectoryPoint()
                point.time_from_start = 5
                point.positions.append(bearing_deg)
                point.positions.append(elevation_deg)
                trajectory_msg.points.append(point)
                trajectory_publisher.publish(trajectory_msg)

                # Send close hand command to simulation
                pose_msg = Int8()
                pose_msg.data = 1
                pose_publisher.publish(pose_msg)    

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
    _stop.set()  # Signal stop
    fitter.stop()
    isolator.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()
    ros_thread.join(timeout=2)  # Wait for ROS thread to finish
    print("[test_lineset_live_shape] done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-cols", type=int, default=_BOARD_COLS)
    parser.add_argument("--board-rows", type=int, default=_BOARD_ROWS)
    args = parser.parse_args()
    run(board_cols=args.board_cols, board_rows=args.board_rows)
