"""
test/test_lineset_live_shape.py
===============================
Live gripper-lineset overlay derived from the FITTED SHAPE geometry.

Instead of running GraspNet, the grasp pose is computed analytically from the
cylinder / cuboid parameters produced by shape_fitter.  Because those parameters
are already EMA-smoothed by ShapeTracker, the lineset converges smoothly and
then locks in place once stable.

Pipeline (mirrors test_shape_fit.py exactly, with grasp drawing layered on top)
------------------------------------------------------------------------------
  1. Detect table plane via chessboard               (detect_table_plane)
  2. Start ObjectIsolator                            (capture.object_isolation)
  3. show_isolated_pcd(isolator, on_new_frame=cb)    (helper.pcd_visualizer)
        → handles window, point-cloud display, frame accumulation, auto-zoom
  4. Per-frame callback `_make_grasp_callback`:
        a. ShapeFitThread → fit_and_track → shape type + smoothed parameters
        b. Update shape wireframe in the scene
        c. _grasp_from_shape → approach / closing vectors from shape geometry
        d. GraspSmoother → EMA blend + convergence lock
        e. Publish ROS2 messages (object pose, joint trajectory, hand pose)
        f. Update gripper lineset in the scene

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
import threading
import argparse

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "capture"))
sys.path.insert(0, _ROOT)

import numpy as np
import open3d as o3d

from capture.object_isolation import ObjectIsolator
from capture.shape_fitter import ShapeTracker, fit_and_track
from helper.pcd_visualizer import show_isolated_pcd
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

    def reset(self):
        """Forget the current pose — used when the object leaves the scene."""
        self._rot    = None
        self._trans  = None
        self._locked = False
        self._streak = 0

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


def _object_params(rot, trans, shape, tracker, shape_ls):
    """
    Compute distance, wrist orientation, and required hand position from the
    grasp pose.

    Camera frame convention (RealSense): X right, Y down, Z forward.

    Parameters
    ----------
    rot      : (3, 3) ndarray
                 rot[:, 0]  approach — palm → fingertips (toward object)
                 rot[:, 1]  closing  — axis between the two fingers
                 rot[:, 2]  cross(approach, closing)
    trans    : (3,) ndarray — object centroid in camera frame (metres)
    shape    : str  — "cylinder" | "cuboid"
    tracker  : ShapeTracker — provides tracker.radius for cylinder
    shape_ls : o3d.geometry.LineSet — cuboid wireframe vertices for cuboid

    Returns
    -------
    distance_m    : float
        Euclidean distance from the camera origin to the object centroid (m).

    elevation_deg : float  — UP / DOWN wrist rotation
        Elevation of the approach axis (palm → fingertips) from horizontal.
          0°   → wrist level, hand approaches horizontally
        +90°   → wrist tilted fully up, hand approaches from below aiming up
        -90°   → wrist tilted fully down, hand approaches from above aiming down

    bearing_deg   : float  — LEFT / RIGHT wrist rotation
        Bearing of the approach axis in the horizontal (XZ) plane.
          0°   → wrist straight forward (neutral)
        +90°   → wrist rotated 90° to the right
        -90°   → wrist rotated 90° to the left

    hand_pos : (3,) ndarray — palm centre in simulation frame (metres)
        XYZ position [X-forward, Y-left, Z-down] the palm must move to so
        that the fingertips just reach the near surface of the object.
        Computed as:
            cam_pos  = object_centroid - approach * (FINGER_LENGTH + obj_half_depth)
            hand_pos = [cam_z, -cam_x, cam_y]   (camera → simulation frame)
        where obj_half_depth is the object's extent from its centroid toward
        the camera along the approach direction (radius for a cylinder, face
        half-depth for a cuboid).

    Neutral reference — palm flat down, arm pointing forward along camera Z:
        approach = [0, 0, 1]  →  elevation = 0°,  bearing = 0°
    """
    distance_m = float(np.linalg.norm(trans))
    approach   = rot[:, 0]

    # UP / DOWN — elevation of the approach axis from horizontal
    # Camera Y is down, so the "up" component of approach is -approach[1]
    elevation_deg = float(np.degrees(np.arcsin(np.clip(-approach[1], -1.0, 1.0))))

    # LEFT / RIGHT — bearing of the approach axis in the horizontal (XZ) plane
    # arctan2(X-component, Z-component): 0° = straight forward, +90° = right, -90° = left
    bearing_deg = float(np.degrees(np.arctan2(approach[0], approach[2])))

    # How far the object surface is from its centroid along the approach direction
    if shape == "cylinder" and tracker.radius is not None:
        obj_half_depth = float(tracker.radius)
    elif shape == "cuboid" and shape_ls is not None:
        verts = np.asarray(shape_ls.points)
        if len(verts) >= 4:
            # Extent of the cuboid along the approach axis; near face is the
            # minimum projection (closest to the camera / palm side)
            proj           = verts @ approach
            obj_half_depth = float((proj.max() - proj.min()) / 2.0)
        else:
            obj_half_depth = 0.0
    else:
        obj_half_depth = 0.0

    # Pull palm back so fingertips reach the object surface (camera frame)
    cam_pos = trans - approach * (FINGER_LENGTH + obj_half_depth)

    # Convert camera frame (X-right, Y-down, Z-forward)
    #         → simulation frame (X-forward, Y-left, Z-down)
    hand_pos = np.array([cam_pos[2], -cam_pos[0], cam_pos[1]])

    return distance_m, elevation_deg, bearing_deg, hand_pos


def ros_spin(node):
    """Background ROS2 spinner so publishers stay responsive."""
    rclpy.spin(node)


# ══════════════════════════════════════════════════════════════════════════════
# ON-NEW-FRAME CALLBACK  (drop-in replacement for test_shape_fit overlay)
# ══════════════════════════════════════════════════════════════════════════════

def _make_grasp_callback(table_normal,
                         object_publisher,
                         trajectory_publisher,
                         pose_publisher):
    """
    Build a callback compatible with show_isolated_pcd that:
      · Runs shape fitting in a background thread (ShapeFitThread)
      · Draws the cylinder / cuboid wireframe
      · Derives a grasp pose from the fitted shape
      · Smooths the pose (GraspSmoother) and draws the gripper lineset
      · Publishes the corresponding object / trajectory / pose ROS2 messages

    Mirrors the structure of test_shape_fit._make_overlay_callback so the
    high-level pipeline (table → isolator → show_isolated_pcd → callback)
    stays identical to test_shape_fit.py.
    """
    tracker  = ShapeTracker()
    smoother = GraspSmoother()
    fitter   = ShapeFitThread(table_normal, tracker)

    # Persistent geometry handles + last-known labels
    state = {
        "shape_ls":   None,    # o3d.geometry.LineSet  — cylinder / cuboid wireframe
        "grasp_ls":   None,    # o3d.geometry.LineSet  — gripper lineset
        "label":      None,    # last shape label drawn
        "lookat_set": False,   # one-shot re-centre, same as test_shape_fit
    }

    def _remove_overlays(vis):
        """Drop both shape and grasp geometries from the scene."""
        if state["shape_ls"] is not None:
            vis.remove_geometry(state["shape_ls"], reset_bounding_box=False)
            state["shape_ls"] = None
        if state["grasp_ls"] is not None:
            vis.remove_geometry(state["grasp_ls"], reset_bounding_box=False)
            state["grasp_ls"] = None
        state["label"] = None

    def _on_frame(obj_verts: np.ndarray, vis: o3d.visualization.Visualizer):
        # ── Object lost — full reset so the next object starts fresh ────────
        if len(obj_verts) == 0:
            tracker.full_reset()
            smoother.reset()
            _remove_overlays(vis)
            vis.update_renderer()
            return

        # ── Hand frame off to background fitter ──────────────────────────────
        fitter.push(obj_verts)

        # ── Pull latest fit result (non-blocking) ────────────────────────────
        new_shape, new_shape_ls = fitter.pop()
        if new_shape_ls is None:
            return                        # fitter hasn't produced anything yet

        # ── Update / add the shape wireframe ─────────────────────────────────
        if new_shape != state["label"]:
            print(f"[shape_fit]  *** shape → {new_shape} ***")
            # Topology changed (cylinder ↔ cuboid) — remove + re-add
            if state["shape_ls"] is not None:
                vis.remove_geometry(state["shape_ls"], reset_bounding_box=False)
            vis.add_geometry(new_shape_ls, reset_bounding_box=False)
            state["shape_ls"] = new_shape_ls
            state["label"]    = new_shape
        else:
            ls = state["shape_ls"]
            ls.points = new_shape_ls.points
            ls.lines  = new_shape_ls.lines
            ls.colors = new_shape_ls.colors
            vis.update_geometry(ls)

        # Re-centre once on the first successful fit (same trick as test_shape_fit)
        if not state["lookat_set"]:
            vis.get_view_control().set_lookat(obj_verts.mean(axis=0).tolist())
            state["lookat_set"] = True

        # ── Derive grasp pose from the fitted shape ──────────────────────────
        new_rot, new_trans = _grasp_from_shape(new_shape, tracker, new_shape_ls)
        if new_rot is None:
            return

        rot, trans = smoother.update(new_rot, new_trans)
        distance_m, elevation_deg, bearing_deg, hand_pos = \
            _object_params(rot, trans, new_shape, tracker, new_shape_ls)

        # ── Publish ROS2 messages ────────────────────────────────────────────
        elevation_rad = np.deg2rad(elevation_deg)
        bearing_rad   = np.deg2rad(bearing_deg)

        # Send object type and position
        object_msg = Float64MultiArray()
        shape_id   = 1. if new_shape == "cylinder" else 0.
        object_msg.data = [shape_id, distance_m, 0., 0., 0., 0., 0.]
        object_publisher.publish(object_msg)

        # Send prosthesis trajectory (arm base movement, wrist rotation)
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names.append('joint_base_x')
        trajectory_msg.joint_names.append('joint_base_y')
        trajectory_msg.joint_names.append('joint_base_z')
        trajectory_msg.joint_names.append('joint_wrist_x')
        trajectory_msg.joint_names.append('joint_wrist_y')
        point = JointTrajectoryPoint()
        point.time_from_start = Duration(sec=5, nanosec=0)
        point.positions.append(hand_pos[0] - 0.12)  # 12 cm offset (wrist base origin)
        point.positions.append(hand_pos[1])
        point.positions.append(hand_pos[2])
        point.positions.append(bearing_rad)
        point.positions.append(elevation_rad)
        trajectory_msg.points.append(point)
        trajectory_publisher.publish(trajectory_msg)

        # Send close-hand command to simulation
        pose_msg = Int8()
        pose_msg.data = 1
        pose_publisher.publish(pose_msg)

        # ── Update / add the gripper lineset ─────────────────────────────────
        new_ls = _gripper_lineset(rot, trans)
        if state["grasp_ls"] is None:
            vis.add_geometry(new_ls, reset_bounding_box=False)
            state["grasp_ls"] = new_ls
        else:
            gls = state["grasp_ls"]
            gls.points = new_ls.points
            gls.lines  = new_ls.lines
            gls.colors = new_ls.colors
            vis.update_geometry(gls)

    # Expose the fitter so the caller can cleanly shut down the worker thread.
    _on_frame.fitter = fitter
    return _on_frame


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

    # ── ROS2 node start ────────────────────────────────────────────────────────
    rclpy.init()
    node = Node('CV_publisher_node')
    object_publisher     = node.create_publisher(Float64MultiArray, '/cv/model', 10)
    trajectory_publisher = node.create_publisher(
        JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
    pose_publisher       = node.create_publisher(Int8, '/pose_goals', 10)

    ros_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    ros_thread.start()

    # ── Build the per-frame callback that fits shape + draws lineset ──────────
    on_frame = _make_grasp_callback(table_normal,
                                    object_publisher,
                                    trajectory_publisher,
                                    pose_publisher)

    try:
        # Same pipeline call as test_shape_fit.run — no custom render loop here
        show_isolated_pcd(
            isolator,
            title="Shape-based Grasp — Live",
            on_new_frame=on_frame,
            camera_up=(0, -1, 0),   # RealSense Y-down convention
        )
    finally:
        print("\n[test_lineset_live_shape] shutting down...")
        on_frame.fitter.stop()
        isolator.stop()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        ros_thread.join(timeout=2)
        print("[test_lineset_live_shape] done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-cols", type=int, default=_BOARD_COLS)
    parser.add_argument("--board-rows", type=int, default=_BOARD_ROWS)
    args = parser.parse_args()
    run(board_cols=args.board_cols, board_rows=args.board_rows)
