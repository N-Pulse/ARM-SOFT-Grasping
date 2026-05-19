"""
test/test_lineset_live_shape.py
===============================
Live gripper-lineset overlay derived from the FITTED SHAPE geometry.

Threading model
---------------
  Background worker thread   FitWorker
    · fit_and_track         (shape fitting + EMA + lock voting)
    · _grasp_from_shape     (pose from smoothed shape params)
    · _fill_gripper_pts     (6-point gripper skeleton)
    · _object_params        (only on lock rising edge)
    · ROS2 publish          (only on lock rising edge)
    → writes a result dict into a single shared slot under a lock

  Main / render thread       _on_frame callback (via show_isolated_pcd)
    · pulls latest result from the worker
    · applies geometry updates to the Open3D Visualizer
    → does NO computation: only add_geometry / update_geometry / remove_geometry

Because all the heavy work lives on the worker, the render loop stays
responsive and the display no longer lags behind the camera.

Convergence clock
-----------------
ShapeTracker EMA-smooths the shape parameters and toggles ``_shape_locked``
after _N_LOCK consistent votes.  ``_grasp_from_shape`` is a deterministic
function of those smoothed parameters, so the grasp pose converges and
locks on exactly the same clock as the shape — no second smoother.

Pipeline (identical to test_shape_fit.py)
-----------------------------------------
  1. Detect table plane via chessboard         (detect_table_plane)
  2. Start ObjectIsolator                      (capture.object_isolation)
  3. show_isolated_pcd(isolator, on_new_frame) (helper.pcd_visualizer)
  4. Per-frame callback handles geometry only.

Usage
-----
    python test/test_lineset_live_shape.py
    python test/test_lineset_live_shape.py --board-cols 10 --board-rows 7

Controls
--------
    Close Open3D window  |  Ctrl+C  |  ESC / q in the camera preview

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
from std_msgs.msg import Int8, Float64MultiArray
from builtin_interfaces.msg import Duration


# ── Chessboard defaults ────────────────────────────────────────────────────────
_BOARD_COLS = 10
_BOARD_ROWS = 7

# ── Gripper geometry ───────────────────────────────────────────────────────────
FINGER_LENGTH   = 0.085
PALM_DEPTH      = 0.06
FINGER_DISTANCE = 0.06
GRIPPER_COLOR   = [1.0, 0.4, 0.0]
_GRIPPER_LINES  = [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5]]

# ── ROS joint trajectory ───────────────────────────────────────────────────────
_JOINT_NAMES = ['joint_base_x', 'joint_base_y', 'joint_base_z',
                'joint_base_roll',
                'joint_wrist_x', 'joint_wrist_y']


# ══════════════════════════════════════════════════════════════════════════════
# Pure-math helpers (run on the worker thread)
#
# Every helper below derives its result purely from the FITTED SHAPE — i.e.
# the EMA-smoothed cylinder parameters held by ShapeTracker, or the 8
# wireframe vertices of the fitted cuboid in shape_ls.points.  None of them
# read the raw point cloud directly.
# ══════════════════════════════════════════════════════════════════════════════

def _shape_centroid(shape, tracker, shape_ls):
    """
    Geometric centroid of the FITTED shape (not of the point cloud).

    Cylinder : axis_pt + axis * h_ctr           (uses smoothed tracker params)
    Cuboid   : mean of the 8 fitted corners     (uses shape_ls wireframe)
    """
    if shape == "cylinder":
        if tracker.axis is None or tracker.axis_pt is None or tracker.h_ctr is None:
            return None
        return tracker.axis_pt + tracker.axis * tracker.h_ctr

    if shape == "cuboid" and shape_ls is not None:
        verts = np.asarray(shape_ls.points)
        if len(verts) >= 8:
            return verts.mean(axis=0)
    return None


def _grasp_from_shape(shape, tracker, shape_ls):
    """
    Compute (rot, trans) from the fitted shape, GraspNet convention.

      rot[:, 0]  approach  — gripper-frame +X, palm → fingertips
      rot[:, 1]  closing   — gripper-frame +Y, between the two fingertips
      rot[:, 2]  cross(approach, closing)
      trans      tool centre point (TCP) = object centroid

    Both grippers approach the object HORIZONTALLY (in the table plane).
    The approach axis is the camera→centroid direction projected onto the
    table plane; the closing axis is perpendicular to approach within that
    plane.  This matches GraspNet's MIN_APPROACH_Z filter, which rejects
    grasps coming from above/below the table.
    """
    trans = _shape_centroid(shape, tracker, shape_ls)
    if trans is None:
        return None, None

    # ── Vertical (table) axis ────────────────────────────────────────────────
    # Cylinder: tracker.axis is the EMA-smoothed table normal.
    # Cuboid : tracker.axis is None (tracker.reset()'d) — recover it from the
    #          fitted wireframe's vertical pillars instead.
    if shape == "cylinder":
        axis = tracker.axis
    elif shape == "cuboid":
        verts = np.asarray(shape_ls.points)
        if len(verts) < 8:
            return None, None
        axis = (verts[1::2] - verts[::2]).mean(axis=0)   # mean vertical pillar
        n    = np.linalg.norm(axis)
        if n < 1e-9:
            return None, None
        axis = axis / n
    else:
        return None, None
    if axis is None:
        return None, None

    # ── Approach: (camera → centroid) projected onto the table plane ─────────
    to_obj   = trans / (np.linalg.norm(trans) + 1e-9)
    approach = to_obj - (to_obj @ axis) * axis
    nrm      = np.linalg.norm(approach)
    if nrm < 1e-6:
        # Object straight above/below camera — pick an arbitrary horizontal
        ref      = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 \
                   else np.array([0., 1., 0.])
        approach = np.cross(axis, ref)
        approach /= np.linalg.norm(approach)
    else:
        approach /= nrm

    # ── Closing: in the table plane, perpendicular to approach ───────────────
    if shape == "cylinder":
        # Perpendicular to both the cylinder axis and approach
        closing  = np.cross(axis, approach)
        closing /= np.linalg.norm(closing)
    else:  # cuboid
        # Pick the bottom-face edge most perpendicular to approach so the
        # fingers grip the shorter cross-section.
        bot = verts[::2]
        e1  = bot[1] - bot[0];  e1 /= (np.linalg.norm(e1) + 1e-9)
        e2  = bot[2] - bot[1];  e2 /= (np.linalg.norm(e2) + 1e-9)
        closing = e2 if abs(approach @ e1) < abs(approach @ e2) else e1
        # Orthogonalise against approach
        closing = closing - (closing @ approach) * approach
        nrm     = np.linalg.norm(closing)
        if nrm < 1e-6:
            closing  = np.cross(axis, approach)
            closing /= np.linalg.norm(closing)
        else:
            closing /= nrm

    third = np.cross(approach, closing)
    rot   = np.column_stack([approach, closing, third])
    if np.linalg.det(rot) < 0:
        rot[:, 2] = -rot[:, 2]
    return rot, trans


def _fill_gripper_pts(pts: np.ndarray, rot, trans):
    """
    Write a 6-point gripper skeleton into the preallocated buffer.

    GraspNet convention — `trans` is the TCP (between the fingertips) and
    `approach` points FROM the gripper INTO the object.  The palm sits
    BEHIND the TCP along -approach by FINGER_LENGTH; palm_back extends one
    further PALM_DEPTH back.

        palm_back ─── palm ─┬─── left_root ──── left_tip
                            └─── right_root ─── right_tip

    Lines (see _GRIPPER_LINES):
        [palm_back→palm], [palm→l_root], [palm→r_root],
        [l_root→l_tip],   [r_root→r_tip]
    """
    approach = rot[:, 0]
    closing  = rot[:, 1]
    half_w   = FINGER_DISTANCE / 2.0

    palm       = trans - approach * FINGER_LENGTH
    palm_back  = palm  - approach * PALM_DEPTH
    left_root  = palm  + closing * half_w
    right_root = palm  - closing * half_w
    left_tip   = trans + closing * half_w     # fingertips at the TCP
    right_tip  = trans - closing * half_w

    pts[0] = palm_back
    pts[1] = palm
    pts[2] = left_root
    pts[3] = right_root
    pts[4] = left_tip
    pts[5] = right_tip


def _object_params(rot, trans, shape, tracker, shape_ls):
    """
    Returns (distance_m, elevation_deg, bearing_deg, hand_pos) in the
    simulation frame (X forward, Y left, Z down).
    """
    distance_m = float(np.linalg.norm(trans))
    approach   = rot[:, 0]

    elevation_deg = float(np.degrees(np.arcsin(np.clip(-approach[1], -1.0, 1.0))))
    bearing_deg   = float(np.degrees(np.arctan2(approach[0], approach[2])))

    if shape == "cylinder" and tracker.radius is not None:
        obj_half_depth = float(tracker.radius)
    elif shape == "cuboid" and shape_ls is not None:
        verts = np.asarray(shape_ls.points)
        if len(verts) >= 4:
            proj = verts @ approach
            obj_half_depth = float((proj.max() - proj.min()) / 2.0)
        else:
            obj_half_depth = 0.0
    else:
        obj_half_depth = 0.0

    cam_pos  = trans - approach * FINGER_LENGTH
    hand_pos = np.array([cam_pos[2], -cam_pos[0], cam_pos[1]])
    return distance_m, elevation_deg, bearing_deg, hand_pos


# ══════════════════════════════════════════════════════════════════════════════
# ROS2 NODE
# ══════════════════════════════════════════════════════════════════════════════

class CVPublisherNode(Node):
    def __init__(self):
        super().__init__('CV_publisher_node')
        self.object_spawn_feedback = 0.
        self.object_pub = self.create_publisher(Float64MultiArray, '/cv/model/pose', 10)
        self.traj_pub   = self.create_publisher(JointTrajectory, '/cv/base/pose', 10)
        self.pose_pub   = self.create_publisher(Int8, '/cv/hand/pose', 10)
        #self.create_subscription(Int8, '/cv/model/pose/feedback', self.object_feedback, 10)

    def object_feedback(self, msg):
        self.object_spawn_feedback = msg.data


# ══════════════════════════════════════════════════════════════════════════════
# FitWorker — all calculation lives here, off the render thread
# ══════════════════════════════════════════════════════════════════════════════

class FitWorker:
    """
    Background pipeline runner.

    Input  : point clouds pushed via .push(verts)  (drops stale frames)
    Output : latest result dict polled via .pop()  (consumed once)

    The result dict, when present, contains:
        shape       : "cylinder" | "cuboid"
        shape_ls    : o3d.geometry.LineSet (fresh each fit)
        grasp_pts   : (6, 3) np.ndarray (gripper skeleton, valid iff has_grasp)
        has_grasp   : bool
        centroid    : (3,) np.ndarray  — used for one-shot lookat
    """

    def __init__(self, table_normal, node):
        self._table_normal = table_normal
        self._node         = node

        self._tracker      = ShapeTracker()
        self._was_locked   = False

        self._in           = queue.Queue(maxsize=1)
        self._out          = None
        self._lock         = threading.Lock()
        self._thread       = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ── Public API (called from main thread) ──────────────────────────────────

    def push(self, verts: np.ndarray):
        try:
            self._in.get_nowait()
        except queue.Empty:
            pass
        self._in.put(verts.copy())

    def pop(self):
        with self._lock:
            r = self._out
            self._out = None
        return r

    def reset(self):
        """Object lost — clear tracker + any pending result."""
        self._tracker.full_reset()
        self._was_locked = False
        with self._lock:
            self._out = None

    def stop(self):
        self._in.put(None)

    # ── Worker loop ───────────────────────────────────────────────────────────

    def _run(self):
        while True:
            try:
                verts = self._in.get(timeout=0.5)
            except queue.Empty:
                continue
            if verts is None:
                break

            # 1. Shape fit — the raw point cloud is consumed HERE and never
            #    looked at again downstream.  All subsequent geometry comes
            #    from the fitted shape (tracker params + shape_ls vertices).
            shape, shape_ls = fit_and_track(
                verts, self._table_normal, self._tracker)
            if shape_ls is None:
                continue

            # 2. Grasp pose from the fitted shape (same clock as the shape)
            rot, trans = _grasp_from_shape(shape, self._tracker, shape_ls)
            has_grasp  = rot is not None

            grasp_pts = np.zeros((6, 3))
            if has_grasp:
                _fill_gripper_pts(grasp_pts, rot, trans)

            # 3. ROS publish on the shape-lock rising edge
            is_locked = bool(self._tracker._shape_locked)
            if has_grasp and is_locked and not self._was_locked:
                self._publish(shape, rot, trans, shape_ls)
            self._was_locked = is_locked

            # 4. Hand the result to the render thread.  The centroid for the
            #    one-shot set_lookat is also taken from the fitted shape, not
            #    from the raw point cloud.
            centroid = trans if has_grasp \
                       else _shape_centroid(shape, self._tracker, shape_ls)
            with self._lock:
                self._out = {
                    "shape":     shape,
                    "shape_ls":  shape_ls,
                    "grasp_pts": grasp_pts,
                    "has_grasp": has_grasp,
                    "centroid":  centroid,
                }

    # ── ROS publish (worker thread; rclpy publishers are thread-safe) ─────────

    def _publish(self, shape, rot, trans, shape_ls):
        d_m, elev, bear, hand = _object_params(
            rot, trans, shape, self._tracker, shape_ls)

        # ── alpha: signed angle between the jaw separation line (closing
        #          axis) and the chessboard / table plane captured at
        #          startup.  alpha = arcsin(closing · n) ∈ [-π/2, +π/2].
        #          The base-roll joint is driven to (π/2 − alpha) so the
        #          forearm rotates to make the jaw line vertical relative
        #          to the table.
        closing   = rot[:, 1]
        alpha_rad = float(np.arcsin(
            np.clip(np.dot(closing, self._table_normal), -1.0, 1.0)))
        base_roll = float(np.pi / 2.0 - alpha_rad)

        obj = Float64MultiArray()
        obj.data = [1. if shape == "cylinder" else 0.,
                    d_m, 0., 0.05, 0., 0., 0.]
        self._node.object_pub.publish(obj)

        traj = JointTrajectory()
        traj.header.frame_id = 'world'
        traj.joint_names = _JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.time_from_start = Duration(sec=2, nanosec=0)
        pt.positions = [hand[0], hand[1], hand[2] - 0.05,
                        base_roll,
                        np.deg2rad(bear), np.deg2rad(elev)]
        traj.points.append(pt)
        self._node.traj_pub.publish(traj)

        pose = Int8()
        pose.data = 1
        self._node.pose_pub.publish(pose)

        print(f"[grasp]  *** LOCKED & PUBLISHED  trans={np.round(trans, 3)}  "
              f"elev={elev:+.1f}°  bear={bear:+.1f}°  "
              f"alpha={np.degrees(alpha_rad):+.1f}°  "
              f"base_roll={np.degrees(base_roll):+.1f}° ***")


# ══════════════════════════════════════════════════════════════════════════════
# ON-NEW-FRAME CALLBACK (main thread — render only)
# ══════════════════════════════════════════════════════════════════════════════

def _make_grasp_callback(worker: FitWorker):
    """
    Build a render-thread callback that does nothing but apply the worker's
    latest result to the Open3D scene.
    """

    # Persistent gripper LineSet — 6 pts × 5 lines, never re-topologised.
    grasp_pts = np.zeros((6, 3))
    grasp_ls  = o3d.geometry.LineSet()
    grasp_ls.points = o3d.utility.Vector3dVector(grasp_pts)
    grasp_ls.lines  = o3d.utility.Vector2iVector(_GRIPPER_LINES)
    grasp_ls.colors = o3d.utility.Vector3dVector([GRIPPER_COLOR] * len(_GRIPPER_LINES))

    state = {
        "shape_ls":    None,    # current shape wireframe handle in the scene
        "label":       None,    # last shape type drawn
        "grasp_added": False,
        "lookat_set":  False,
    }

    def _remove_overlays(vis):
        if state["shape_ls"] is not None:
            vis.remove_geometry(state["shape_ls"], reset_bounding_box=False)
            state["shape_ls"] = None
        if state["grasp_added"]:
            vis.remove_geometry(grasp_ls, reset_bounding_box=False)
            state["grasp_added"] = False
        state["label"] = None

    def _on_frame(obj_verts: np.ndarray, vis: o3d.visualization.Visualizer,
                  shape_hint: str | None = None):
        # ── Object lost — reset worker + drop overlays ───────────────────────
        if len(obj_verts) == 0:
            worker.reset()
            _remove_overlays(vis)
            return

        # ── Hand off to worker; pull latest result if any ────────────────────
        worker.push(obj_verts)
        result = worker.pop()
        if result is None:
            return

        # ── Update / add shape wireframe ─────────────────────────────────────
        new_shape    = result["shape"]
        new_shape_ls = result["shape_ls"]
        if new_shape != state["label"]:
            print(f"[shape_fit]  *** shape → {new_shape} ***")
            if state["shape_ls"] is not None:
                vis.remove_geometry(state["shape_ls"], reset_bounding_box=False)
            vis.add_geometry(new_shape_ls, reset_bounding_box=False)
            state["shape_ls"] = new_shape_ls
            state["label"]    = new_shape
        else:
            sls = state["shape_ls"]
            sls.points = new_shape_ls.points
            sls.lines  = new_shape_ls.lines
            sls.colors = new_shape_ls.colors
            vis.update_geometry(sls)

        if not state["lookat_set"] and result["centroid"] is not None:
            vis.get_view_control().set_lookat(result["centroid"].tolist())
            state["lookat_set"] = True

        # ── Update gripper lineset in place ──────────────────────────────────
        if result["has_grasp"]:
            grasp_pts[:] = result["grasp_pts"]
            grasp_ls.points = o3d.utility.Vector3dVector(grasp_pts)
            if not state["grasp_added"]:
                vis.add_geometry(grasp_ls, reset_bounding_box=False)
                state["grasp_added"] = True
            else:
                vis.update_geometry(grasp_ls)

    return _on_frame


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _ros_spin(node):
    rclpy.spin(node)


def run(board_cols=_BOARD_COLS, board_rows=_BOARD_ROWS):
    table_normal, _ = detect_table_plane(board_cols, board_rows)

    isolator = ObjectIsolator(min_points=50)
    isolator.start()
    print("isolator started...")
    isolator.ready.wait()
    print("opening viewer.\n")

    rclpy.init()
    node = CVPublisherNode()

    ros_thread = threading.Thread(target=_ros_spin, args=(node,), daemon=True)
    ros_thread.start()

    worker   = FitWorker(table_normal, node)
    on_frame = _make_grasp_callback(worker)

    try:
        show_isolated_pcd(
            isolator,
            title="Shape-based Grasp — Live",
            on_new_frame=on_frame,
            camera_up=(0, -1, 0),
        )
    finally:
        print("\n[test_lineset_live_shape] shutting down...")
        worker.stop()
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
