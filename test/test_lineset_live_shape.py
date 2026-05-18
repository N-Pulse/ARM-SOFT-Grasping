"""
test/test_lineset_live_shape.py
===============================
Live gripper-lineset overlay derived from the FITTED SHAPE geometry.

The grasp pose is a deterministic function of the cylinder / cuboid
parameters produced by shape_fitter.  Those parameters are already
EMA-smoothed and lock-tracked inside ShapeTracker — so the grasp pose
converges and locks on EXACTLY the same clock as the shape, with no
extra smoothing layer.

Pipeline (identical to test_shape_fit.py, with grasp drawing layered on top)
----------------------------------------------------------------------------
  1. Detect table plane via chessboard          (detect_table_plane)
  2. Start ObjectIsolator                       (capture.object_isolation)
  3. show_isolated_pcd(isolator, on_new_frame)  (helper.pcd_visualizer)
  4. Per-frame callback `_make_grasp_callback`:
        a. ShapeFitThread → fit_and_track → shape + smoothed parameters
        b. Update shape wireframe
        c. _grasp_from_shape → approach / closing vectors from smoothed params
        d. Update persistent gripper lineset in place
        e. Publish ROS2 messages on the shape-lock rising edge ONLY

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


# ══════════════════════════════════════════════════════════════════════════════
# Background fit thread
# ══════════════════════════════════════════════════════════════════════════════

class ShapeFitThread:
    """
    Run fit_and_track on a worker thread; never block the render loop.

    Single-slot input queue — the worker always processes the most recent
    frame and silently drops stale ones.
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
            if verts is None:
                break
            shape, ls = fit_and_track(verts, self._table_normal, self._tracker)
            if ls is not None:
                with self._lock:
                    self._out["shape"] = shape
                    self._out["ls"]    = ls

    def push(self, verts: np.ndarray):
        try:
            self._in.get_nowait()
        except queue.Empty:
            pass
        self._in.put(verts.copy())

    def pop(self):
        with self._lock:
            shape           = self._out["shape"]
            ls              = self._out["ls"]
            self._out["ls"] = None
        return shape, ls

    def stop(self):
        self._in.put(None)


# ══════════════════════════════════════════════════════════════════════════════
# Grasp geometry derived from the (already-smoothed) shape parameters
# ══════════════════════════════════════════════════════════════════════════════

def _grasp_from_shape(shape, tracker, shape_ls):
    """
    Compute (rot, trans) from the fitted shape.

    All inputs (tracker.axis / axis_pt / radius / h_ctr / height; cuboid
    wireframe vertices) are already EMA-smoothed by ShapeTracker, so the
    output pose converges and locks on the tracker's clock.

      rot[:, 0]  approach  — palm → fingertips
      rot[:, 1]  closing   — between the two fingers
      trans      palm centre
    """
    if shape == "cylinder":
        axis    = tracker.axis
        axis_pt = tracker.axis_pt
        if axis is None or axis_pt is None or tracker.h_ctr is None:
            return None, None

        trans = axis_pt + axis * tracker.h_ctr   # cylinder centroid

        # Approach: camera → centre, projected ⊥ axis (horizontal grip)
        to_obj   = trans / (np.linalg.norm(trans) + 1e-9)
        approach = to_obj - (to_obj @ axis) * axis
        nrm      = np.linalg.norm(approach)
        if nrm < 1e-6:
            ref      = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 \
                       else np.array([0., 1., 0.])
            approach = np.cross(axis, ref)
            approach /= np.linalg.norm(approach)
        else:
            approach /= nrm

        closing = np.cross(axis, approach)
        closing /= np.linalg.norm(closing)
        third   = np.cross(approach, closing)

    elif shape == "cuboid":
        verts = np.asarray(shape_ls.points)
        if len(verts) < 8:
            return None, None

        trans    = verts.mean(axis=0)
        approach = trans / (np.linalg.norm(trans) + 1e-9)

        # Bottom corners — pick footprint edge most ⊥ to approach for closing
        bot = verts[::2]
        e1  = bot[1] - bot[0];  e1 /= (np.linalg.norm(e1) + 1e-9)
        e2  = bot[2] - bot[1];  e2 /= (np.linalg.norm(e2) + 1e-9)
        closing = e2 if abs(approach @ e1) < abs(approach @ e2) else e1
        closing = closing - (closing @ approach) * approach
        nrm     = np.linalg.norm(closing)
        if nrm < 1e-6:
            closing = np.cross(approach, np.array([0., -1., 0.]))
            closing /= np.linalg.norm(closing)
        else:
            closing /= nrm

        third = np.cross(approach, closing)

    else:
        return None, None

    rot = np.column_stack([approach, closing, third])
    if np.linalg.det(rot) < 0:
        rot[:, 2] = -rot[:, 2]
    return rot, trans


def _fill_gripper_pts(pts: np.ndarray, rot, trans):
    """Write the 6 gripper-skeleton points into the preallocated buffer."""
    approach = rot[:, 0]
    closing  = rot[:, 1]
    half_w   = FINGER_DISTANCE / 2.0
    tip_c    = trans + approach * FINGER_LENGTH

    pts[0] = trans - approach * PALM_DEPTH
    pts[1] = trans
    pts[2] = trans + closing * half_w
    pts[3] = trans - closing * half_w
    pts[4] = tip_c + closing * half_w
    pts[5] = tip_c - closing * half_w


def _object_params(rot, trans, shape, tracker, shape_ls):
    """
    From the grasp pose, return (distance_m, elevation_deg, bearing_deg,
    hand_pos) in the simulation frame.

    Camera frame: X right, Y down, Z forward.
    Sim frame:    X forward, Y left, Z down.
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

    cam_pos  = trans - approach * (FINGER_LENGTH + obj_half_depth)
    hand_pos = np.array([cam_pos[2], -cam_pos[0], cam_pos[1]])
    return distance_m, elevation_deg, bearing_deg, hand_pos


# ══════════════════════════════════════════════════════════════════════════════
# ON-NEW-FRAME CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _make_grasp_callback(table_normal,
                         object_publisher,
                         trajectory_publisher,
                         pose_publisher):
    """
    Build a callback compatible with show_isolated_pcd that:
      · runs shape fitting on a background thread,
      · updates the shape wireframe,
      · updates the gripper lineset (in place, persistent geometry),
      · publishes the grasp to ROS once on the shape-lock rising edge.

    Convergence clock — the only EMA in play is ShapeTracker's, which
    governs both the shape parameters and the derived grasp pose.
    """
    tracker = ShapeTracker()
    fitter  = ShapeFitThread(table_normal, tracker)

    # Persistent gripper geometry — 6 pts × 5 lines, never re-topologised.
    grasp_pts = np.zeros((6, 3))
    grasp_ls  = o3d.geometry.LineSet()
    grasp_ls.points = o3d.utility.Vector3dVector(grasp_pts)
    grasp_ls.lines  = o3d.utility.Vector2iVector(_GRIPPER_LINES)
    grasp_ls.colors = o3d.utility.Vector3dVector([GRIPPER_COLOR] * len(_GRIPPER_LINES))

    state = {
        "shape_ls":    None,
        "label":       None,
        "grasp_added": False,
        "lookat_set":  False,
        "was_locked":  False,
    }

    # Reusable ROS message templates — joint_names never change.
    _joint_names = ['joint_base_x', 'joint_base_y', 'joint_base_z',
                    'joint_wrist_x', 'joint_wrist_y']

    def _publish(new_shape, rot, trans):
        d_m, elev, bear, hand = _object_params(
            rot, trans, new_shape, tracker, state["shape_ls"])

        obj = Float64MultiArray()
        obj.data = [1. if new_shape == "cylinder" else 0.,
                    d_m, 0., 0., 0., 0., 0.]
        object_publisher.publish(obj)

        traj = JointTrajectory()
        traj.joint_names = _joint_names
        pt = JointTrajectoryPoint()
        pt.time_from_start = Duration(sec=5, nanosec=0)
        pt.positions = [hand[0] - 0.12, hand[1], hand[2],
                        np.deg2rad(bear), np.deg2rad(elev)]
        traj.points.append(pt)
        trajectory_publisher.publish(traj)

        pose = Int8()
        pose.data = 1
        pose_publisher.publish(pose)

        print(f"[grasp]  *** LOCKED & PUBLISHED  trans={np.round(trans, 3)}  "
              f"elev={elev:+.1f}°  bear={bear:+.1f}° ***")

    def _remove_overlays(vis):
        if state["shape_ls"] is not None:
            vis.remove_geometry(state["shape_ls"], reset_bounding_box=False)
            state["shape_ls"] = None
        if state["grasp_added"]:
            vis.remove_geometry(grasp_ls, reset_bounding_box=False)
            state["grasp_added"] = False
        state["label"]      = None
        state["was_locked"] = False

    def _on_frame(obj_verts: np.ndarray, vis: o3d.visualization.Visualizer):
        # ── Object lost — reset everything ───────────────────────────────────
        if len(obj_verts) == 0:
            tracker.full_reset()
            _remove_overlays(vis)
            return

        # Feed the background fitter; pop the most recent result (if any)
        fitter.push(obj_verts)
        new_shape, new_shape_ls = fitter.pop()
        if new_shape_ls is None:
            return                          # no new fit → no work this tick

        # ── Update / add shape wireframe ─────────────────────────────────────
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

        if not state["lookat_set"]:
            vis.get_view_control().set_lookat(obj_verts.mean(axis=0).tolist())
            state["lookat_set"] = True

        # ── Derive grasp pose from the tracker's smoothed parameters ─────────
        rot, trans = _grasp_from_shape(new_shape, tracker, state["shape_ls"])
        if rot is None:
            return

        # ── Update persistent gripper lineset in place ───────────────────────
        _fill_gripper_pts(grasp_pts, rot, trans)
        grasp_ls.points = o3d.utility.Vector3dVector(grasp_pts)
        if not state["grasp_added"]:
            vis.add_geometry(grasp_ls, reset_bounding_box=False)
            state["grasp_added"] = True
        else:
            vis.update_geometry(grasp_ls)

        # ── Publish ROS only on the shape-lock rising edge ───────────────────
        is_locked = bool(tracker._shape_locked)
        if is_locked and not state["was_locked"]:
            _publish(new_shape, rot, trans)
        state["was_locked"] = is_locked

    _on_frame.fitter = fitter
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
    print("Waiting for YOLO to load...")
    isolator.ready.wait()
    print("YOLO ready — opening viewer.\n")

    rclpy.init()
    node = Node('CV_publisher_node')
    object_pub     = node.create_publisher(Float64MultiArray, '/cv/model', 10)
    trajectory_pub = node.create_publisher(
        JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
    pose_pub       = node.create_publisher(Int8, '/pose_goals', 10)

    ros_thread = threading.Thread(target=_ros_spin, args=(node,), daemon=True)
    ros_thread.start()

    on_frame = _make_grasp_callback(table_normal,
                                    object_pub, trajectory_pub, pose_pub)

    try:
        show_isolated_pcd(
            isolator,
            title="Shape-based Grasp — Live",
            on_new_frame=on_frame,
            camera_up=(0, -1, 0),
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
