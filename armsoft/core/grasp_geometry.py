"""
armsoft/grasp_geometry.py
==========================
Pure-numpy grasp mathematics: fitted primitive in, gripper pose out.

Everything here derives from the FITTED SHAPE — the wireframe vertices produced
by :func:`armsoft.shape_fitter.fit_once` — never from the raw point cloud, so
the result is as smooth as the fit rather than as noisy as the sensor.

The only Open3D contact is reading ``line_set.points``; pass a plain (N, 3)
array instead and these functions work with no Open3D at all
(see :func:`wireframe_vertices`).
"""

from __future__ import annotations

import numpy as np

# ── Gripper geometry (metres) ─────────────────────────────────────────────────
FINGER_LENGTH    = 0.085
PALM_DEPTH       = 0.06
FINGER_CLEARANCE = 0.008   # gap between fingertip and object surface

#: Line topology of the 6-point gripper skeleton.
GRIPPER_LINES = [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5]]

#: Joint names of the published arm trajectory, in `joint_positions` order.
JOINT_NAMES = ['joint_base_x', 'joint_base_y', 'joint_base_z',
               'joint_base_roll', 'joint_wrist_x', 'joint_wrist_y']


def wireframe_vertices(shape_ls) -> np.ndarray:
    """Accept an Open3D LineSet or a plain (N, 3) array and return (N, 3)."""
    if shape_ls is None:
        return np.zeros((0, 3))
    if isinstance(shape_ls, np.ndarray):
        return shape_ls
    return np.asarray(shape_ls.points)


def shape_centroid(shape_ls) -> np.ndarray | None:
    """Geometric centroid of the fitted shape (mean of wireframe vertices)."""
    verts = wireframe_vertices(shape_ls)
    if len(verts) == 0:
        return None
    return verts.mean(axis=0)


def grasp_from_shape(shape: str, table_normal, shape_ls):
    """
    (rot, trans, half_width) from the fitted shape.

      rot[:, 0] approach   palm → fingertips
      rot[:, 1] closing    between the two fingertips
      rot[:, 2] cross(approach, closing)
      trans     tool centre point = fitted-shape centroid

    The approach axis is the camera→centroid direction projected onto the table
    plane, so the gripper always comes in horizontally.

    Returns ``(None, None, None)`` when no valid grasp can be derived.
    """
    trans = shape_centroid(shape_ls)
    if trans is None:
        return None, None, None
    verts = wireframe_vertices(shape_ls)

    # ── vertical (table) axis ────────────────────────────────────────────────
    if shape == "cylinder":
        if table_normal is None:
            return None, None, None
        axis = np.asarray(table_normal, float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
    elif shape == "cuboid":
        if len(verts) < 8:
            return None, None, None
        axis = (verts[1::2] - verts[::2]).mean(axis=0)
        n = np.linalg.norm(axis)
        if n < 1e-9:
            return None, None, None
        axis = axis / n
    else:
        return None, None, None

    # ── approach: (camera → centroid) projected onto the table plane ─────────
    to_obj   = trans / (np.linalg.norm(trans) + 1e-9)
    approach = to_obj - (to_obj @ axis) * axis
    nrm = np.linalg.norm(approach)
    if nrm < 1e-6:
        ref = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 else np.array([0., 1., 0.])
        approach = np.cross(axis, ref)
        approach /= np.linalg.norm(approach)
    else:
        approach /= nrm

    # ── closing: in the table plane, perpendicular to approach ──────────────
    if shape == "cylinder":
        closing = np.cross(axis, approach)
        closing /= np.linalg.norm(closing)
    else:
        bot = verts[::2]
        e1 = bot[1] - bot[0]; e1 /= (np.linalg.norm(e1) + 1e-9)
        e2 = bot[2] - bot[0]; e2 /= (np.linalg.norm(e2) + 1e-9)
        closing = e2 if abs(approach @ e1) < abs(approach @ e2) else e1
        closing = closing - (closing @ approach) * approach
        nrm = np.linalg.norm(closing)
        if nrm < 1e-6:
            closing = np.cross(axis, approach)
            closing /= np.linalg.norm(closing)
        else:
            closing /= nrm

    third = np.cross(approach, closing)
    rot = np.column_stack([approach, closing, third])
    if np.linalg.det(rot) < 0:
        rot[:, 2] = -rot[:, 2]

    proj   = verts @ closing
    half_w = (proj.max() - proj.min()) / 2.0 + FINGER_CLEARANCE
    return rot, trans, float(half_w)


def gripper_skeleton(rot, trans, half_w: float) -> np.ndarray:
    """
    6-point gripper skeleton, same ordering as the live viewer::

        palm_back ─── palm ─┬─── left_root ──── left_tip
                            └─── right_root ─── right_tip
    """
    approach, closing = rot[:, 0], rot[:, 1]
    palm      = trans - approach * FINGER_LENGTH
    palm_back = palm - approach * PALM_DEPTH
    pts = np.zeros((6, 3))
    pts[0] = palm_back
    pts[1] = palm
    pts[2] = palm + closing * half_w
    pts[3] = palm - closing * half_w
    pts[4] = trans + closing * half_w
    pts[5] = trans - closing * half_w
    return pts


def object_params(rot, trans):
    """
    (distance_m, elevation_deg, bearing_deg, hand_pos) in the simulation frame
    (X forward, Y left, Z down).
    """
    distance_m = float(np.linalg.norm(trans))
    approach   = rot[:, 0]
    elevation_deg = float(np.degrees(np.arcsin(np.clip(-approach[1], -1.0, 1.0))))
    bearing_deg   = float(np.degrees(np.arctan2(approach[0], approach[2])))
    cam_pos  = trans - approach * FINGER_LENGTH
    hand_pos = np.array([cam_pos[2], -cam_pos[0], cam_pos[1]])
    return distance_m, elevation_deg, bearing_deg, hand_pos


def base_roll(rot, table_normal) -> float:
    """
    Base-roll joint target = π/2 − alpha, where alpha is the signed angle
    between the jaw closing axis and the table plane.
    """
    if table_normal is None:
        return float(np.pi / 2.0)
    n = np.asarray(table_normal, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    alpha = float(np.arcsin(np.clip(np.dot(rot[:, 1], n), -1.0, 1.0)))
    return float(np.pi / 2.0 - alpha)


def joint_positions(rot, trans, table_normal) -> list[float]:
    """
    The six joint targets for the arm, in the order given by
    :data:`JOINT_NAMES`.
    """
    _d, elev, bear, hand = object_params(rot, trans)
    return [float(hand[0] + 0.012), float(hand[1]), float(hand[2] - 0.05),
            base_roll(rot, table_normal),
            float(np.deg2rad(bear)), float(np.deg2rad(elev))]
