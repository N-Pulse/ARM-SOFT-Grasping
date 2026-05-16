"""
test_shape_fit.py
=================
Curvature-based primitive fitting with table-plane-constrained axis snapping.

Chessboard specs (from calibration target config)
--------------------------------------------------
  Board size   : 200 × 150 mm
  Rows         : 8   →  inner corners = 7
  Columns      : 11  →  inner corners = 10
  Square width : 15 mm
  → board_shape = (10, 7) for cv2.findChessboardCorners

Pipeline
--------
  INIT
    detect_table_plane()
      · OpenCV finds the 10×7 inner corners in the colour frame
      · Each corner is back-projected to 3D via depth + RealSense intrinsics
      · SVD plane fit on the 3D corners → table_normal
      · Residual check: mean error must be < 5 mm (one third of a square)

  PER FRAME  — three explicit stages:

    ① CLASSIFY  (curvature)
       · Per-point quadratic patch fit → κ₁ ≤ κ₂
       · Median over patch → "cylinder" or "cuboid"
       · Consensus zero-curvature eigenvector → raw axis direction

    ② CONFIRM AXIS  (table constraint)
       · Raw axis is noisy (a few degrees error from normal estimation)
       · Physical constraint: the cylinder axis is either
           parallel to table_normal   (object standing upright)
           perpendicular to table_normal (object lying on its side)
       · Snap to whichever is closer (threshold 45°)
       · This gives a clean, physically-grounded axis for the next step

    ③ BEST FIT  (least-squares cylinder to the confirmed axis)
       · Project all PCD points onto the plane ⊥ to the confirmed axis
       · Fit a 2D circle (linearised least squares)
         → axis position (centre of circle in 3D) + radius
       · This minimises the mean squared radial error: Σ(dist_to_axis - r)²
       · Height: 1st/99th percentile of axial projections
                 refined by end-cap face detection if top/bottom are visible
       · Build Open3D LineSet at the fitted (axis, center, r, height)

Why axis-first, then fit?
--------------------------
  If the axis direction has even 3° of error, the projected circle is
  elliptical rather than circular, and the circle fit centre shifts
  significantly (especially for large radii).  Snapping first gives the
  circle fit a geometrically clean axis, producing the smallest possible
  radial residuals.

Wireframe colours
-----------------
  cylinder → orange   [1.0, 0.5, 0.0]
  cuboid   → lavender [0.8, 0.6, 1.0]

Usage
-----
  python test_shape_fit.py [--debug]
  python test_shape_fit.py --board-cols 10 --board-rows 7   # (default)
"""

import sys
import os
import argparse

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from capture.object_isolation import ObjectIsolator
from helper.pcd_visualizer import show_isolated_pcd


# ── Colours ────────────────────────────────────────────────────────────────────
_COLOR = {
    # "sphere":   [0.2, 0.8, 1.0],   # disabled — see module docstring
    "cylinder": [1.0, 0.5, 0.0],
    "cuboid":   [0.8, 0.6, 1.0],
}

# ── Chessboard defaults (match your calibration target) ───────────────────────
# Rows=8, Cols=11 on the printed board → inner corners = (10, 7)
_BOARD_INNER_COLS = 10          # columns - 1
_BOARD_INNER_ROWS = 7           # rows    - 1
_SQUARE_SIZE_M    = 0.015       # 15 mm

# ── Fitting tuning ─────────────────────────────────────────────────────────────
_KNN_NORMAL       = 30
_KNN_CURV         = 25
_SUBSAMPLE        = 600
_FLAT_THRESH      = 2.0         # |κ| < this (m⁻¹) → zero curvature
_ANISO_RATIO      = 5.0         # |κ_max/κ_min| > this → cylinder
_R_MIN            = 0.005       # m
_R_MAX            = 2.0         # m
_CAP_NORMAL_DOT   = 0.7         # |n·axis| > this → end-cap point
_CAP_MIN_PTS      = 10
_SNAP_THRESH_DEG  = 45.0        # curvature-axis fallback: angle below which → vertical
_MAX_RADIAL_ERR   = 0.015       # m — if mean radial error > this → fallback cuboid

# Aspect ratio thresholds for vertical / horizontal decision
# aspect = extent_along_table_normal / max_extent_in_table_plane
# 250 mm bottle (r≈30 mm) seen from the side : aspect ≈ 250/60  ≈ 4.2  → vertical
# 120 mm can (r≈35 mm) lying down            : aspect ≈  70/120 ≈ 0.58 → horizontal
_ASPECT_VERTICAL   = 1.5    # aspect > this → definitely vertical
_ASPECT_HORIZONTAL = 0.85   # aspect < this → definitely horizontal
                             # between the two → curvature axis angle as fallback


# ══════════════════════════════════════════════════════════════════════════════
# INIT — table plane from chessboard
# ══════════════════════════════════════════════════════════════════════════════

def detect_table_plane(board_cols=_BOARD_INNER_COLS,
                       board_rows=_BOARD_INNER_ROWS):
    """
    Stream frames from the RealSense until the chessboard is visible and a
    clean plane fit is obtained.

    Returns
    -------
    table_normal : (3,) unit vector perpendicular to table, toward camera
    table_d      : float  plane offset (normal·x + d = 0)

    Blocks until success or ESC is pressed (returns None, None on abort).
    """
    board_shape = (board_cols, board_rows)
    criteria    = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    pipe   = rs.pipeline()
    cfg    = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    align  = rs.align(rs.stream.color)

    profile     = pipe.start(cfg)
    intr        = (profile.get_stream(rs.stream.color)
                          .as_video_stream_profile()
                          .get_intrinsics())
    depth_scale = (profile.get_device()
                          .first_depth_sensor()
                          .get_depth_scale())
    fx, fy      = intr.fx, intr.fy
    cx, cy      = intr.ppx, intr.ppy

    print(f"\n[table]  Chessboard: inner corners {board_cols}×{board_rows}, "
          f"square {_SQUARE_SIZE_M*1000:.0f} mm")
    print("[table]  Point the camera at the chessboard on the table.")
    print("[table]  Press ESC to skip table detection.\n")

    try:
        while True:
            frames      = pipe.wait_for_frames()
            aligned     = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asarray(color_frame.get_data())
            depth_img = np.asarray(depth_frame.get_data()).astype(np.float32) \
                        * depth_scale                       # metres

            gray         = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, board_shape, None)

            # ── Live preview ─────────────────────────────────────────────────
            display = color_img.copy()
            if found:
                cv2.drawChessboardCorners(display, board_shape, corners, True)
            cv2.putText(display,
                        "FOUND — hold still" if found else "Searching for chessboard …",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 220, 0) if found else (0, 80, 255), 2)
            cv2.imshow("Table plane detection  [ESC to skip]", display)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                print("[table]  Skipped — running without table constraint.\n")
                return None, None

            if not found:
                continue

            # ── Sub-pixel refinement ─────────────────────────────────────────
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # ── Back-project corners to 3D ────────────────────────────────────
            pts_3d = []
            for (u, v) in corners.reshape(-1, 2):
                ui, vi = int(round(u)), int(round(v))
                if not (0 <= ui < depth_img.shape[1] and
                        0 <= vi < depth_img.shape[0]):
                    continue
                z = float(depth_img[vi, ui])
                if z < 0.05 or z > 3.0:        # 5 cm – 3 m sanity window
                    continue
                pts_3d.append([(u - cx) * z / fx,
                                (v - cy) * z / fy,
                                z])

            if len(pts_3d) < 6:
                print("[table]  Too many missing depth values — retry.")
                continue

            pts_3d   = np.array(pts_3d)
            centroid = pts_3d.mean(axis=0)

            # ── SVD plane fit ─────────────────────────────────────────────────
            _, _, Vt = np.linalg.svd(pts_3d - centroid)
            normal   = Vt[-1]
            normal  /= np.linalg.norm(normal)
            d        = -float(normal @ centroid)

            # Orient normal toward the camera (camera at origin)
            if float(normal @ centroid) < 0:
                normal, d = -normal, -d

            # ── Quality check: residuals must be < 5 mm ───────────────────────
            # (one-third of the 15 mm square — tight enough to catch bad depth)
            mean_res = float(np.abs((pts_3d - centroid) @ normal).mean())
            if mean_res > 0.005:
                print(f"[table]  Residual {mean_res*1000:.1f} mm > 5 mm — retry.")
                continue

            print(f"[table]  ✓  normal={np.round(normal, 3)}  "
                  f"residual={mean_res*1000:.2f} mm\n")
            cv2.waitKey(400)
            cv2.destroyAllWindows()
            return normal, d

    finally:
        pipe.stop()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE ① — Local curvature → classify → raw axis
# ══════════════════════════════════════════════════════════════════════════════

def _fit_local_curvature(p, n, neighbours):
    """
    Quadratic patch h = a·u² + b·u·v + c·v² in the tangent frame of (p, n).
    Returns (κ₁, κ₂, zero_axis_3d) or None on degenerate input.
    κ₁ ≤ κ₂;  zero_axis is the direction of minimal curvature.
    """
    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    t1  = np.cross(n, ref);  t1 /= np.linalg.norm(t1)
    t2  = np.cross(n, t1);   t2 /= np.linalg.norm(t2)

    d = neighbours - p
    u = d @ t1;  v = d @ t2;  h = d @ n

    A = np.column_stack([u ** 2, u * v, v ** 2])
    if np.linalg.matrix_rank(A) < 3:
        return None

    (a, b, c), *_ = np.linalg.lstsq(A, h, rcond=None)

    II           = np.array([[2*a, b], [b, 2*c]])
    evals, evecs = np.linalg.eigh(II)          # ascending κ₁ ≤ κ₂
    ev_min       = evecs[:, 0]
    zero_axis    = ev_min[0] * t1 + ev_min[1] * t2

    return float(evals[0]), float(evals[1]), zero_axis


def _aggregate_curvatures(pts, normals):
    tree = cKDTree(pts)
    k1s, k2s, axs = [], [], []

    for i in range(len(pts)):
        _, idx = tree.query(pts[i], k=_KNN_CURV + 1)
        r = _fit_local_curvature(pts[i], normals[i], pts[idx[1:]])
        if r is None:
            continue
        k1s.append(r[0]);  k2s.append(r[1]);  axs.append(r[2])

    if not k1s:
        return 0.0, 0.0, None

    kappa1 = float(np.median(k1s))
    kappa2 = float(np.median(k2s))

    axes  = np.array(axs)
    signs = np.sign(axes @ axes[0]);  signs[signs == 0] = 1
    axes *= signs[:, None]
    m    = axes.mean(axis=0);  nrm = np.linalg.norm(m)
    axis = m / nrm if nrm > 1e-6 else None

    return kappa1, kappa2, axis


def _classify(k1, k2):
    # "sphere" intentionally disabled
    return "cuboid" if abs(k2) < _FLAT_THRESH else "cylinder"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE ② — Confirm axis with table constraint
# ══════════════════════════════════════════════════════════════════════════════

def _confirm_axis(raw_axis, table_normal, pts):
    """
    Decide whether the cylinder is vertical or horizontal, then snap the axis.

    Decision strategy — two signals, primary wins:
    ────────────────────────────────────────────────────────────────────────
    PRIMARY  Point-cloud aspect ratio  (geometric, robust to curvature noise)
      · extent_up   = PCD range along table_normal
      · extent_wide = largest PCD range in the table plane
      · aspect = extent_up / extent_wide

      aspect > _ASPECT_VERTICAL   → vertical   (tall object, e.g. bottle)
      aspect < _ASPECT_HORIZONTAL → horizontal  (wide object, e.g. can lying)

    FALLBACK  Curvature axis angle  (used only when aspect is ambiguous)
      · |raw_axis · table_normal| ≥ cos(_SNAP_THRESH_DEG) → vertical
      · otherwise                                          → horizontal
    ────────────────────────────────────────────────────────────────────────

    For horizontal cylinders the in-plane direction is kept from raw_axis,
    because curvature correctly tells us which way the object is oriented
    even when the tilt estimate is noisy.

    Returns (confirmed_axis, "vertical" | "horizontal").
    """
    # ── Primary: aspect ratio of the PCD bounding box ────────────────────────
    centroid = pts.mean(axis=0)
    d        = pts - centroid

    # Height: extent along the table normal direction
    proj_up    = d @ table_normal
    extent_up  = float(proj_up.max() - proj_up.min())

    # Width: build two orthonormal vectors in the table plane and take max range
    ref  = np.array([1., 0., 0.]) if abs(table_normal[0]) < 0.9 \
           else np.array([0., 1., 0.])
    p1   = np.cross(table_normal, ref);  p1 /= np.linalg.norm(p1)
    p2   = np.cross(table_normal, p1);   p2 /= np.linalg.norm(p2)
    u    = d @ p1;  v = d @ p2
    extent_wide = float(max(u.max() - u.min(), v.max() - v.min()))

    aspect = extent_up / (extent_wide + 1e-6)
    print(f"[shape_fit]  extent_up={extent_up*1000:.0f} mm  "
          f"extent_wide={extent_wide*1000:.0f} mm  "
          f"aspect={aspect:.2f}")

    if aspect > _ASPECT_VERTICAL:
        orientation = "vertical"
    elif aspect < _ASPECT_HORIZONTAL:
        orientation = "horizontal"
    else:
        # ── Fallback: curvature axis angle ────────────────────────────────────
        cos_a     = abs(float(raw_axis @ table_normal))
        threshold = float(np.cos(np.radians(_SNAP_THRESH_DEG)))
        orientation = "vertical" if cos_a >= threshold else "horizontal"
        print(f"[shape_fit]  aspect ambiguous → curvature fallback → {orientation}")

    # ── Snap axis ─────────────────────────────────────────────────────────────
    if orientation == "vertical":
        snapped = table_normal.copy()
    else:
        # Project raw_axis onto the table plane (removes out-of-plane tilt noise)
        snapped = raw_axis - (raw_axis @ table_normal) * table_normal
        nrm     = np.linalg.norm(snapped)
        if nrm < 1e-6:
            return raw_axis, "unknown"
        snapped = snapped / nrm

    # Preserve sign (same hemisphere as raw_axis)
    if float(snapped @ raw_axis) < 0:
        snapped = -snapped

    return snapped, orientation


# ══════════════════════════════════════════════════════════════════════════════
# STAGE ③ — Best fit to the confirmed axis
# ══════════════════════════════════════════════════════════════════════════════

def _best_fit_cylinder(pts, normals, axis):
    """
    Given a confirmed axis direction, find the cylinder that best fits the PCD.

    Method
    ------
    Project every point onto the plane ⊥ to *axis* (dropping the axial
    component — all cross-sections of a true cylinder are identical circles).
    Fit a 2D circle by linearised least squares:

        (u - cu)² + (v - cv)² = r²
        → -2cu·u - 2cv·v + (cu²+cv²-r²) = -(u²+v²)

    This minimises  Σ (radial_distance_to_axis - r)²  over all points,
    which is the correct objective for cylinder fitting.

    The axis passes through the 3D point  centroid + cu·e1 + cv·e2
    (the circle centre lifted back into world space).

    Returns
    -------
    axis_pt  : (3,)  a point on the cylinder axis
    radius   : float
    mean_err : float  mean radial residual (m) — used for quality check
    or (None, None, inf) on failure.
    """
    centroid = pts.mean(axis=0)

    # 2D orthonormal frame in the cross-section plane
    ref = np.array([0., 0., 1.]) if abs(axis[2]) < 0.9 else np.array([1., 0., 0.])
    e1  = np.cross(axis, ref);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(axis, e1);   e2 /= np.linalg.norm(e2)

    d = pts - centroid
    u = d @ e1
    v = d @ e2                  # axial component dropped — intentional

    A  = np.column_stack([-2*u, -2*v, np.ones(len(u))])
    b  = -(u**2 + v**2)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    cu, cv, dval = x

    r_sq = cu**2 + cv**2 - dval
    if r_sq <= 0:
        return None, None, np.inf

    r        = float(np.sqrt(r_sq))
    axis_pt  = centroid + cu * e1 + cv * e2

    # Radial residual: how well the PCD lies on this cylinder
    along      = (pts - axis_pt) @ axis
    on_axis    = axis_pt + np.outer(along, axis)
    radial     = np.linalg.norm(pts - on_axis, axis=1)
    mean_err   = float(np.mean(np.abs(radial - r)))

    return axis_pt, r, mean_err


def _estimate_height(pts, normals, axis, centroid):
    """
    Axial extent via 1st/99th percentile + end-cap face refinement.
    """
    proj  = (pts - centroid) @ axis
    h_min = float(np.percentile(proj, 1))
    h_max = float(np.percentile(proj, 99))

    cap_mask = np.abs(normals @ axis) > _CAP_NORMAL_DOT
    if cap_mask.sum() >= _CAP_MIN_PTS:
        cp    = proj[cap_mask]
        span  = h_max - h_min
        cmin  = float(cp.min());  cmax = float(cp.max())
        if cmin < h_min + 0.3 * span:
            h_min = cmin
            print(f"[shape_fit]  bottom cap → h_min = {h_min:.3f} m")
        if cmax > h_max - 0.3 * span:
            h_max = cmax
            print(f"[shape_fit]  top cap    → h_max = {h_max:.3f} m")

    return h_min, h_max


# ══════════════════════════════════════════════════════════════════════════════
# Wireframe builders
# ══════════════════════════════════════════════════════════════════════════════

def _build_cylinder(axis, axis_pt, r, h_min, h_max):
    """
    Build a cylinder LineSet at the given axis / center / radius / height.
    axis_pt is the axis at h=0 relative to the point-cloud centroid;
    the full 3D center is axis_pt + axis * mid.
    """
    height = float(np.clip(h_max - h_min, 0.005, 5.0))
    center = axis_pt + axis * (h_min + h_max) / 2.0

    z   = np.array([0., 0., 1.])
    v   = np.cross(z, axis)
    s   = np.linalg.norm(v)
    c   = float(np.dot(z, axis))
    if s < 1e-6:
        R = np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    else:
        vx = np.array([[0,    -v[2],  v[1]],
                       [v[2],  0,    -v[0]],
                       [-v[1], v[0],  0   ]])
        R  = np.eye(3) + vx + vx @ vx * (1.0 - c) / (s**2)

    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=r, height=height, resolution=20
    )
    mesh.rotate(R, center=np.zeros(3))
    mesh.translate(center)
    ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    ls.paint_uniform_color(_COLOR["cylinder"])
    return ls


def _build_cuboid(pts):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    plane, inlier_idx = pcd.segment_plane(
        distance_threshold=0.005, ransac_n=3, num_iterations=200
    )
    n   = np.array(plane[:3]);  n /= np.linalg.norm(n) + 1e-9
    inl = pts[np.array(inlier_idx)]

    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    t1  = np.cross(n, ref);  t1 /= np.linalg.norm(t1)
    t2  = np.cross(n, t1);   t2 /= np.linalg.norm(t2)

    u  = inl @ t1;  v = inl @ t2;  w = inl @ n
    du = u.max()-u.min();  dv = v.max()-v.min()
    dw = max(w.max()-w.min(), 0.005)

    mesh = o3d.geometry.TriangleMesh.create_box(width=du, height=dv, depth=dw)
    mesh.translate([-du/2, -dv/2, -dw/2])
    R = np.column_stack([t1, t2, n])
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    mesh.rotate(R, center=np.zeros(3))
    ctr = np.array([
        (u.max()+u.min())/2*t1[i] +
        (v.max()+v.min())/2*t2[i] +
        (w.max()+w.min())/2*n[i]
        for i in range(3)
    ])
    mesh.translate(ctr)
    ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    ls.paint_uniform_color(_COLOR["cuboid"])
    return ls


# ══════════════════════════════════════════════════════════════════════════════
# Top-level: per-frame fitting
# ══════════════════════════════════════════════════════════════════════════════

def fit_shape(pts: np.ndarray, table_normal: np.ndarray = None):
    """
    Three-stage pipeline: classify → confirm axis → best fit.

    Parameters
    ----------
    pts          : (N, 3) isolated object points
    table_normal : (3,) unit vector ⊥ to table (from detect_table_plane).
                   Pass None to skip axis snapping.

    Returns
    -------
    (shape_name, LineSet)  or  (None, None)
    """
    if len(pts) < 50:
        return None, None

    # ── Normal estimation ─────────────────────────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=_KNN_NORMAL)
    )
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0., 0., 0.])
    )
    normals = np.asarray(pcd.normals)

    N = len(pts)
    if N > _SUBSAMPLE:
        idx     = np.random.choice(N, _SUBSAMPLE, replace=False)
        s_pts   = pts[idx];  s_norms = normals[idx]
    else:
        s_pts, s_norms = pts, normals

    # ── STAGE ①  Classify + raw axis ─────────────────────────────────────────
    k1, k2, raw_axis = _aggregate_curvatures(s_pts, s_norms)
    shape            = _classify(k1, k2)
    print(f"[shape_fit]  κ₁={k1:+.2f}  κ₂={k2:+.2f}  → {shape}")

    if shape == "cylinder" and raw_axis is None:
        print("[shape_fit]  axis extraction failed → cuboid")
        shape = "cuboid"

    if shape != "cylinder":
        return shape, _build_cuboid(pts)

    # ── STAGE ②  Confirm axis with table constraint ───────────────────────────
    if table_normal is not None:
        axis, orientation = _confirm_axis(raw_axis, table_normal, pts)
        print(f"[shape_fit]  axis confirmed: {orientation}  {np.round(axis, 3)}")
    else:
        axis = raw_axis

    # ── STAGE ③  Best fit to the confirmed axis ───────────────────────────────
    axis_pt, r, mean_err = _best_fit_cylinder(pts, normals, axis)

    if axis_pt is None:
        print("[shape_fit]  circle fit failed → cuboid")
        return "cuboid", _build_cuboid(pts)

    r = float(np.clip(r, _R_MIN, _R_MAX))

    if mean_err > _MAX_RADIAL_ERR:
        print(f"[shape_fit]  radial error {mean_err*1000:.1f} mm > "
              f"{_MAX_RADIAL_ERR*1000:.0f} mm → cuboid")
        return "cuboid", _build_cuboid(pts)

    centroid         = pts.mean(axis=0)
    h_min, h_max     = _estimate_height(pts, normals, axis, centroid)

    print(f"[shape_fit]  ✓ cylinder  r={r*1000:.1f} mm  "
          f"h={( h_max-h_min)*1000:.1f} mm  "
          f"err={mean_err*1000:.2f} mm")

    ls = _build_cylinder(axis, axis_pt, r, h_min, h_max)
    return "cylinder", ls


# ══════════════════════════════════════════════════════════════════════════════
# Visualizer callback
# ══════════════════════════════════════════════════════════════════════════════

def _make_overlay_callback(table_normal):
    state = {"ls": None, "label": None}

    def _on_frame(obj_verts: np.ndarray, vis: o3d.visualization.Visualizer):
        shape, new_ls = fit_shape(obj_verts, table_normal=table_normal)
        if new_ls is None:
            return

        if shape != state["label"]:
            print(f"[shape_fit]  *** shape → {shape} ***")

        if state["ls"] is None:
            vis.add_geometry(new_ls)
            state["ls"] = new_ls
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

def run(board_cols=_BOARD_INNER_COLS, board_rows=_BOARD_INNER_ROWS,
        debug=False):

    table_normal, _ = detect_table_plane(board_cols=board_cols,
                                         board_rows=board_rows)

    isolator = ObjectIsolator(min_points=50)
    isolator.start()
    print("Waiting for camera …")
    isolator.ready.wait()
    print("Camera ready — opening visualizer.\n")

    try:
        show_isolated_pcd(
            isolator,
            on_new_frame=_make_overlay_callback(table_normal),
            debug=debug,
        )
    finally:
        isolator.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--board-cols", type=int, default=_BOARD_INNER_COLS,
                   help=f"Inner corner columns (default {_BOARD_INNER_COLS})")
    p.add_argument("--board-rows", type=int, default=_BOARD_INNER_ROWS,
                   help=f"Inner corner rows (default {_BOARD_INNER_ROWS})")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    run(board_cols=args.board_cols, board_rows=args.board_rows, debug=args.debug)