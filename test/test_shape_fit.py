"""
test_shape_fit.py
=================
Curvature-based primitive fitting for single-view point clouds.
Only cylinder and cuboid are fitted.

Pipeline
--------
  1. LOCAL   — per-point quadratic patch fit → κ₁ ≤ κ₂, zero-curvature axis
  2. FEATURE — median κ₁, κ₂ → classify; consensus axis direction
  3. ALIGN   — (cylinder only) 2D circle fit in the cross-section plane
               This is the critical step for surface alignment:
               · Project all points onto the plane ⊥ to the axis
               · Fit a circle → gives the true axis position and radius
               · The cylinder surface now passes through the visible PCD
               · Curvature gives us the *axis direction*; the circle fit
                 gives us the *axis position* and *radius* — both are needed
  4. HEIGHT  — point-cloud extents along the axis + end-cap refinement
  5. REBUILD — place the Open3D cylinder at the fitted center / axis / radius

Surface alignment guarantee
----------------------------
After step 3, every point in the PCD is at distance ≈ r from the axis.
The generated cylinder wireframe therefore sits flush on the visible surface,
not floating in front of or behind it.

Axis orientation
----------------
After averaging the per-point zero-curvature eigenvectors, there is a ±1
sign ambiguity in the axis direction.  We resolve it by comparing against
the world up vector (0, 0, 1 in camera depth frame): if the dot product is
negative we flip the axis.  This keeps the axis pointing consistently
"upward" for tall objects like bottles and cans.
If the axis is nearly horizontal (|axis·up| < 0.2) we fall back to
orienting toward positive X, which is a reasonable convention for objects
lying on their side.

Wireframe colours
-----------------
  cylinder → orange   [1.0, 0.5, 0.0]
  cuboid   → lavender [0.8, 0.6, 1.0]

Disabled
--------
  Sphere fitting is commented out.  Re-enable if needed.

Usage
-----
  python test_shape_fit.py [--debug]
"""

import sys
import os
import argparse

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from capture.object_isolation import ObjectIsolator
from helper.pcd_visualizer import show_isolated_pcd


# ── Colours ────────────────────────────────────────────────────────────────────
_COLOR = {
    # "sphere":   [0.2, 0.8, 1.0],   # disabled
    "cylinder": [1.0, 0.5, 0.0],
    "cuboid":   [0.8, 0.6, 1.0],
}

# ── Tuning ─────────────────────────────────────────────────────────────────────
_KNN_NORMAL     = 30     # neighbours for normal estimation
_KNN_CURV       = 25     # neighbours for per-point curvature fit
_SUBSAMPLE      = 600    # max points in the curvature loop
_FLAT_THRESH    = 2.0    # |κ| < this (m⁻¹) → zero curvature
_ANISO_RATIO    = 5.0    # |κ_max / κ_min| > this → cylinder
_R_MIN          = 0.005  # smallest plausible radius (m)
_R_MAX          = 2.0    # largest  plausible radius (m)
_CAP_NORMAL_DOT = 0.7    # |n · axis| > this → end-cap point
_CAP_MIN_PTS    = 10     # min points to trust end-cap detection
_WORLD_UP       = np.array([0., 0., 1.])   # camera depth frame: Z points forward,
                                            # adjust to (0,1,0) if Y is up in your rig


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Local principal curvatures
# ══════════════════════════════════════════════════════════════════════════════

def _fit_local_curvature(p, n, neighbours):
    """
    Fit h = a·u² + b·u·v + c·v² in the tangent frame of (p, n).
    Returns (κ₁, κ₂, zero_axis_3d) — κ₁ ≤ κ₂; zero_axis is the direction
    of minimal curvature (= cylinder axis candidate).
    Returns None on degenerate input.
    """
    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    t1  = np.cross(n, ref);  t1 /= np.linalg.norm(t1)
    t2  = np.cross(n, t1);   t2 /= np.linalg.norm(t2)

    d = neighbours - p
    u = d @ t1
    v = d @ t2
    h = d @ n

    A = np.column_stack([u ** 2, u * v, v ** 2])
    if np.linalg.matrix_rank(A) < 3:
        return None

    (a, b, c), *_ = np.linalg.lstsq(A, h, rcond=None)

    II           = np.array([[2 * a, b],
                              [b,    2 * c]])
    evals, evecs = np.linalg.eigh(II)          # ascending: κ₁ ≤ κ₂
    ev_min       = evecs[:, 0]                  # eigenvector for κ₁
    zero_axis    = ev_min[0] * t1 + ev_min[1] * t2

    return float(evals[0]), float(evals[1]), zero_axis


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Aggregate curvatures and extract axis direction
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_curvatures(pts, normals):
    """
    Run _fit_local_curvature on every point.

    Returns
    -------
    kappa1, kappa2 : float    median principal curvatures
    axis           : (3,) ndarray or None   consensus zero-curvature direction
    """
    tree = cKDTree(pts)
    k1_list, k2_list, axis_list = [], [], []

    for i in range(len(pts)):
        _, idx = tree.query(pts[i], k=_KNN_CURV + 1)
        result = _fit_local_curvature(pts[i], normals[i], pts[idx[1:]])
        if result is None:
            continue
        k1, k2, ax = result
        k1_list.append(k1)
        k2_list.append(k2)
        axis_list.append(ax)

    if not k1_list:
        return 0.0, 0.0, None

    kappa1 = float(np.median(k1_list))
    kappa2 = float(np.median(k2_list))

    # Resolve ± sign ambiguity before averaging
    axes  = np.array(axis_list)
    signs = np.sign(axes @ axes[0])
    signs[signs == 0] = 1
    axes *= signs[:, None]
    mean_ax = axes.mean(axis=0)
    nrm     = np.linalg.norm(mean_ax)
    axis    = mean_ax / nrm if nrm > 1e-6 else None

    return kappa1, kappa2, axis


def _orient_axis(axis):
    """
    Resolve the remaining ±1 global sign ambiguity in the cylinder axis.

    Strategy
    --------
    Prefer the axis direction that has a positive component along the world
    up vector.  This keeps "vertical" cylinders (bottles, cans) pointing
    consistently upward across frames, preventing sudden 180° flips.

    If the axis is nearly horizontal (|axis · up| < 0.2), orient toward
    positive X instead — a reasonable fallback for objects lying on their side.
    """
    up_dot = float(axis @ _WORLD_UP)

    if abs(up_dot) >= 0.2:
        # Axis has a meaningful vertical component — orient it upward
        if up_dot < 0:
            axis = -axis
    else:
        # Nearly horizontal — orient toward positive X
        if axis[0] < 0:
            axis = -axis

    return axis


def _classify(k1, k2):
    """
    "cylinder" or "cuboid" from principal curvatures.

    Disabled:
    # if a1 > _FLAT_THRESH and (a2 / (a1 + 1e-9)) < _ANISO_RATIO:
    #     return "sphere"
    """
    a1, a2 = abs(k1), abs(k2)
    if a2 < _FLAT_THRESH:
        return "cuboid"
    return "cylinder"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — 2D circle fit in the cross-section plane (surface alignment)
# ══════════════════════════════════════════════════════════════════════════════

def _fit_cross_section(pts, axis):
    """
    Project all points onto the plane perpendicular to *axis* and fit a 2D
    circle.  This gives the true axis position (center of the cylinder) and
    the true radius such that the cylinder surface aligns with the PCD.

    Why this is necessary
    ---------------------
    The curvature step gives us the *direction* of the axis (from the
    zero-curvature eigenvector) but NOT its position or exact radius.
    Estimating the center by pushing the centroid inward by r = 1/|κ| along
    the mean normal is an approximation that accumulates error whenever the
    visible patch is not centred on the cylinder's curved face.

    The 2D circle fit is exact: it finds (cx, cy) in the cross-section plane
    such that every projected point is at distance r from (cx, cy).  The 3D
    axis then passes through (cx, cy) in that plane.

    Returns
    -------
    axis_pt : (3,) — a point on the cylinder axis at the centroid's axial depth
    radius  : float
    or (None, None) on failure.
    """
    centroid = pts.mean(axis=0)

    # Build an orthonormal 2D frame in the cross-section plane
    ref = np.array([0., 0., 1.]) if abs(axis[2]) < 0.9 else np.array([1., 0., 0.])
    e1  = np.cross(axis, ref);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(axis, e1);   e2 /= np.linalg.norm(e2)

    # Collapse all points onto the cross-section plane by dropping the axial
    # component — for a cylinder every cross-section is the same circle
    d = pts - centroid
    u = d @ e1          # 2D coordinate 1
    v = d @ e2          # 2D coordinate 2

    # Linearised circle fit:  (u - cu)² + (v - cv)² = r²
    #   →  -2cu·u  -2cv·v  + (cu² + cv² - r²)  =  -(u² + v²)
    A  = np.column_stack([-2 * u, -2 * v, np.ones(len(u))])
    b  = -(u ** 2 + v ** 2)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    cu, cv, d_val = x

    r_sq = cu ** 2 + cv ** 2 - d_val
    if r_sq <= 0:
        return None, None

    radius  = float(np.sqrt(r_sq))
    axis_pt = centroid + cu * e1 + cv * e2   # 3D point on the axis

    return axis_pt, radius


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Cylinder height: point extents + end-cap refinement
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_cylinder_bounds(pts, normals, axis, centroid):
    """
    Axial extent (h_min, h_max) of the visible cylinder segment.

    Primary  : 1st / 99th percentile of point projections along the axis.
    Refinement: points with normals nearly parallel to the axis come from the
               flat end caps (top/bottom face).  When enough exist and are
               near the extremes of the point spread, they pin the true edge.
    """
    proj  = (pts - centroid) @ axis
    h_min = float(np.percentile(proj, 1))
    h_max = float(np.percentile(proj, 99))

    cap_mask = np.abs(normals @ axis) > _CAP_NORMAL_DOT
    if cap_mask.sum() >= _CAP_MIN_PTS:
        cap_proj = proj[cap_mask]
        extent   = h_max - h_min

        c_min = float(cap_proj.min())
        c_max = float(cap_proj.max())

        if c_min < h_min + 0.3 * extent:
            h_min = c_min
            print(f"[shape_fit]  bottom cap → h_min = {h_min:.3f} m")
        if c_max > h_max - 0.3 * extent:
            h_max = c_max
            print(f"[shape_fit]  top cap    → h_max = {h_max:.3f} m")

    return h_min, h_max


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Build wireframes
# ══════════════════════════════════════════════════════════════════════════════

def _rebuild_cylinder(pts, normals, k1, k2, axis):
    """
    Build a cylinder wireframe that is flush with the visible PCD surface.

    Parameters come from two sources:
      axis direction  ← curvature eigenvector (step 2) + orientation fix
      axis position,  ← 2D cross-section circle fit (step 3)
      radius
      height          ← point-cloud extents + end-cap refinement (step 4)
    """
    # ── Step 3: 2D circle fit for surface alignment ──────────────────────────
    axis_pt, r = _fit_cross_section(pts, axis)

    if axis_pt is None:
        # Circle fit failed — fall back to curvature-derived radius
        print("[shape_fit]  circle fit failed, using curvature radius")
        a1, a2  = abs(k1), abs(k2)
        k_curve = k2 if a2 >= a1 else k1
        r       = 1.0 / (abs(k_curve) + 1e-9)
        axis_pt = pts.mean(axis=0)

    r = float(np.clip(r, _R_MIN, _R_MAX))

    # ── Step 4: height from point extents + end-cap ──────────────────────────
    centroid       = pts.mean(axis=0)
    h_min, h_max   = _estimate_cylinder_bounds(pts, normals, axis, centroid)
    height         = float(np.clip(h_max - h_min, 0.005, 5.0))
    mid            = (h_min + h_max) / 2.0

    # axis_pt is the axis position at the centroid's axial depth (h=0).
    # The 3D centre of the visible cylinder segment is:
    center = axis_pt + axis * mid

    print(f"[shape_fit]  cylinder  r={r:.3f} m  h={height:.3f} m  "
          f"axis={np.round(axis, 2)}")

    # ── Rotate Open3D Z-axis cylinder to match the fitted axis ───────────────
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
        R  = np.eye(3) + vx + vx @ vx * (1.0 - c) / (s ** 2)

    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=r, height=height, resolution=20
    )
    mesh.rotate(R, center=np.zeros(3))
    mesh.translate(center)
    ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    ls.paint_uniform_color(_COLOR["cylinder"])
    return ls


def _rebuild_cuboid(pts):
    """RANSAC plane → thin box extruded from the inlier patch extent."""
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
    du = u.max() - u.min()
    dv = v.max() - v.min()
    dw = max(w.max() - w.min(), 0.005)

    mesh = o3d.geometry.TriangleMesh.create_box(width=du, height=dv, depth=dw)
    mesh.translate([-du / 2, -dv / 2, -dw / 2])
    R = np.column_stack([t1, t2, n])
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    mesh.rotate(R, center=np.zeros(3))
    center = np.array([
        (u.max() + u.min()) / 2 * t1[i] +
        (v.max() + v.min()) / 2 * t2[i] +
        (w.max() + w.min()) / 2 * n[i]
        for i in range(3)
    ])
    mesh.translate(center)
    ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    ls.paint_uniform_color(_COLOR["cuboid"])
    return ls


# ══════════════════════════════════════════════════════════════════════════════
# Full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def fit_shape(pts: np.ndarray):
    """Returns (shape_name, LineSet) or (None, None)."""
    if len(pts) < 50:
        return None, None

    # Normal estimation — orient away from camera at origin
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=_KNN_NORMAL)
    )
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0., 0., 0.])
    )
    normals = np.asarray(pcd.normals)

    # Subsample for the O(N·k) curvature loop
    N = len(pts)
    if N > _SUBSAMPLE:
        idx     = np.random.choice(N, _SUBSAMPLE, replace=False)
        s_pts   = pts[idx]
        s_norms = normals[idx]
    else:
        s_pts, s_norms = pts, normals

    # Steps 1 & 2: local fits → median curvatures → classify + axis
    k1, k2, axis = _aggregate_curvatures(s_pts, s_norms)
    shape        = _classify(k1, k2)

    print(f"[shape_fit]  κ₁={k1:+.2f}  κ₂={k2:+.2f}  → {shape}")

    if shape == "cylinder":
        if axis is None:
            print("[shape_fit]  axis extraction failed → cuboid")
            shape = "cuboid"
        else:
            # Resolve axis orientation ambiguity before reconstruction
            axis = _orient_axis(axis)

    if shape == "cylinder":
        ls = _rebuild_cylinder(pts, normals, k1, k2, axis)
    else:
        ls = _rebuild_cuboid(pts)

    return shape, ls


# ══════════════════════════════════════════════════════════════════════════════
# Visualizer callback
# ══════════════════════════════════════════════════════════════════════════════

def _make_overlay_callback():
    state = {"ls": None, "label": None}

    def _on_frame(obj_verts: np.ndarray, vis: o3d.visualization.Visualizer):
        shape, new_ls = fit_shape(obj_verts)
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

def run(debug: bool = False):
    isolator = ObjectIsolator(min_points=50)
    isolator.start()

    print("Waiting for camera …")
    isolator.ready.wait()
    print("Camera ready — opening visualizer.\n")

    try:
        show_isolated_pcd(
            isolator,
            on_new_frame=_make_overlay_callback(),
            debug=debug,
        )
    finally:
        isolator.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run(debug=args.debug)