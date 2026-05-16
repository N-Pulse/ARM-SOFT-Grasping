"""
test_shape_fit.py
=================
Curvature-based primitive fitting for single-view point clouds.
Only cylinder and cuboid are fitted (sphere is disabled — see note below).

Design constraints
------------------
  · The camera captures only one face of the object.
  · However, the *entire* physical object is within the camera frame,
    so at least ~30% of the surface is visible.
  · This guarantees that the curvature signature of the visible patch is
    representative of the true primitive:
      - a cylinder exposes enough of its curved face that κ₁ ≈ 0, κ₂ ≠ 0
        is stable and reliable;
      - a flat face exposes enough surface that κ₁ ≈ κ₂ ≈ 0 is reliable.
  · Given this, curvature alone is sufficient for classification.
    No secondary consistency check is needed.

Why sphere is disabled
----------------------
  Sphere (κ₁ ≈ κ₂ ≠ 0) is commented out.  With ≥30% of the object visible,
  a sphere is distinguishable in principle, but in practice the objects in
  this pipeline are bottles, cans, and boxes — not spheres.  Re-enable the
  sphere branch if needed.

Pipeline
--------
  1. LOCAL   — per-point quadratic patch fit → κ₁ ≤ κ₂, zero-curvature axis
  2. FEATURE — median κ₁, κ₂ over patch → classify; consensus axis direction
  3. REBUILD — cylinder: r from curvature, height from point-cloud extent
                         + end-cap refinement if top/bottom faces are visible
               cuboid : RANSAC plane + thin box

Wireframe colours
-----------------
  cylinder → orange   [1.0, 0.5, 0.0]
  cuboid   → lavender [0.8, 0.6, 1.0]

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
_KNN_CURV       = 25     # neighbours for local quadratic curvature fit
_SUBSAMPLE      = 600    # max points in the curvature loop (speed)
_FLAT_THRESH    = 2.0    # |κ| < this (m⁻¹) → treated as zero curvature
_ANISO_RATIO    = 5.0    # |κ_max / κ_min| > this → cylinder
_R_MIN          = 0.005  # smallest plausible radius (m)
_R_MAX          = 2.0    # largest  plausible radius (m)
_CAP_NORMAL_DOT = 0.7    # |n · axis| > this → point belongs to an end cap
_CAP_MIN_PTS    = 10     # minimum end-cap points needed to trust cap detection


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Local principal curvatures at a single point
# ══════════════════════════════════════════════════════════════════════════════

def _fit_local_curvature(p, n, neighbours):
    """
    Fit  h = a·u² + b·u·v + c·v²  in the tangent frame of (p, n).

    The second fundamental form  [[2a, b], [b, 2c]]  yields eigenvalues κ₁ ≤ κ₂
    (principal curvatures) and eigenvectors (principal directions).

    The eigenvector of κ₁ (the *smaller* curvature) is the zero-curvature
    direction — this becomes the cylinder axis candidate.

    Returns (κ₁, κ₂, zero_axis_3d) or None on degenerate input.
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
    evals, evecs = np.linalg.eigh(II)      # ascending: κ₁ ≤ κ₂

    zero_axis = evecs[0, 0] * t1 + evecs[1, 0] * t2   # κ₁ eigenvector in 3D

    return float(evals[0]), float(evals[1]), zero_axis


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Aggregate and classify
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_curvatures(pts, normals):
    """
    Run _fit_local_curvature on every point; return:
      kappa1, kappa2 — median principal curvatures over the patch
      axis           — consensus zero-curvature direction (cylinder axis)
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

    # Robust mean of unit axes — resolve ± sign ambiguity before averaging
    axes  = np.array(axis_list)
    signs = np.sign(axes @ axes[0])
    signs[signs == 0] = 1
    axes *= signs[:, None]
    mean_ax = axes.mean(axis=0)
    nrm     = np.linalg.norm(mean_ax)
    axis    = mean_ax / nrm if nrm > 1e-6 else None

    return kappa1, kappa2, axis


def _classify(k1, k2):
    """
    Classify into "cylinder" or "cuboid" from principal curvatures.

    With ≥30% of the object visible, the curvature signature is stable:
      · κ₂ negligible              → flat surface → cuboid
      · κ₁ ≈ 0 but κ₂ significant → one curved direction → cylinder
      · both significant (isotropic) → would be sphere, but sphere is
        disabled; treated as cylinder (falls back to cuboid if axis fails)

    Disabled:
    # if a1 > _FLAT_THRESH and (a2 / (a1 + 1e-9)) < _ANISO_RATIO:
    #     return "sphere"
    """
    a1, a2 = abs(k1), abs(k2)

    if a2 < _FLAT_THRESH:
        return "cuboid"

    # One or both curvatures are significant → attempt cylinder.
    # With ≥30% surface visible, a genuine cylinder will have a clear axis;
    # if axis extraction fails the pipeline falls back to cuboid automatically.
    return "cylinder"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Cylinder height from point-cloud extents + end-cap detection
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_cylinder_bounds(pts, normals, axis, centroid):
    """
    Find the axial extent (h_min, h_max) of the visible cylinder segment.

    Two evidence sources:

    1. Point extents (primary)
       Project every point onto the axis; take 1st/99th percentile to trim
       depth-camera fringe noise at object boundaries.

    2. End-cap points (refinement)
       Points whose normals are nearly parallel to the axis come from the
       flat top or bottom face of the cylinder (when those faces are visible).
       These give sharper bounds than the raw point spread because they sit
       exactly on the rim of the cylinder.

       Guard: end-cap bounds are only accepted if the detected cap lies within
       the outer 30% of the raw extent — this prevents mid-surface noisy
       normals from being mistaken for caps.
    """
    proj  = (pts - centroid) @ axis
    h_min = float(np.percentile(proj, 1))
    h_max = float(np.percentile(proj, 99))

    # End-cap detection: normals roughly parallel to axis
    cap_mask = np.abs(normals @ axis) > _CAP_NORMAL_DOT

    if cap_mask.sum() >= _CAP_MIN_PTS:
        cap_proj = proj[cap_mask]
        extent   = h_max - h_min

        candidate_min = float(cap_proj.min())
        candidate_max = float(cap_proj.max())

        if candidate_min < h_min + 0.3 * extent:
            h_min = candidate_min
            print(f"[shape_fit]  bottom cap detected  h_min = {h_min:.3f} m")

        if candidate_max > h_max - 0.3 * extent:
            h_max = candidate_max
            print(f"[shape_fit]  top cap detected     h_max = {h_max:.3f} m")

    return h_min, h_max


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Build wireframes
# ══════════════════════════════════════════════════════════════════════════════

def _rebuild_cylinder(pts, normals, k1, k2, axis):
    """
    Build a cylinder wireframe.

    Radius  r = 1 / |κ_max|         (from curvature — no bounding-box guessing)
    Height  from _estimate_cylinder_bounds (point extent + end-cap refinement)
    Centre  centroid pushed inward along the mean radial normal by r,
            then shifted to the axial midpoint.
    """
    a1, a2  = abs(k1), abs(k2)
    k_curve = k2 if a2 >= a1 else k1
    r       = float(np.clip(1.0 / (abs(k_curve) + 1e-9), _R_MIN, _R_MAX))

    centroid            = pts.mean(axis=0)
    h_min, h_max        = _estimate_cylinder_bounds(pts, normals, axis, centroid)
    height              = float(np.clip(h_max - h_min, 0.005, 5.0))
    mid                 = (h_min + h_max) / 2.0

    # Inward radial direction: mean normal minus its axial component
    mean_n  = normals.mean(axis=0)
    mean_n -= (mean_n @ axis) * axis
    nrm     = np.linalg.norm(mean_n)
    if nrm > 1e-6:
        mean_n /= nrm
        center = centroid - mean_n * r + axis * mid
    else:
        center = centroid + axis * mid

    # Rotate Open3D's default Z-axis cylinder to match the fitted axis
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
    """
    RANSAC plane fit → thin box extruded from the inlier patch extent.
    """
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
# Full pipeline dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def fit_shape(pts: np.ndarray):
    """
    Full pipeline for one frame.  Returns (shape_name, LineSet) or (None, None).
    """
    if len(pts) < 50:
        return None, None

    # Normal estimation — orient away from camera (at origin in depth frame)
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

    # Steps 1 & 2: local fits → aggregate → classify
    k1, k2, axis = _aggregate_curvatures(s_pts, s_norms)
    shape        = _classify(k1, k2)

    print(f"[shape_fit]  κ₁={k1:+.2f}  κ₂={k2:+.2f}  → {shape}")

    # Cylinder requires a valid axis; fall back to cuboid if extraction failed
    if shape == "cylinder" and axis is None:
        print("[shape_fit]  axis extraction failed → cuboid")
        shape = "cuboid"

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