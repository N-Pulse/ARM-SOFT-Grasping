"""
test_shape_fit.py
=================
Curvature-based geometric primitive fitting for single-view point clouds.

Pipeline
--------
  1. LOCAL   — for every point, fit a quadratic patch in its tangent frame
               → principal curvatures κ₁ ≤ κ₂ and the zero-curvature axis
  2. FEATURE — aggregate over the visible patch:
               · median κ₁, κ₂  →  shape classification
               · consensus zero-curvature direction  →  cylinder axis
               · mean curvature H = (κ₁+κ₂)/2  →  radius
  3. REBUILD — extrapolate the full primitive from the extracted features,
               not from point extents

Why not global fitting?
-----------------------
A partial point cloud (one visible face) cannot disambiguate sphere / cylinder /
flat by position alone — all three look identical head-on.  Curvature is a
*local* differential property that encodes shape type in even a small patch.

Shape classification
--------------------
  κ₁ ≈ κ₂ ≠ 0          →  sphere    (isotropic curvature in all directions)
  κ₁ ≈ 0,  κ₂ ≠ 0      →  cylinder  (flat along axis, curved across it)
  κ₁ ≈ κ₂ ≈ 0          →  cuboid / flat

Wireframe colours
-----------------
  sphere   → cyan     [0.2, 0.8, 1.0]
  cylinder → orange   [1.0, 0.5, 0.0]
  cuboid   → lavender [0.8, 0.6, 1.0]

Usage
-----
  python test_shape_fit.py [--debug]

  --debug   show the raw full point cloud instead of the isolated object

Dependencies
------------
  open3d, numpy, scipy
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
    "sphere":   [0.2, 0.8, 1.0],
    "cylinder": [1.0, 0.5, 0.0],
    "cuboid":   [0.8, 0.6, 1.0],
}

# ── Tuning ─────────────────────────────────────────────────────────────────────
# All curvature values are in m⁻¹ (inverse metres).
# A sphere/cylinder of radius 5 cm has κ ≈ 20 m⁻¹.
# A flat surface has κ ≈ 0 m⁻¹.
# Adjust _FLAT_THRESH if your objects are much larger or smaller than ~5–20 cm.

_KNN_NORMAL   = 30     # neighbours for Open3D normal estimation
_KNN_CURV     = 25     # neighbours for local quadratic curvature fit
_SUBSAMPLE    = 600    # max points used in the curvature loop (speed)
_FLAT_THRESH  = 2.0    # |κ| < this  →  treated as "zero curvature"
_ANISO_RATIO  = 5.0    # |κ_max/κ_min| > this  →  cylinder rather than sphere
_R_MIN        = 0.005  # smallest plausible radius (m)
_R_MAX        = 2.0    # largest  plausible radius (m)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Local curvature at a single point
# ══════════════════════════════════════════════════════════════════════════════

def _fit_local_curvature(p, n, neighbours):
    """
    Estimate the principal curvatures at point *p* with outward normal *n*
    using the surrounding *neighbours* (k×3 array, not including *p* itself).

    Method
    ------
    Express each neighbour in the local tangent frame (t1, t2, n).
    Its coordinates are (u, v, h) where u,v are in-plane and h is height.
    For a smooth surface:  h ≈ a·u² + b·u·v + c·v²   (quadratic, no linear
    term because n is the exact normal at p).

    The second fundamental form  II = [[2a, b], [b, 2c]]  has eigenvalues
    equal to the two principal curvatures κ₁ ≤ κ₂.

    Returns
    -------
    (kappa1, kappa2, zero_axis_3d)
      kappa1, kappa2 : float  — principal curvatures, |κ₁| ≤ |κ₂|
      zero_axis_3d   : (3,)   — world-space direction of *minimal* curvature
                               (= cylinder axis when κ₁ ≈ 0)
    Returns None if the local neighbourhood is degenerate.
    """
    # --- build orthonormal tangent frame ---
    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    t1  = np.cross(n, ref);  t1 /= np.linalg.norm(t1)
    t2  = np.cross(n, t1);   t2 /= np.linalg.norm(t2)

    # --- project neighbours into (u, v, h) ---
    d = neighbours - p
    u = d @ t1
    v = d @ t2
    h = d @ n

    # --- least-squares quadratic fit: h = a·u² + b·u·v + c·v² ---
    A = np.column_stack([u ** 2, u * v, v ** 2])
    if np.linalg.matrix_rank(A) < 3:
        return None                         # degenerate (coplanar neighbours)

    (a, b, c), *_ = np.linalg.lstsq(A, h, rcond=None)

    # --- second fundamental form → principal curvatures ---
    II            = np.array([[2 * a, b],
                               [b,    2 * c]])
    evals, evecs  = np.linalg.eigh(II)     # ascending: |κ₁| ≤ |κ₂|

    # Convert the minimal-curvature 2-D eigenvector back to world space
    ev_min     = evecs[:, 0]               # eigenvec of the smaller |κ|
    zero_axis  = ev_min[0] * t1 + ev_min[1] * t2

    return float(evals[0]), float(evals[1]), zero_axis


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Aggregate curvatures over the whole visible patch
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate_curvatures(pts, normals):
    """
    Run _fit_local_curvature on every point in *pts* and aggregate:
      · median κ₁, κ₂  (robust to outliers from noisy regions)
      · mean zero-curvature axis  (cylinder axis candidate)

    Returns
    -------
    kappa1 : float     median of the smaller principal curvature
    kappa2 : float     median of the larger principal curvature
    axis   : (3,) or None   consensus zero-curvature direction
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

    # Robust average of unit axes (handle ± sign ambiguity)
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
    Map (κ₁, κ₂) to a shape label.

    Decision logic
    --------------
    |κ₂| < _FLAT_THRESH          → both curvatures negligible  → cuboid
    |κ₁| < _FLAT_THRESH  OR
    |κ₂/κ₁| > _ANISO_RATIO       → one direction is flat        → cylinder
    otherwise                    → both directions curved        → sphere
    """
    a1, a2 = abs(k1), abs(k2)      # |κ₁| ≤ |κ₂| from eigh

    if a2 < _FLAT_THRESH:
        return "cuboid"

    if a1 < _FLAT_THRESH or (a2 / (a1 + 1e-9)) > _ANISO_RATIO:
        return "cylinder"

    return "sphere"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Reconstruct full primitive from extracted features
# ══════════════════════════════════════════════════════════════════════════════

def _rebuild_sphere(pts, normals, k1, k2):
    """
    Reconstruct a sphere from curvature.

    Radius   r = 1/|H|  where  H = (κ₁+κ₂)/2  (mean curvature).
    Centre   = surface centroid shifted *inward* along the mean outward normal
               by r.  (Normals point away from the centre; we go the other way.)
    """
    H  = (k1 + k2) / 2.0
    r  = float(np.clip(1.0 / (abs(H) + 1e-9), _R_MIN, _R_MAX))

    centroid = pts.mean(axis=0)
    mean_n   = normals.mean(axis=0)
    mean_n  /= np.linalg.norm(mean_n) + 1e-9
    center   = centroid - mean_n * r        # inward = opposite outward normal

    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=20)
    mesh.translate(center)
    ls   = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    ls.paint_uniform_color(_COLOR["sphere"])
    return ls


def _rebuild_cylinder(pts, normals, k1, k2, axis):
    """
    Reconstruct a cylinder from curvature + axis direction.

    Radius   r = 1/|κ_max|  (the non-zero principal curvature).
    Axis     already extracted in Step 2 as the zero-curvature eigenvector.
    Height   from the point extents projected along the axis
             (only for wireframe height — this is the *visible* segment).
    Centre   surface centroid pushed inward along the mean radial direction.
    """
    a1, a2  = abs(k1), abs(k2)
    k_curve = k2 if a2 >= a1 else k1
    r       = float(np.clip(1.0 / (abs(k_curve) + 1e-9), _R_MIN, _R_MAX))

    centroid = pts.mean(axis=0)

    # Height: extent of visible points along axis
    proj   = (pts - centroid) @ axis
    height = float(np.clip(proj.max() - proj.min(), 0.005, 5.0))
    mid    = float((proj.max() + proj.min()) / 2.0)

    # Radial direction: mean normal with axial component removed
    mean_n  = normals.mean(axis=0)
    mean_n -= (mean_n @ axis) * axis        # project out the axial component
    nrm     = np.linalg.norm(mean_n)
    if nrm > 1e-6:
        mean_n /= nrm
        center = centroid - mean_n * r + axis * mid
    else:
        center = centroid + axis * mid

    # Rotate Open3D's Z-axis-aligned cylinder to match the fitted axis
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
    Fit a plane to the flat patch (RANSAC) and extrude a thin box.

    The box face matches the inlier patch extent; depth is whatever the
    point spread perpendicular to the plane is (minimum 5 mm so it's visible).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    plane, inlier_idx = pcd.segment_plane(
        distance_threshold=0.005, ransac_n=3, num_iterations=200
    )
    n   = np.array(plane[:3]);  n /= np.linalg.norm(n) + 1e-9
    inl = pts[np.array(inlier_idx)]

    # Tangent frame for the fitted plane
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
    Run the full three-step pipeline on one frame of isolated object points.

    Parameters
    ----------
    pts : (N, 3) float array   world-space points of the isolated object

    Returns
    -------
    (shape_name, LineSet)  or  (None, None) if there are too few points.
    """
    if len(pts) < 50:
        return None, None

    # --- Normal estimation ---
    # Orient normals away from the camera.  For a RealSense in depth-frame
    # coordinates the camera sits at the origin, so camera_location=[0,0,0].
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=_KNN_NORMAL)
    )
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0., 0., 0.])
    )
    normals = np.asarray(pcd.normals)

    # --- Subsample for the O(N·k) curvature loop ---
    N = len(pts)
    if N > _SUBSAMPLE:
        idx     = np.random.choice(N, _SUBSAMPLE, replace=False)
        s_pts   = pts[idx]
        s_norms = normals[idx]
    else:
        s_pts, s_norms = pts, normals

    # --- Steps 1 & 2: local fits → aggregate ---
    k1, k2, axis = _aggregate_curvatures(s_pts, s_norms)
    shape        = _classify(k1, k2)

    print(f"[shape_fit]  κ₁={k1:+.2f} m⁻¹   κ₂={k2:+.2f} m⁻¹   → {shape}")

    # --- Step 3: reconstruct full primitive ---
    if shape == "sphere":
        ls = _rebuild_sphere(pts, normals, k1, k2)

    elif shape == "cylinder":
        if axis is None:
            # Axis extraction failed — fall back to cuboid
            shape = "cuboid"
            ls    = _rebuild_cuboid(pts)
        else:
            ls = _rebuild_cylinder(pts, normals, k1, k2, axis)

    else:   # cuboid
        ls = _rebuild_cuboid(pts)

    return shape, ls


# ══════════════════════════════════════════════════════════════════════════════
# Visualizer callback
# ══════════════════════════════════════════════════════════════════════════════

def _make_overlay_callback():
    """
    Returns an on_new_frame callback that maintains a single wireframe
    LineSet in the Open3D visualizer, updating it in-place each frame.
    """
    state = {"ls": None, "label": None}

    def _on_frame(obj_verts: np.ndarray, vis: o3d.visualization.Visualizer):
        shape, new_ls = fit_shape(obj_verts)
        if new_ls is None:
            return

        if shape != state["label"]:
            print(f"[shape_fit]  *** shape → {shape} ***")

        if state["ls"] is None:
            # First frame: add geometry to the scene
            vis.add_geometry(new_ls)
            state["ls"] = new_ls
        else:
            # Subsequent frames: update in-place (avoids flickering)
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
    print("Camera ready — opening visualizer window.\n")

    try:
        show_isolated_pcd(
            isolator,
            on_new_frame=_make_overlay_callback(),
            debug=debug,
        )
    finally:
        isolator.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Curvature-based shape fitting over a single-view point cloud."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the full scene point cloud instead of the isolated object.",
    )
    args = parser.parse_args()
    run(debug=args.debug)