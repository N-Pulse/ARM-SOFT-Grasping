"""
test_shape_fit.py

Fits a geometric primitive (sphere, cylinder, cuboid) to the visible surface
of the isolated red object and draws it as a wireframe overlay.

Since we only ever see one side of the object, fitting is done by matching
the observed surface patch — not by guessing from bounding-box extents.

Fitting strategy
----------------
Three primitives are tried every frame; the one whose surface best explains
the observed points (lowest mean point-to-surface distance) wins.

  Sphere   — least-squares sphere fit: find center c and radius r such that
              every point lies on the sphere surface (||p - c|| = r).
              Solved as a linear system.

  Cylinder — the normals of a cylinder all point radially outward and are
              therefore perpendicular to the axis.  PCA on the normals gives
              the axis direction (smallest-variance eigenvector).  Points are
              then projected onto the plane perpendicular to that axis and a
              2D circle is fitted to find the radius and axis position.

  Cuboid   — RANSAC plane fit.  If the surface is flat this wins trivially.
              The wireframe is a thin box whose face matches the visible patch.

Shape colours
-------------
  sphere   → cyan
  cylinder → orange
  cuboid   → lavender

Usage:
    python test_shape_fit.py [--debug]

Controls:
    Close the Open3D window or press Ctrl+C to stop.
"""

import sys
import os
import argparse
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from capture.object_isolation import ObjectIsolator
from helper.pcd_visualizer import show_isolated_pcd


_COLORS = {
    "sphere":   [0.2, 0.8, 1.0],   # cyan
    "cylinder": [1.0, 0.5, 0.0],   # orange
    "cuboid":   [0.8, 0.6, 1.0],   # lavender
}


# ── Surface fitting ────────────────────────────────────────────────────────────

def _fit_sphere(pts):
    """Least-squares sphere fit.

    Linearise  ||p - c||² = r²  as:
        -2px·cx - 2py·cy - 2pz·cz + (||c||² - r²) = -||p||²

    Returns (center, radius, mean_residual) or (None, None, inf) on failure.
    """
    A = np.column_stack([-2.0 * pts, np.ones(len(pts))])
    b = -(pts ** 2).sum(axis=1)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, cz, d = x
    r_sq = cx**2 + cy**2 + cz**2 - d
    if r_sq <= 0:
        return None, None, np.inf
    center = np.array([cx, cy, cz])
    r      = float(np.sqrt(r_sq))
    resid  = float(np.mean(np.abs(np.linalg.norm(pts - center, axis=1) - r)))
    return center, r, resid


def _fit_cylinder(pts, normals):
    """Fit a cylinder using normals to find the axis, then a 2-D circle.

    Cylinder normals are all perpendicular to the axis, so they span a plane.
    The axis direction is the eigenvector of the normal covariance matrix with
    the *smallest* eigenvalue (the direction with least normal variance).

    Returns (axis, center_on_axis, radius, height, mean_residual)
    or      (None, None, None, None, inf) on failure.
    """
    if len(normals) < 10:
        return None, None, None, None, np.inf

    # Axis from normal PCA
    cov           = np.cov(normals.T)
    evals, evecs  = np.linalg.eigh(cov)
    axis          = evecs[:, 0]                   # smallest eigenvalue
    axis          = axis / (np.linalg.norm(axis) + 1e-9)

    # Project points perpendicular to axis for 2-D circle fit
    mean_pt    = pts.mean(axis=0)
    along_axis = (pts - mean_pt) @ axis           # scalar projection on axis
    pts_perp   = pts - np.outer(along_axis, axis) # 3-D but in the cross-section plane

    A = np.column_stack([-2.0 * pts_perp, np.ones(len(pts_perp))])
    b = -(pts_perp ** 2).sum(axis=1)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, cz, d = x
    r_sq = cx**2 + cy**2 + cz**2 - d
    if r_sq <= 0:
        return None, None, None, None, np.inf

    r             = float(np.sqrt(r_sq))
    axis_center   = np.array([cx, cy, cz])
    radial_dist   = np.linalg.norm(pts_perp - axis_center, axis=1)
    resid         = float(np.mean(np.abs(radial_dist - r)))
    height        = float(along_axis.max() - along_axis.min())
    # Center of the cylinder segment in world space
    mid_along     = (along_axis.max() + along_axis.min()) / 2.0
    world_center  = axis_center + axis * mid_along

    return axis, world_center, r, height, resid


def _fit_plane(pts):
    """RANSAC plane fit.

    Returns (normal, offset_d, inlier_pts, mean_residual).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    plane, inlier_idx = pcd.segment_plane(
        distance_threshold=0.005,
        ransac_n=3,
        num_iterations=100,
    )
    n     = np.array(plane[:3])
    d     = float(plane[3])
    resid = float(np.mean(np.abs(pts @ n + d)))
    return n, d, pts[inlier_idx], resid


# ── Wireframe builders ─────────────────────────────────────────────────────────

def _wireframe_sphere(center, r):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=20)
    mesh.translate(center)
    ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    ls.paint_uniform_color(_COLORS["sphere"])
    return ls


def _wireframe_cylinder(axis, center, r, height):
    """Build a cylinder wireframe along *axis* centred at *center*."""
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=r, height=height, resolution=20
    )
    # Rotate Z-axis to match the fitted axis
    z    = np.array([0.0, 0.0, 1.0])
    v    = np.cross(z, axis)
    s    = np.linalg.norm(v)
    c    = float(np.dot(z, axis))
    if s < 1e-6:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R  = np.eye(3) + vx + vx @ vx * (1.0 - c) / (s ** 2)

    mesh.rotate(R, center=np.zeros(3))
    mesh.translate(center)
    ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    ls.paint_uniform_color(_COLORS["cylinder"])
    return ls


def _wireframe_cuboid(plane_normal, inlier_pts):
    """Build a thin box whose face matches the visible flat patch.

    Projects inlier points onto the plane, finds the 2-D bounding rectangle,
    and extrudes a box with a small depth so the wireframe is visible.
    """
    n = plane_normal / (np.linalg.norm(plane_normal) + 1e-9)

    # Build two orthonormal tangent vectors on the plane
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1  = np.cross(n, ref);  t1 /= np.linalg.norm(t1)
    t2  = np.cross(n, t1);   t2 /= np.linalg.norm(t2)

    # 2-D extents of the inlier patch
    u   = inlier_pts @ t1
    v   = inlier_pts @ t2
    w   = inlier_pts @ n
    du, dv, dw = u.max() - u.min(), v.max() - v.min(), max(w.max() - w.min(), 0.005)

    mesh = o3d.geometry.TriangleMesh.create_box(width=du, height=dv, depth=dw)
    mesh.translate([-du / 2, -dv / 2, -dw / 2])

    # Rotation: box axes [X,Y,Z] → [t1, t2, n]
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
    ls.paint_uniform_color(_COLORS["cuboid"])
    return ls


# ── Main fitting dispatcher ────────────────────────────────────────────────────

def _fit_shape(pts: np.ndarray):
    """Fit the best-matching surface primitive to *pts*.

    Returns (shape_name, lineset) or (None, None) if fitting fails.
    """
    if len(pts) < 50:
        return None, None

    # Estimate surface normals
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(pts)
    pcd_tmp.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
    )
    normals = np.asarray(pcd_tmp.normals)

    # Try all three fits and measure residuals
    sph_center, sph_r,  sph_err               = _fit_sphere(pts)
    cyl_axis, cyl_ctr, cyl_r, cyl_h, cyl_err  = _fit_cylinder(pts, normals)
    pln_n, pln_d, pln_inliers, pln_err         = _fit_plane(pts)

    candidates = []
    if sph_r is not None and 0.005 < sph_r < 1.0:
        candidates.append(("sphere",   sph_err))
    if cyl_r is not None and 0.005 < cyl_r < 1.0 and cyl_h > 0:
        candidates.append(("cylinder", cyl_err))
    candidates.append(("cuboid", pln_err))

    shape = min(candidates, key=lambda x: x[1])[0]

    if shape == "sphere":
        ls = _wireframe_sphere(sph_center, sph_r)
    elif shape == "cylinder":
        ls = _wireframe_cylinder(cyl_axis, cyl_ctr, cyl_r, cyl_h)
    else:
        ls = _wireframe_cuboid(pln_n, pln_inliers)

    return shape, ls


# ── on_new_frame callback ─────────────────────────────────────────────────────

def _make_shape_overlay():
    state = {"lineset": None, "label": None}

    def _on_frame(obj_verts: np.ndarray, vis: o3d.visualization.Visualizer):
        shape, new_ls = _fit_shape(obj_verts)
        if new_ls is None:
            return

        if shape != state["label"]:
            print(f"[shape_fit] shape → {shape}")

        if state["lineset"] is None:
            vis.add_geometry(new_ls)
            state["lineset"] = new_ls
        else:
            ls = state["lineset"]
            ls.points = new_ls.points
            ls.lines  = new_ls.lines
            ls.colors = new_ls.colors
            vis.update_geometry(ls)

        state["label"] = shape

    return _on_frame


# ── Entry point ───────────────────────────────────────────────────────────────

def run(debug: bool = False):
    isolator = ObjectIsolator(min_points=50)
    isolator.start()

    print("Waiting for camera to be ready...")
    isolator.ready.wait()
    print("Ready — opening window.\n")

    try:
        show_isolated_pcd(
            isolator,
            on_new_frame=_make_shape_overlay(),
            debug=debug,
        )
    finally:
        isolator.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the full point cloud instead of the isolated object.",
    )
    args = parser.parse_args()
    run(debug=args.debug)
