"""
test_shape_fit.py

Fits the best geometric primitive (ball, cylinder, cone, pyramid, cuboid) to
the isolated red object point cloud and draws it as a coloured wireframe
overlay in the Open3D window.

Identical to test_obj_iso.py up to the shape-fitting overlay injected via the
on_new_frame callback.  Only fits when a red object is detected (obj_verts
non-empty); the wireframe is hidden otherwise.

How fitting works
-----------------
1. PCA on the isolated points → three orthogonal principal axes + extents.
2. Two ratios (S/L, M/L) determine the shape class.
3. Mesh dimensions are computed from actual point distributions (mean radial
   distance from axis, etc.) rather than raw bounding extents, so the
   wireframe surface overlaps the real cloud as closely as possible.
4. The mesh is rotated into world frame and converted to a LineSet wireframe.

Shape colours
-------------
  ball     → cyan
  cylinder → orange
  cone     → pink-red
  pyramid  → green
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


# ── Shape colours ─────────────────────────────────────────────────────────────

_COLORS = {
    "ball":     [0.2, 0.8, 1.0],   # cyan
    "cylinder": [1.0, 0.5, 0.0],   # orange
    "cone":     [1.0, 0.2, 0.4],   # pink-red
    "pyramid":  [0.4, 1.0, 0.2],   # green
    "cuboid":   [0.8, 0.6, 1.0],   # lavender
}


# ── Mesh builders ─────────────────────────────────────────────────────────────
# All meshes are built in PCA-local coordinates:
#   local Z = major axis (L),  local Y = mid axis (M),  local X = minor (S)
# Sizes are derived from the actual point distribution so the wireframe surface
# sits on the real cloud rather than wrapping an abstract bounding box.
#
# pts_local : (N,3) array — points projected onto PCA axes, centred at origin.
# L, M, S   : extents along major / mid / minor axes.

def _mesh_ball(pts_local, L, M, S):
    # Mean distance from centroid ≈ sphere radius observed from any direction.
    r = float(np.mean(np.linalg.norm(pts_local, axis=1)))
    return o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=20)


def _mesh_cylinder(pts_local, L, M, S):
    # Radial distance from the major axis (local Z).
    # pts_local[:,1] = mid projection, pts_local[:,2] = minor projection.
    radial = np.sqrt(pts_local[:, 1] ** 2 + pts_local[:, 2] ** 2)
    r = float(np.mean(radial))
    return o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=L,
                                                     resolution=20)


def _mesh_cone(pts_local, L, M, S):
    # Estimate base radius from the widest cross-section (one end of major axis).
    # Split points into bottom half and top half along local Z, take the half
    # with larger mean radial spread as the base.
    radial = np.sqrt(pts_local[:, 1] ** 2 + pts_local[:, 2] ** 2)
    z      = pts_local[:, 0]
    lo_r   = float(np.mean(radial[z < np.median(z)])) if (z < np.median(z)).any() else 0.0
    hi_r   = float(np.mean(radial[z > np.median(z)])) if (z > np.median(z)).any() else 0.0
    r_base = max(lo_r, hi_r, (M + S) / 4)

    mesh = o3d.geometry.TriangleMesh.create_cone(radius=r_base, height=L,
                                                 resolution=20)
    # Open3D cone: base at Z=0, tip at Z=L → centroid at Z=L/4; centre it.
    mesh.translate([0.0, 0.0, -L / 4.0])
    return mesh


def _mesh_pyramid(pts_local, L, M, S):
    # Half base side from the wider end of the cloud.
    radial = np.sqrt(pts_local[:, 1] ** 2 + pts_local[:, 2] ** 2)
    z      = pts_local[:, 0]
    lo_r   = float(np.mean(radial[z < np.median(z)])) if (z < np.median(z)).any() else 0.0
    hi_r   = float(np.mean(radial[z > np.median(z)])) if (z > np.median(z)).any() else 0.0
    b = max(lo_r, hi_r, (M + S) / 4)

    base_z = -L / 4.0      # centroid of pyramid at 1/4 height
    apex_z =  L * 3.0 / 4.0
    verts = np.array([
        [-b, -b, base_z], [ b, -b, base_z],
        [ b,  b, base_z], [-b,  b, base_z],
        [ 0,  0, apex_z],
    ], dtype=np.float64)
    tris = np.array([
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
        [0, 2, 1], [0, 3, 2],
    ], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    return mesh


def _mesh_cuboid(pts_local, L, M, S):
    # PCA extents are exactly the right dimensions for a box.
    mesh = o3d.geometry.TriangleMesh.create_box(width=S, height=M, depth=L)
    mesh.translate([-S / 2.0, -M / 2.0, -L / 2.0])
    return mesh


# ── Shape classifier + wireframe builder ──────────────────────────────────────

def _fit_shape(pts: np.ndarray):
    """Fit a geometric primitive to *pts* (N×3).

    Returns (shape_name, lineset) — the lineset is a coloured wireframe already
    positioned and oriented in world space.
    Returns (None, None) when there are too few points.
    """
    if len(pts) < 30:
        return None, None

    # ── PCA ───────────────────────────────────────────────────────────────────
    center   = pts.mean(axis=0)
    centered = pts - center

    _, evecs = np.linalg.eigh(np.cov(centered.T))  # eigenvalues ascending
    evecs    = evecs[:, ::-1]                        # now descending: col0=major

    # Project points onto PCA axes; extents along each axis.
    pts_local = centered @ evecs           # (N,3) — local coords
    extents   = pts_local.max(axis=0) - pts_local.min(axis=0)
    L, M, S   = extents                   # large, mid, small

    if L < 1e-6:
        return None, None

    # ── Classify ──────────────────────────────────────────────────────────────
    r1 = S / L   # small / large
    r2 = M / L   # mid   / large

    if r1 > 0.75:
        shape = "ball"
    elif r2 > 0.72:
        shape = "cylinder"
    elif r1 < 0.50 and abs(r1 - r2) < 0.18:
        shape = "cone"
    elif r1 > 0.35 and r2 > 0.50 and abs(r1 - r2) < 0.25:
        shape = "pyramid"
    else:
        shape = "cuboid"

    # ── Build mesh in PCA-local frame (major axis = local Z) ──────────────────
    builders = {
        "ball":     _mesh_ball,
        "cylinder": _mesh_cylinder,
        "cone":     _mesh_cone,
        "pyramid":  _mesh_pyramid,
        "cuboid":   _mesh_cuboid,
    }
    mesh = builders[shape](pts_local, L, M, S)

    # ── Rotate into world frame ────────────────────────────────────────────────
    # The mesh was built with local-Z = major, local-Y = mid, local-X = minor.
    # We need the rotation R such that:
    #   R @ [0,0,1] = evecs[:,0]  (major)
    #   R @ [0,1,0] = evecs[:,1]  (mid)
    #   R @ [1,0,0] = evecs[:,2]  (minor)
    # That gives R = evecs reordered as [minor | mid | major] columns.
    R = evecs[:, [2, 1, 0]]
    # Ensure proper rotation (det = +1); flip one column if it's a reflection.
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1

    mesh.rotate(R, center=np.zeros(3))
    mesh.translate(center)

    # ── Convert to wireframe ──────────────────────────────────────────────────
    lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    lineset.paint_uniform_color(_COLORS[shape])

    return shape, lineset


# ── on_new_frame callback ─────────────────────────────────────────────────────

def _make_shape_overlay():
    """Return a stateful callback for show_isolated_pcd's on_new_frame hook.

    The callback is only invoked when obj_verts is non-empty (red detected),
    so the wireframe is naturally absent when there is no detection.
    """
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
