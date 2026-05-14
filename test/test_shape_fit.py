"""
test_shape_fit.py

Fits the best geometric primitive (ball, cylinder, cone, pyramid, cuboid) to
the isolated object point cloud and draws it as a coloured wireframe overlay
in the Open3D window.

Follows the same design as test_obj_iso.py: ObjectIsolator supplies the masked
point cloud; show_isolated_pcd handles all window / camera / cv2-preview logic;
the shape overlay is injected via the on_new_frame callback.

How fitting works
-----------------
1. PCA on the isolated points → three orthogonal principal axes + extents.
2. Two ratios (small/large, mid/large) determine the shape class:
       r1 = S/L,  r2 = M/L
3. The chosen mesh is built at the origin in PCA-local coordinates, then
   rotated to align with the principal axes and translated to the centroid.
4. The mesh is converted to a LineSet wireframe and added/updated in the viewer.

Shape colours
-------------
  ball     → cyan
  cylinder → orange
  cone     → pink-red
  pyramid  → green
  cuboid   → lavender

Usage:
    python test_shape_fit.py

Controls:
    Close the Open3D window or press Ctrl+C to stop.
"""

import sys
import os
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


# ── Mesh builders (PCA-local frame, centred at origin, major axis = Z) ────────

def _mesh_ball(L, M, S):
    r = (L + M + S) / 6
    return o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=20)


def _mesh_cylinder(L, M, S):
    r = (M + S) / 4          # radius from the two smaller extents
    return o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=L,
                                                     resolution=20)


def _mesh_cone(L, M, S):
    r = (M + S) / 4
    mesh = o3d.geometry.TriangleMesh.create_cone(radius=r, height=L,
                                                 resolution=20)
    # Open3D cone: base at Z=0, tip at Z=L → centroid at Z=L/4; centre it
    mesh.translate([0.0, 0.0, -L / 4.0])
    return mesh


def _mesh_pyramid(L, M, S):
    b = (M + S) / 4      # half base side
    # Place centroid at origin: square pyramid centroid is at 1/4 height
    base_z = -L / 4.0
    apex_z =  L * 3.0 / 4.0
    verts = np.array([
        [-b, -b, base_z], [ b, -b, base_z],
        [ b,  b, base_z], [-b,  b, base_z],
        [ 0,  0, apex_z],
    ], dtype=np.float64)
    tris = np.array([
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],   # lateral
        [0, 2, 1], [0, 3, 2],                           # base
    ], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    return mesh


def _mesh_cuboid(L, M, S):
    mesh = o3d.geometry.TriangleMesh.create_box(width=S, height=M, depth=L)
    mesh.translate([-S / 2.0, -M / 2.0, -L / 2.0])    # centre at origin
    return mesh


# ── Shape classifier + wireframe builder ──────────────────────────────────────

def _fit_shape(pts: np.ndarray):
    """Fit a geometric primitive to *pts* (N×3 float32/64).

    Returns (shape_name, lineset) — the lineset is a coloured wireframe
    already positioned and oriented in world space.
    Returns (None, None) when there are too few points.
    """
    if len(pts) < 30:
        return None, None

    # ── PCA ───────────────────────────────────────────────────────────────────
    center   = pts.mean(axis=0)
    centered = pts - center

    _, evecs = np.linalg.eigh(np.cov(centered.T))   # eigenvalues ascending
    # Reorder so column 0 = major axis (largest variance)
    evecs = evecs[:, ::-1]                           # shape (3, 3)

    # Extents along each principal axis
    proj    = centered @ evecs                       # (N, 3)
    extents = proj.max(axis=0) - proj.min(axis=0)   # [L, M, S]
    L, M, S = extents                                # large, mid, small

    # ── Classify ──────────────────────────────────────────────────────────────
    r1 = S / L   # small / large  (→ 1 = isotropic, → 0 = flat/needle)
    r2 = M / L   # mid  / large

    if r1 > 0.75:
        shape = "ball"
    elif r2 > 0.72:
        shape = "cylinder"          # two large axes, one small
    elif r1 < 0.50 and abs(r1 - r2) < 0.18:
        shape = "cone"              # elongated, roughly circular cross-section
    elif r1 > 0.35 and r2 > 0.50 and abs(r1 - r2) < 0.25:
        shape = "pyramid"           # elongated, squarish cross-section
    else:
        shape = "cuboid"

    # ── Build mesh in PCA-local frame (centred at origin, major axis = Z) ─────
    builders = {
        "ball":     _mesh_ball,
        "cylinder": _mesh_cylinder,
        "cone":     _mesh_cone,
        "pyramid":  _mesh_pyramid,
        "cuboid":   _mesh_cuboid,
    }
    mesh = builders[shape](L, M, S)

    # ── Transform to world frame ───────────────────────────────────────────────
    # evecs columns are [major, mid, minor] axes in world coords.
    # The mesh was built so that Z=major, Y=mid, X=minor.
    # evecs already maps that local frame to world → apply as rotation.
    mesh.rotate(evecs, center=np.zeros(3))
    mesh.translate(center)

    # ── Convert to wireframe ──────────────────────────────────────────────────
    lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    lineset.paint_uniform_color(_COLORS[shape])

    return shape, lineset


# ── on_new_frame callback factory ─────────────────────────────────────────────

def _make_shape_overlay():
    """Return a stateful callback for show_isolated_pcd's on_new_frame hook."""
    state = {"lineset": None, "label": None}

    def _on_frame(obj_verts: np.ndarray, vis: o3d.visualization.Visualizer):
        shape, new_ls = _fit_shape(obj_verts)
        if new_ls is None:
            return

        if shape != state["label"]:
            print(f"[shape_fit] shape → {shape}")

        if state["lineset"] is None:
            # First detection — register with the visualiser
            vis.add_geometry(new_ls)
            state["lineset"] = new_ls
        else:
            # Update in-place (no re-registration needed)
            ls = state["lineset"]
            ls.points = new_ls.points
            ls.lines  = new_ls.lines
            ls.colors = new_ls.colors
            vis.update_geometry(ls)

        state["label"] = shape

    return _on_frame


# ── Entry point ───────────────────────────────────────────────────────────────

def run():
    isolator = ObjectIsolator(min_points=50)
    isolator.start()

    print("Waiting for YOLO to load...")
    isolator.ready.wait()
    print("YOLO ready — opening window.\n")

    try:
        show_isolated_pcd(isolator, on_new_frame=_make_shape_overlay())
    finally:
        isolator.stop()


if __name__ == "__main__":
    run()
