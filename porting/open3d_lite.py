"""
porting/open3d_lite.py
======================
A numpy + scipy stand-in for the *subset* of Open3D that armsoft's model uses,
so the pipeline can run on a board where Open3D is too heavy or unavailable.

This is a porting prototype, not the final form: it lets the unmodified
pipeline run without Open3D so both can be compared frame by frame.  The long-
term change is to make ``shape_fitter`` use plain arrays directly (see
porting/README.md).

    import porting.open3d_lite as o3l
    o3l.install()            # BEFORE importing armsoft: sys.modules["open3d"] = shim
    import armsoft

Covered (the complete list armsoft.core touches):

    geometry.PointCloud          .points .normals .voxel_down_sample()
                                 .cluster_dbscan() .estimate_normals()
                                 .orient_normals_towards_camera_location()
    geometry.KDTreeSearchParamKNN
    geometry.LineSet             .points .lines .colors .paint_uniform_color()
                                 LineSet.create_from_triangle_mesh()
    geometry.TriangleMesh        create_cylinder() .rotate() .translate()
    utility.Vector3dVector / Vector2iVector   (plain numpy arrays here)

Not covered: ``visualization`` (the --viz window) — use --save-preview.

Semantics copied from Open3D 0.19/0.20 and checked against it
(porting/compare_open3d.py):
  voxel    grid anchored at min_bound - voxel/2, output = mean per voxel
  DBSCAN   neighbours within eps *including the point itself*; core point if
           count >= min_points; noise = -1
  normals  PCA of the k nearest neighbours (self included), smallest eigvec
  cylinder 2 cap centres + (split+1) rings of `resolution` vertices, split=4
"""

from __future__ import annotations

import sys
import types

import numpy as np
from scipy.spatial import cKDTree

__version__ = "lite-0.1"


# ── utility ─────────────────────────────────────────────────────────────────

def Vector3dVector(a):                      # noqa: N802 — mirror Open3D names
    return np.asarray(a, dtype=np.float64).reshape(-1, 3)


def Vector2iVector(a):                      # noqa: N802
    return np.asarray(a, dtype=np.int32).reshape(-1, 2)


# ── point-cloud algorithms (usable directly, without the classes) ───────────

def voxel_down_sample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Mean of the points in each occupied voxel (Open3D's grid convention)."""
    pts = np.asarray(pts, np.float64)
    if len(pts) == 0:
        return pts.reshape(0, 3)
    origin = pts.min(axis=0) - voxel * 0.5
    idx = np.floor((pts - origin) / voxel).astype(np.int64)
    _, inv, counts = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((len(counts), 3))
    np.add.at(out, inv, pts)
    return out / counts[:, None]


def dbscan(pts: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    """DBSCAN labels (-1 = noise), same definition as Open3D's cluster_dbscan."""
    n = len(pts)
    labels = np.full(n, -1, np.int32)
    if n == 0:
        return labels
    nbrs = cKDTree(pts).query_ball_point(pts, r=eps)
    core = np.fromiter((len(nb) >= min_points for nb in nbrs), bool, n)
    cluster = 0
    for seed in range(n):
        if labels[seed] != -1 or not core[seed]:
            continue
        labels[seed] = cluster
        stack = [seed]
        while stack:
            p = stack.pop()
            if not core[p]:
                continue                   # border point: joins, does not expand
            for q in nbrs[p]:
                if labels[q] == -1:
                    labels[q] = cluster
                    stack.append(q)
        cluster += 1
    return labels


def estimate_normals(pts: np.ndarray, knn: int) -> np.ndarray:
    """Unit normals from PCA over the k nearest neighbours (self included)."""
    k = min(knn, len(pts))
    _, idx = cKDTree(pts).query(pts, k=k)
    nb = pts[idx.reshape(len(pts), k)]                       # (N, k, 3)
    d = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", d, d)
    _, vecs = np.linalg.eigh(cov)                            # ascending
    return vecs[:, :, 0]


# ── geometry classes ────────────────────────────────────────────────────────

class KDTreeSearchParamKNN:
    def __init__(self, knn: int = 30):
        self.knn = int(knn)


class PointCloud:
    def __init__(self):
        self.points = np.zeros((0, 3))
        self.normals = np.zeros((0, 3))
        self.colors = np.zeros((0, 3))

    def voxel_down_sample(self, voxel_size: float) -> "PointCloud":
        out = PointCloud()
        out.points = voxel_down_sample(self.points, voxel_size)
        return out

    def cluster_dbscan(self, eps: float, min_points: int, print_progress: bool = False):
        return dbscan(np.asarray(self.points), eps, min_points)

    def estimate_normals(self, search_param: KDTreeSearchParamKNN | None = None, **_):
        knn = search_param.knn if search_param is not None else 30
        self.normals = estimate_normals(np.asarray(self.points), knn)

    def orient_normals_towards_camera_location(self, camera_location=(0.0, 0.0, 0.0)):
        to_cam = np.asarray(camera_location, float) - self.points
        flip = np.einsum("ij,ij->i", self.normals, to_cam) < 0
        self.normals[flip] *= -1.0

    def has_normals(self) -> bool:
        return len(self.normals) == len(self.points) and len(self.points) > 0


class LineSet:
    def __init__(self, points=None, lines=None):
        self.points = np.zeros((0, 3)) if points is None else Vector3dVector(points)
        self.lines = np.zeros((0, 2), np.int32) if lines is None else Vector2iVector(lines)
        self.colors = np.zeros((0, 3))

    def paint_uniform_color(self, color) -> "LineSet":
        self.colors = np.tile(np.asarray(color, float), (len(self.lines), 1))
        return self

    @staticmethod
    def create_from_triangle_mesh(mesh: "TriangleMesh") -> "LineSet":
        tri = np.asarray(mesh.triangles)
        e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
        return LineSet(mesh.vertices.copy(), np.unique(e, axis=0))


class TriangleMesh:
    def __init__(self, vertices=None, triangles=None):
        self.vertices = np.zeros((0, 3)) if vertices is None else Vector3dVector(vertices)
        self.triangles = np.zeros((0, 3), np.int32) if triangles is None else np.asarray(triangles, np.int32)

    @staticmethod
    def create_cylinder(radius: float = 1.0, height: float = 2.0,
                        resolution: int = 20, split: int = 4, **_) -> "TriangleMesh":
        """Same vertex order as Open3D: top centre, bottom centre, then rings top→bottom."""
        a = 2.0 * np.pi * np.arange(resolution) / resolution
        ring = np.c_[radius * np.cos(a), radius * np.sin(a)]
        z = height / 2.0 - height * np.arange(split + 1) / split
        rings = np.concatenate([np.c_[ring, np.full(resolution, zi)] for zi in z])
        verts = np.vstack([[0, 0, height / 2.0], [0, 0, -height / 2.0], rings])
        R, tris = resolution, []
        vid = lambda k, i: 2 + k * R + (i % R)                  # noqa: E731
        for i in range(R):
            tris.append([0, vid(0, i), vid(0, i + 1)])
            tris.append([1, vid(split, i + 1), vid(split, i)])
            for k in range(split):
                tris.append([vid(k, i), vid(k + 1, i + 1), vid(k, i + 1)])
                tris.append([vid(k, i), vid(k + 1, i), vid(k + 1, i + 1)])
        return TriangleMesh(verts, tris)

    def rotate(self, R, center=(0.0, 0.0, 0.0)) -> "TriangleMesh":
        c = np.asarray(center, float)
        self.vertices = (self.vertices - c) @ np.asarray(R, float).T + c
        return self

    def translate(self, t, relative: bool = True) -> "TriangleMesh":
        self.vertices = self.vertices + np.asarray(t, float)
        return self


# ── module plumbing ─────────────────────────────────────────────────────────

def install() -> types.ModuleType:
    """Register this shim as ``open3d`` (call before importing armsoft)."""
    mod = types.ModuleType("open3d")
    mod.__version__ = __version__
    mod.__file__ = __file__
    mod.geometry = types.SimpleNamespace(
        PointCloud=PointCloud, LineSet=LineSet, TriangleMesh=TriangleMesh,
        KDTreeSearchParamKNN=KDTreeSearchParamKNN)
    mod.utility = types.SimpleNamespace(Vector3dVector=Vector3dVector,
                                        Vector2iVector=Vector2iVector)
    sys.modules["open3d"] = mod
    return mod
