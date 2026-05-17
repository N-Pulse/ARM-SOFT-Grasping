"""
capture/shape_fitter.py
=======================
Reusable curvature-based primitive-fitting module.

Exports
-------
ShapeTracker          — temporal-convergence tracker (EMA + streak/lock)
fit_and_track(pts, table_normal, tracker)
                      — one-frame entry point; returns (shape_str, LineSet)

All heavy geometry helpers are kept private (leading underscore).

Pipeline (per frame)
--------------------
  ① Normals + curvature  →  κ₁, κ₂, raw_axis
  ② Classify             →  "cylinder" | "cuboid"
        Primary  : lateral-normal azimuth entropy
        Secondary: 2-D rigid-angle corner detection
        Fallback : κ₂ curvature signal
  ③ Confirm orientation  →  aspect-ratio primary, curvature-angle fallback
  ④ Best fit             →  2D circle in cross-section plane
                             → axis_pt, radius, mean_err
  ⑤ Height               →  percentile extents + end-cap refinement
  ⑥ Tracker.update()     →  EMA blend of all params into running estimate
  ⑦ Build wireframe      →  cylinder : smoothed tracker params → TriangleMesh
                             cuboid   : detected hull corners + mirroring
                                        → 8 vertices, 12-edge LineSet

Tracker tuning
--------------
  ALPHA      EMA base learning rate  (0.2)
  N_LOCK     Consecutive frames before orientation / shape locks (6)
  N_UNLOCK   Consecutive opposing frames needed to break the lock (20)

Wireframe colours
-----------------
  cylinder → orange   [1.0, 0.5, 0.0]
  cuboid   → lavender [0.8, 0.6, 1.0]
"""

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# ── Colours ────────────────────────────────────────────────────────────────────
_COLOR = {
    "cylinder": [1.0, 0.5, 0.0],
    "cuboid":   [0.8, 0.6, 1.0],
}

# ── Curvature ──────────────────────────────────────────────────────────────────
_KNN_NORMAL   = 30
_KNN_CURV     = 25
_SUBSAMPLE    = 600
_FLAT_THRESH  = 2.0         # |κ| < this (m⁻¹) → zero curvature
_ANISO_RATIO  = 5.0

# ── Geometry ───────────────────────────────────────────────────────────────────
_R_MIN            = 0.005
_R_MAX            = 2.0
_CAP_NORMAL_DOT   = 0.7
_CAP_MIN_PTS      = 10
_MAX_RADIAL_ERR   = 0.020   # m — raw fits with error > this are downweighted

# ── Orientation decision ────────────────────────────────────────────────────────
_ASPECT_VERTICAL   = 1.5    # extent_up / extent_wide > this → vertical
_ASPECT_HORIZONTAL = 0.85   # extent_up / extent_wide < this → horizontal
_SNAP_THRESH_DEG   = 45.0   # curvature-angle fallback threshold

# ── Edge / vertex detection ───────────────────────────────────────────────────
_EDGE_KNN              = 12    # kNN for normal-coherence edge scoring
_EDGE_COHERENCE_THRESH = 0.60  # normal coherence below this → edge/corner point
_EDGE_FRAC_CUBOID      = 0.06  # > 6 % edge points → strong cuboid signal
_CORNER_ANGLE_MAX      = 120.0 # convex-hull vertex interior angle ≤ this → rigid corner
_MIN_CORNERS_CUBOID    = 1     # even a single detected corner is strong cuboid evidence

# ── Normal-distribution classifier ────────────────────────────────────────────
# A cylinder's lateral normals fan out uniformly in all azimuths around the axis
# (entropy ≈ max).  A cuboid's lateral normals collapse into 2–4 tight clusters
# (one per visible flat face), leaving large empty sectors and low entropy.
_NORMAL_HIST_BINS      = 36    # 10 ° per bin
_NORMAL_CLUSTER_THRESH = 0.35  # normalised entropy below this → clustered → cuboid

# ── Tracker ────────────────────────────────────────────────────────────────────
_ALPHA     = 0.20           # EMA base learning rate
_N_LOCK    = 6              # frames before orientation / shape locks
_N_UNLOCK  = 20             # opposing frames needed to break the lock


# ══════════════════════════════════════════════════════════════════════════════
# STAGE ① — Local curvature
# ══════════════════════════════════════════════════════════════════════════════

def _fit_local_curvature(p, n, neighbours):
    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    t1  = np.cross(n, ref); t1 /= np.linalg.norm(t1)
    t2  = np.cross(n, t1);  t2 /= np.linalg.norm(t2)
    d   = neighbours - p
    u, v, h = d @ t1, d @ t2, d @ n
    A   = np.column_stack([u**2, u*v, v**2])
    if np.linalg.matrix_rank(A) < 3:
        return None
    (a, b, c), *_ = np.linalg.lstsq(A, h, rcond=None)
    II            = np.array([[2*a, b], [b, 2*c]])
    evals, evecs  = np.linalg.eigh(II)
    ev            = evecs[:, 0]
    return float(evals[0]), float(evals[1]), ev[0]*t1 + ev[1]*t2


def _aggregate_curvatures(pts, normals):
    tree = cKDTree(pts)
    k1s, k2s, axs = [], [], []
    for i in range(len(pts)):
        _, idx = tree.query(pts[i], k=_KNN_CURV + 1)
        r = _fit_local_curvature(pts[i], normals[i], pts[idx[1:]])
        if r is None:
            continue
        k1s.append(r[0]); k2s.append(r[1]); axs.append(r[2])
    if not k1s:
        return 0., 0., None
    k1, k2 = float(np.median(k1s)), float(np.median(k2s))
    axes   = np.array(axs)
    signs  = np.sign(axes @ axes[0]); signs[signs == 0] = 1
    axes  *= signs[:, None]
    m      = axes.mean(axis=0); nrm = np.linalg.norm(m)
    return k1, k2, m/nrm if nrm > 1e-6 else None


# ══════════════════════════════════════════════════════════════════════════════
# Edge / vertex detection helpers
# ══════════════════════════════════════════════════════════════════════════════

def _edge_mask(pts: np.ndarray, normals: np.ndarray,
               k: int = _EDGE_KNN) -> np.ndarray:
    """
    Return bool mask (N,) where True = edge or corner point.

    Normal coherence = |mean of k unit normals|  ∈ [0, 1].
      · Interior of a flat face   : normals nearly identical  → coherence ≈ 1
      · Edge between two faces    : normals split two ways    → coherence < 1
      · Corner (three faces meet) : normals spread three ways → coherence ≪ 1

    Points whose coherence < _EDGE_COHERENCE_THRESH are flagged as edges.
    """
    tree  = cKDTree(pts)
    score = np.zeros(len(pts))
    for i in range(len(pts)):
        _, idx   = tree.query(pts[i], k=k + 1)
        mean_n   = normals[idx[1:]].mean(axis=0)
        score[i] = 1.0 - np.linalg.norm(mean_n)   # 0 = flat-face, 1 = random
    return score > (1.0 - _EDGE_COHERENCE_THRESH)


def _detect_2d_corners(pts_2d: np.ndarray,
                       angle_max: float = _CORNER_ANGLE_MAX) -> np.ndarray:
    """
    Find rigid-corner vertices on the convex hull of a 2-D point set.

    A vertex is a rigid corner if its interior angle ≤ angle_max degrees.
    For a perfect rectangle every hull corner is ≈ 90°; noise may push it to
    ~100°.  A cylinder's convex hull has no such sharp interior angles.

    Returns (M, 2) array of detected corner positions, or empty array.
    """
    if len(pts_2d) < 4:
        return np.empty((0, 2))
    pts_f = pts_2d.astype(np.float32).reshape(-1, 1, 2)
    hull  = cv2.convexHull(pts_f, returnPoints=True).reshape(-1, 2).astype(float)
    n     = len(hull)
    if n < 3:
        return np.empty((0, 2))

    corners = []
    for i in range(n):
        prev_ = hull[(i - 1) % n]
        curr  = hull[i]
        next_ = hull[(i + 1) % n]
        v1 = prev_ - curr;  n1 = np.linalg.norm(v1)
        v2 = next_ - curr;  n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        angle = np.degrees(np.arccos(np.clip((v1 / n1) @ (v2 / n2), -1.0, 1.0)))
        if angle <= angle_max:
            corners.append(curr)
    return np.array(corners) if corners else np.empty((0, 2))


def _complete_rectangle_2d(corners: np.ndarray,
                            pts_2d: np.ndarray) -> np.ndarray:
    """
    Return exactly 4 rectangle corners in CCW angular order.

    M == 4  → the four detected corners are already known; order and return.
    M other → fall back to cv2.minAreaRect of the full projected cloud.

    Note: partial-completion heuristics (parallelogram rule for M==3, edge
    extension for M==2) were removed because they require the detected corners
    to be exactly consecutive rectangle vertices — an assumption that breaks
    silently with noisy point clouds and produces skewed wireframes.
    minAreaRect is more robust when fewer than 4 clean corners are found.
    """
    def _ccw(c: np.ndarray) -> np.ndarray:
        ctr    = c.mean(axis=0)
        angles = np.arctan2(c[:, 1] - ctr[1], c[:, 0] - ctr[0])
        return c[np.argsort(angles)]

    if len(corners) == 4:
        return _ccw(corners)

    # Fallback: minimum-area bounding rectangle of the full point cloud
    pts_f = pts_2d.astype(np.float32).reshape(-1, 1, 2)
    box   = cv2.boxPoints(cv2.minAreaRect(pts_f)).astype(float)
    return _ccw(box)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE ② — Classification
# ══════════════════════════════════════════════════════════════════════════════

def _normal_cluster_score(normals: np.ndarray, axis: np.ndarray) -> float:
    """
    Measure how clustered the lateral surface normals are around the axis.

    Returns a normalised-entropy score in [0, 1]:
      · ≈ 1.0  uniform / smoothly varying  →  cylinder
      · ≈ 0.0  collapsed into discrete bins →  cuboid

    Algorithm
    ---------
    1. Discard cap normals (those nearly parallel to the axis).
    2. Project each remaining normal onto the plane ⊥ axis and normalise.
    3. Compute the azimuth angle of each projected normal.
    4. Build a histogram over [−π, π) with _NORMAL_HIST_BINS bins.
    5. Return the normalised Shannon entropy of that histogram.
    """
    axial    = np.abs(normals @ axis)
    lat_mask = axial < 0.70
    if lat_mask.sum() < 10:
        return 1.0   # too few lateral normals → inconclusive → assume cylinder

    lat_n  = normals[lat_mask]
    horiz  = lat_n - np.outer(lat_n @ axis, axis)
    norms  = np.linalg.norm(horiz, axis=1)
    good   = norms > 0.30
    if good.sum() < 10:
        return 1.0

    horiz_n = horiz[good] / norms[good, None]

    ref = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 else np.array([0., 1., 0.])
    e1  = np.cross(axis, ref);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(axis, e1)
    angles = np.arctan2(horiz_n @ e2, horiz_n @ e1)   # (M,) in [−π, π)

    hist, _ = np.histogram(angles, bins=_NORMAL_HIST_BINS, range=(-np.pi, np.pi))
    p       = hist.astype(float) / (hist.sum() + 1e-8)
    entropy = -float(np.sum(p * np.log(p + 1e-10)))
    return entropy / np.log(_NORMAL_HIST_BINS)         # normalise to [0, 1]


def _classify(k1: float, k2: float,
              pts: np.ndarray, normals: np.ndarray,
              axis: np.ndarray) -> str:
    """
    Classify the isolated object as 'cylinder' or 'cuboid'.

    Three independent signals are computed; the decision is conservative —
    "cylinder" requires ALL signals to agree, so a cube is hard to miss.

    Signals
    -------
    1. Normal-distribution entropy  (primary, most robust)
    2. 2-D rigid-corner detection   (geometric, axis-projected)
    3. Curvature anisotropy         (fallback, noisy for edge-heavy regions)

    Decision
    --------
    → "cylinder"  only when ALL three say "not cuboid"
    → "cuboid"    if ANY signal fires
    """
    # Signal 1: normal-distribution entropy
    norm_entropy = _normal_cluster_score(normals, axis)
    is_clustered = norm_entropy < _NORMAL_CLUSTER_THRESH

    # Signal 2: 2-D rigid corners
    ref = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 else np.array([0., 1., 0.])
    e1  = np.cross(axis, ref);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(axis, e1)
    d   = pts - pts.mean(axis=0)
    pts_2d     = np.column_stack([d @ e1, d @ e2])
    corners_2d = _detect_2d_corners(pts_2d)
    has_corners = len(corners_2d) >= _MIN_CORNERS_CUBOID

    # Signal 3: curvature
    curv_cylinder = abs(k2) >= _FLAT_THRESH

    print(f"[shape_fitter]  norm_entropy={norm_entropy:.3f}  "
          f"corners={len(corners_2d)}  curv_cyl={curv_cylinder}")

    if curv_cylinder and not is_clustered and not has_corners:
        return "cylinder"
    return "cuboid"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE ③ — Confirm orientation (aspect ratio primary, curvature fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _confirm_orientation(pts, raw_axis, table_normal):
    """
    Returns ("vertical" | "horizontal", snapped_axis).

    PRIMARY — PCD bounding-box aspect ratio (robust to curvature noise):
      aspect = extent_along_table_normal / max_extent_in_table_plane
      > _ASPECT_VERTICAL   → vertical
      < _ASPECT_HORIZONTAL → horizontal

    FALLBACK — curvature axis angle (only when aspect is ambiguous).
    """
    d   = pts - pts.mean(axis=0)
    up  = float((d @ table_normal).ptp())

    ref = np.array([1., 0., 0.]) if abs(table_normal[0]) < 0.9 else np.array([0., 1., 0.])
    p1  = np.cross(table_normal, ref); p1 /= np.linalg.norm(p1)
    p2  = np.cross(table_normal, p1);  p2 /= np.linalg.norm(p2)
    wide = float(max((d @ p1).ptp(), (d @ p2).ptp()))

    aspect = up / (wide + 1e-6)
    print(f"[shape_fitter]  up={up*1e3:.0f}mm  wide={wide*1e3:.0f}mm  "
          f"aspect={aspect:.2f}", end="")

    if aspect > _ASPECT_VERTICAL:
        orient = "vertical";    print("  → vertical (aspect)")
    elif aspect < _ASPECT_HORIZONTAL:
        orient = "horizontal";  print("  → horizontal (aspect)")
    else:
        cos_a  = abs(float(raw_axis @ table_normal))
        orient = "vertical" if cos_a >= np.cos(np.radians(_SNAP_THRESH_DEG)) \
                 else "horizontal"
        print(f"  → {orient} (curvature fallback)")

    if orient == "vertical":
        axis = table_normal.copy()
    else:
        axis = raw_axis - (raw_axis @ table_normal) * table_normal
        nrm  = np.linalg.norm(axis)
        axis = raw_axis if nrm < 1e-6 else axis / nrm

    if axis @ raw_axis < 0:
        axis = -axis
    return orient, axis


# ══════════════════════════════════════════════════════════════════════════════
# STAGE ④ — Best fit (2D circle) + height
# ══════════════════════════════════════════════════════════════════════════════

def _best_fit_cylinder(pts, normals, axis):
    """
    Least-squares 2D circle fit in the plane ⊥ to axis.
    Returns (axis_pt, radius, mean_radial_err) or (None, None, inf).
    """
    centroid = pts.mean(axis=0)
    ref = np.array([0., 0., 1.]) if abs(axis[2]) < 0.9 else np.array([1., 0., 0.])
    e1  = np.cross(axis, ref); e1 /= np.linalg.norm(e1)
    e2  = np.cross(axis, e1);  e2 /= np.linalg.norm(e2)
    d   = pts - centroid
    u, v = d @ e1, d @ e2
    A    = np.column_stack([-2*u, -2*v, np.ones(len(u))])
    b    = -(u**2 + v**2)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    cu, cv, dv = x
    r_sq = cu**2 + cv**2 - dv
    if r_sq <= 0:
        return None, None, np.inf
    r       = float(np.sqrt(r_sq))
    axis_pt = centroid + cu*e1 + cv*e2
    along   = (pts - axis_pt) @ axis
    on_ax   = axis_pt + np.outer(along, axis)
    err     = float(np.mean(np.abs(np.linalg.norm(pts - on_ax, axis=1) - r)))
    return axis_pt, r, err


def _estimate_height(pts, normals, axis, centroid):
    proj  = (pts - centroid) @ axis
    h_min = float(np.percentile(proj, 1))
    h_max = float(np.percentile(proj, 99))
    cap   = np.abs(normals @ axis) > _CAP_NORMAL_DOT
    if cap.sum() >= _CAP_MIN_PTS:
        cp = proj[cap]; span = h_max - h_min
        cm, cx = float(cp.min()), float(cp.max())
        if cm < h_min + 0.3*span:
            h_min = cm
            print(f"[shape_fitter]  bottom cap {h_min*1e3:.0f}mm")
        if cx > h_max - 0.3*span:
            h_max = cx
            print(f"[shape_fitter]  top cap    {h_max*1e3:.0f}mm")
    return h_min, h_max


# ══════════════════════════════════════════════════════════════════════════════
# STAGE ⑤ — Temporal convergence tracker
# ══════════════════════════════════════════════════════════════════════════════

class ShapeTracker:
    """
    Maintains a temporally-smoothed estimate of the cylinder parameters AND
    a voted commitment to the current shape type ("cylinder" | "cuboid").

    Shape voting
    ------------
    Per-frame classifier output is noisy.  The shape type is decided by a
    streak-and-lock mechanism:
      · After _N_LOCK consecutive frames agree on the same shape, the type
        is *locked* — the wireframe type stops flickering.
      · While locked, _N_UNLOCK consecutive opposing frames are required to
        flip — preventing noise-driven oscillation.
      · Until locked, the most recent raw vote is shown directly.

    Cylinder parameter smoothing
    ----------------------------
    Each call to update() blends one frame's raw fit into the running estimate
    via EMA.  reset() clears only these cylinder parameters; shape voting state
    is intentionally preserved so a transient cuboid frame doesn't erase the
    accumulated streak.

    Orientation locking
    -------------------
    After N_LOCK consecutive frames agree on "vertical" or "horizontal", the
    orientation is locked.  Unlocking requires N_UNLOCK consecutive opposing
    frames.
    """

    def __init__(self):
        self.orient    = None      # "vertical" | "horizontal" | None
        self._streak   = 0
        self._locked   = False
        self._flip_str = 0

        # Smoothed cylinder parameters
        self.axis    = None        # (3,) unit vector
        self.axis_pt = None        # (3,) point on axis
        self.radius  = None        # float (m)
        self.h_ctr   = None        # (h_min+h_max)/2
        self.height  = None        # h_max-h_min

        # Shape voting — separate state so reset() does not clear it
        self.shape          = None  # "cylinder" | "cuboid" | None
        self._shape_streak  = 0
        self._shape_locked  = False
        self._shape_flip    = 0

    # ── Shape voting ──────────────────────────────────────────────────────────

    def vote_shape(self, raw_shape: str) -> str:
        """
        Submit one frame's raw shape vote and return the currently committed
        shape type.

        Before the shape locks (< _N_LOCK consistent votes), the most recent
        raw vote is returned so the user sees the classifier converging.
        After locking, the committed type is returned until _N_UNLOCK
        consecutive opposing votes force a flip.
        """
        if self.shape is None:
            self.shape         = raw_shape
            self._shape_streak = 1
            self._shape_locked = False
            self._shape_flip   = 0
            return raw_shape

        if raw_shape == self.shape:
            self._shape_flip = 0
            if not self._shape_locked:
                self._shape_streak += 1
                if self._shape_streak >= _N_LOCK:
                    self._shape_locked = True
                    print(f"[shape_fitter]  *** shape LOCKED → {self.shape} ***")
        else:
            if self._shape_locked:
                self._shape_flip += 1
                if self._shape_flip >= _N_UNLOCK:
                    print(f"[shape_fitter]  *** shape UNLOCKED → {raw_shape} ***")
                    self.shape         = raw_shape
                    self._shape_locked = False
                    self._shape_streak = 1
                    self._shape_flip   = 0
            else:
                self.shape         = raw_shape
                self._shape_streak = 1
                self._shape_flip   = 0

        return self.shape

    # ── Cylinder parameter update ─────────────────────────────────────────────

    def update(self, raw_orient, raw_axis, raw_axis_pt, raw_r,
               raw_h_min, raw_h_max, raw_err, table_normal):
        """
        Parameters
        ----------
        raw_*        : estimates from the current frame
        raw_err      : mean radial error (m) — scales learning rate
        table_normal : used to force axis when orientation is locked vertical

        Returns
        -------
        (axis, axis_pt, radius, h_min, h_max)  — smoothed
        """
        alpha = float(np.clip(_ALPHA / (1.0 + raw_err * 30), 0.04, 0.35))

        raw_h_ctr = (raw_h_min + raw_h_max) / 2.0
        raw_h     = raw_h_max - raw_h_min

        if self.axis is None:
            self.axis    = table_normal.copy()
            self.axis_pt = raw_axis_pt.copy()
            self.radius  = raw_r
            self.h_ctr   = raw_h_ctr
            self.height  = raw_h
        else:
            self.h_ctr   = (1 - alpha)*self.h_ctr  + alpha*raw_h_ctr
            self.height  = (1 - alpha)*self.height + alpha*raw_h
            self.axis_pt = raw_axis_pt.copy()
            self.radius  = raw_r

        # Axis is ALWAYS the chessboard table normal — no convergence drift
        self.axis = table_normal.copy()

        h_min = self.h_ctr - self.height / 2.0
        h_max = self.h_ctr + self.height / 2.0
        return self.axis, self.axis_pt, self.radius, h_min, h_max

    def reset(self):
        """Reset cylinder parameters only; shape voting state is preserved."""
        self.orient    = None
        self._streak   = 0
        self._locked   = False
        self._flip_str = 0
        self.axis      = None
        self.axis_pt   = None
        self.radius    = None
        self.h_ctr     = None
        self.height    = None


# ══════════════════════════════════════════════════════════════════════════════
# Wireframe builders
# ══════════════════════════════════════════════════════════════════════════════

def _build_cylinder(axis, axis_pt, r, h_min, h_max):
    r      = float(np.clip(r, _R_MIN, _R_MAX))
    height = float(np.clip(h_max - h_min, 0.005, 5.0))
    center = axis_pt + axis * (h_min + h_max) / 2.0

    z  = np.array([0., 0., 1.])
    v  = np.cross(z, axis);  s = np.linalg.norm(v);  c = float(np.dot(z, axis))
    if s < 1e-6:
        R = np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R  = np.eye(3) + vx + vx @ vx * (1.0 - c) / (s**2)

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=height, resolution=20)
    mesh.rotate(R, center=np.zeros(3))
    mesh.translate(center)
    ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    ls.paint_uniform_color(_COLOR["cylinder"])
    return ls


def _build_cuboid(pts: np.ndarray, axis: np.ndarray) -> o3d.geometry.LineSet:
    """
    Build a tight cuboid wireframe using the known axis and visible vertices
    detected from the point cloud, then mirror to complete the rectangle.

    Algorithm
    ---------
    1. Project pts onto the plane ⊥ axis (the horizontal footprint).
    2. Find convex-hull vertices whose interior angle ≤ _CORNER_ANGLE_MAX.
    3. Complete the rectangle (4 found → use directly; else minAreaRect).
    4. Reconstruct 8 3D vertices and build the 12-edge LineSet.
    """
    n   = axis.copy()
    ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    e1  = np.cross(n, ref);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(n, e1)

    mean_pt = pts.mean(axis=0)
    d       = pts - mean_pt
    u       = (d @ e1).astype(np.float32)
    v       = (d @ e2).astype(np.float32)
    w       = pts @ n                          # absolute height along axis
    pts_2d  = np.column_stack([u, v])          # (N, 2), centred

    h_min = float(np.percentile(w, 1))
    h_max = float(np.percentile(w, 99))
    h_min = h_min if h_max - h_min > 0.005 else h_min - 0.0025

    if len(pts) > 500:
        idx    = np.random.choice(len(pts), 500, replace=False)
        sub_2d = pts_2d[idx]
    else:
        sub_2d = pts_2d

    corners_2d   = _detect_2d_corners(sub_2d)
    rect_corners = _complete_rectangle_2d(corners_2d, pts_2d)  # (4, 2) CCW

    mean_horiz = mean_pt - (mean_pt @ n) * n

    verts = []
    for (cu, cv) in rect_corners:
        horiz = mean_horiz + cu * e1 + cv * e2
        verts.append(horiz + h_min * n)   # bottom corner
        verts.append(horiz + h_max * n)   # top corner
    # Layout: [bot_0, top_0, bot_1, top_1, bot_2, top_2, bot_3, top_3]
    verts = np.array(verts)               # (8, 3)

    lines = []
    for i in range(4):
        j = (i + 1) % 4
        lines.append([2 * i,     2 * j    ])   # bottom ring edge i → j
        lines.append([2 * i + 1, 2 * j + 1])   # top    ring edge i → j
        lines.append([2 * i,     2 * i + 1])   # vertical pillar at corner i

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(verts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(_COLOR["cuboid"])
    return ls


# ══════════════════════════════════════════════════════════════════════════════
# Per-frame entry point
# ══════════════════════════════════════════════════════════════════════════════

def fit_and_track(pts: np.ndarray, table_normal, tracker: ShapeTracker):
    """
    Run one frame through the full pipeline and update the tracker.

    Parameters
    ----------
    pts          : (N, 3) isolated object point cloud
    table_normal : (3,) unit normal of the table plane, or None
    tracker      : ShapeTracker instance (shared across frames)

    Returns
    -------
    (shape_name, LineSet) or (None, None)
    """
    if len(pts) < 50:
        return None, None

    # Normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=_KNN_NORMAL))
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0., 0., 0.]))
    normals = np.asarray(pcd.normals)

    N = len(pts)
    if N > _SUBSAMPLE:
        idx    = np.random.choice(N, _SUBSAMPLE, replace=False)
        sp, sn = pts[idx], normals[idx]
    else:
        sp, sn = pts, normals

    # ① Curvature + classify + shape voting
    k1, k2, raw_axis = _aggregate_curvatures(sp, sn)
    _cls_axis = (table_normal.copy() if table_normal is not None
                 else (raw_axis if raw_axis is not None
                       else np.array([0., 0., 1.])))
    raw_shape = _classify(k1, k2, sp, sn, _cls_axis)

    if raw_axis is None:
        shape = "cuboid"
    else:
        shape = tracker.vote_shape(raw_shape)

    print(f"[shape_fitter]  κ₁={k1:+.2f}  κ₂={k2:+.2f}  "
          f"raw={raw_shape}  committed={shape}  "
          f"streak={tracker._shape_streak}  locked={tracker._shape_locked}")

    if shape == "cuboid":
        tracker.reset()
        _cub_axis = table_normal if table_normal is not None else _cls_axis
        return "cuboid", _build_cuboid(pts, _cub_axis)

    # ② Axis is always the chessboard table normal
    if table_normal is not None:
        axis = table_normal.copy()
    else:
        axis = raw_axis
    orient = "vertical"

    # ③ Best fit
    axis_pt, r, err = _best_fit_cylinder(pts, normals, axis)
    if axis_pt is None:
        tracker.reset()
        return "cuboid", _build_cuboid(pts, axis)

    r = float(np.clip(r, _R_MIN, _R_MAX))

    # ④ Height
    centroid = pts.mean(axis=0)
    h_min, h_max = _estimate_height(pts, normals, axis, centroid)

    print(f"[shape_fitter]  raw  r={r*1e3:.1f}mm  "
          f"h={(h_max-h_min)*1e3:.1f}mm  err={err*1e3:.2f}mm")

    # ⑤ Tracker blend
    if table_normal is not None:
        s_axis, s_pt, s_r, s_hmin, s_hmax = tracker.update(
            orient, axis, axis_pt, r, h_min, h_max, err, table_normal
        )
    else:
        s_axis, s_pt, s_r, s_hmin, s_hmax = axis, axis_pt, r, h_min, h_max

    print(f"[shape_fitter]  smooth r={s_r*1e3:.1f}mm  "
          f"h={(s_hmax-s_hmin)*1e3:.1f}mm  "
          f"locked={tracker._locked}  orient={tracker.orient}")

    ls = _build_cylinder(s_axis, s_pt, s_r, s_hmin, s_hmax)
    return "cylinder", ls
