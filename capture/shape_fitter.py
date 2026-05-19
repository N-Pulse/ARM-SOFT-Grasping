"""
capture/shape_fitter.py
=======================
Reusable primitive-fitting module.

Exports
-------
ShapeTracker          — temporal-convergence tracker (EMA + streak/lock)
fit_and_track(pts, table_normal, tracker)
                      — one-frame entry point; returns (shape_str, LineSet)

Pipeline (per frame)
--------------------
  ① SOR                →  remove stray / background points
  ② Normals            →  per-point surface normals, oriented toward camera
  ③ Classify           →  "cylinder" | "cuboid"
        Two-cluster split by surface-normal orientation vs. table axis:

        Cluster A (parallel to chessboard)
          Points whose normals are nearly aligned with the table normal
          (|n · axis| > _CAP_NORMAL_DOT) — the top/bottom face.
          Project onto the horizontal plane:
            rect_score ≥ _RECT_SCORE_CUBOID or right-angle corners → cuboid
            rect_score ≤ _RECT_SCORE_CYLINDER                       → cylinder

        Cluster B (perpendicular to chessboard)
          Points whose normals are nearly perpendicular to the table normal
          (|n · axis| < _PERP_DOT_THRESH) — the lateral / side walls.
          Project onto the horizontal plane (footprint):
            rect_score ≥ _RECT_SCORE_CUBOID or right-angle corners → cuboid evidence

        Decision: any cuboid evidence (A or B) → cuboid
                  cylinder evidence from A, no cuboid evidence   → cylinder
                  neither conclusive → fallback

        Fallback : lateral-normal azimuth entropy  +  full-cloud rect_score
  ④ Best fit           →  2D circle in cross-section plane (cylinder only)
                           axis_pt, radius, mean_err
  ⑤ Height             →  percentile extents + end-cap refinement
  ⑥ Tracker.update()   →  EMA blend into running estimate
  ⑦ Build wireframe    →  cylinder : TriangleMesh → LineSet
                           cuboid   : percentile-rect footprint → 12-edge LineSet

Tracker tuning  (do not change without user approval)
--------------
  ALPHA      EMA base learning rate          (0.20)
  N_LOCK     Frames before shape locks       (6)
  N_UNLOCK   Opposing frames to break lock   (20)

Wireframe colours
-----------------
  cylinder → orange   [1.0, 0.5, 0.0]
  cuboid   → lavender [0.8, 0.6, 1.0]
"""

from __future__ import annotations

import time
import cv2
import numpy as np
import open3d as o3d


# ── Colours ────────────────────────────────────────────────────────────────────
_COLOR = {
    "cylinder": [1.0, 0.5, 0.0],
    "cuboid":   [0.8, 0.6, 1.0],
}

# ── Normal estimation ──────────────────────────────────────────────────────────
_KNN_NORMAL = 10   # reduced from 15 — sufficient after voxel downsampling

# ── Geometry ───────────────────────────────────────────────────────────────────
_R_MIN          = 0.005
_R_MAX          = 2.0
_CAP_NORMAL_DOT = 0.7    # normal·axis above this → end-cap / top-face point
_CAP_MIN_PTS    = 10     # minimum cap points for height refinement

# ── 2-D corner detection ──────────────────────────────────────────────────────
_CORNER_ANGLE_MAX   = 120.0   # hull vertex interior angle ≤ this → rigid corner
_MIN_CORNERS_CUBOID = 2       # ≥ this many rigid corners → cuboid evidence

# ── Two-cluster classifier ────────────────────────────────────────────────────
# Cluster A — normals parallel to table axis (|n·axis| > _CAP_NORMAL_DOT)
#             → top / bottom face points
#   rect_score = convex-hull area / minAreaRect area  (bird's-eye projection)
#     square  → ~1.00,   circle → ~π/4 ≈ 0.785
_TOP_FACE_MIN_PTS    = 30     # minimum top-face points to trust cluster A
_RECT_SCORE_CUBOID   = 0.88   # rect_score ≥ this → rectangular  → cuboid
_RECT_SCORE_CYLINDER = 0.83   # rect_score ≤ this → circular     → cylinder
                               # gap 0.83–0.88 → ambiguous in cluster A

# Cluster B — normals perpendicular to table axis (|n·axis| < _PERP_DOT_THRESH)
#             → lateral / side-wall points
#   Project onto horizontal plane; right-angle corners or high rect_score → cuboid
_PERP_DOT_THRESH     = 0.50   # |n·axis| below this → side-wall cluster
_PERP_MIN_PTS        = 20     # minimum side-wall points to use cluster B

# ── Lateral-normal entropy (fallback classifier) ──────────────────────────────
_NORMAL_HIST_BINS      = 36    # 10° per bin
_NORMAL_CLUSTER_THRESH = 0.35  # normalised entropy < this → clustered → cuboid

# ── Voxel downsampling  (FIRST step — runs before everything else) ────────────
_VOXEL_SIZE = 0.005   # 5 mm grid — keeps shape detail, reduces N by ~5-10×
                       # Raise to 0.008 for faster but coarser fit;
                       # Lower to 0.003 for denser clouds / larger objects.

# ── Statistical outlier removal ───────────────────────────────────────────────
_SOR_NEIGHBORS = 20    # neighbours used for mean-distance statistics
_SOR_STD_RATIO = 2.5   # remove points > this many σ above the mean
                        # (2.5 keeps edge/corner points; 2.0 was too aggressive)

# ── DBSCAN clustering ─────────────────────────────────────────────────────────
_DBSCAN_EPS        = 0.012   # neighbourhood radius (m) — ~12 mm covers typical
                              # point spacing while bridging small surface gaps
_DBSCAN_MIN_PTS    = 5       # minimum cluster size (smaller after downsampling)

# ── Tracker ────────────────────────────────────────────────────────────────────
_ALPHA    = 0.20   # EMA base learning rate


# ══════════════════════════════════════════════════════════════════════════════
# Cluster extraction
# ══════════════════════════════════════════════════════════════════════════════

def _largest_cluster(pts: np.ndarray,
                     eps: float = _DBSCAN_EPS,
                     min_pts: int = _DBSCAN_MIN_PTS) -> np.ndarray:
    """
    Return the cluster closest to the camera among all DBSCAN clusters.

    When multiple clusters exist (e.g. object + table fragment), always prefer
    the one with the smallest mean depth (z) — i.e. closest to the camera.
    This gives a stable, deterministic choice that never jumps between clusters.

    Falls back to the full input when every point is labelled noise (-1).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    labels = np.asarray(pcd.cluster_dbscan(
        eps=eps, min_points=min_pts, print_progress=False))

    unique = np.unique(labels[labels >= 0])
    if len(unique) == 0:
        return pts   # all noise — fall back

    # Sort all clusters by size (descending), take the top 2, then among
    # those pick the one closest to the camera (smallest mean z).
    # This avoids jumping: a tiny stray fragment near the camera is never
    # preferred over the actual object, but if two large clusters compete
    # the nearer one always wins consistently.
    by_size   = sorted(unique, key=lambda lbl: (labels == lbl).sum(), reverse=True)
    top2      = by_size[:2]
    best_label = min(top2, key=lambda lbl: pts[labels == lbl, 2].mean())
    kept = pts[labels == best_label]

    n_removed = len(pts) - len(kept)

    n_removed = len(pts) - len(kept)
    if n_removed > 0:
        print(f"[shape_fitter]  cluster: kept {len(kept)}/{len(pts)} pts "
              f"(removed {n_removed} from {labels.max()+1} cluster(s))")
    return kept
_N_LOCK   = 6      # consecutive agreeing frames to lock shape
_N_UNLOCK = 20     # consecutive opposing frames to break the lock


# ══════════════════════════════════════════════════════════════════════════════
# 2-D geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _detect_2d_corners(pts_2d: np.ndarray,
                        angle_max: float = _CORNER_ANGLE_MAX) -> np.ndarray:
    """
    Find rigid-corner vertices on the convex hull of a 2-D point set.

    A vertex is a rigid corner if its interior angle ≤ angle_max degrees.
    A perfect rectangle has corners ≈ 90°; a cylinder's convex hull has none.

    Returns (M, 2) array of corner positions, or empty (0, 2) array.
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


def _rect_score_2d(pts_2d: np.ndarray) -> float:
    """convex-hull area / minAreaRect area.  ~1.0 for square, ~0.785 for circle."""
    pts_f     = pts_2d.astype(np.float32).reshape(-1, 1, 2)
    hull_area = float(cv2.contourArea(cv2.convexHull(pts_f)))
    _, (wm, hm), _ = cv2.minAreaRect(pts_f)
    return hull_area / (wm * hm + 1e-6)


def _fit_rect_percentile(pts_2d: np.ndarray,
                          lo: float = 0.0,
                          hi: float = 100.0) -> np.ndarray:
    """
    Fit a tight rectangle to pts_2d.

    Uses cv2.minAreaRect for orientation (robust to partial visibility), then
    min/max projections for extents so the wireframe covers the full point
    cloud (SOR already removed noise, so we can trust the extremes).

    Returns (4, 2) corners in CCW angular order.
    """
    pts_f = pts_2d.astype(np.float32).reshape(-1, 1, 2)
    _, _, angle = cv2.minAreaRect(pts_f)

    rad = np.deg2rad(angle)
    ax1 = np.array([ np.cos(rad), np.sin(rad)])
    ax2 = np.array([-np.sin(rad), np.cos(rad)])

    p1 = pts_2d @ ax1
    p2 = pts_2d @ ax2
    mn1, mx1 = float(np.percentile(p1, lo)), float(np.percentile(p1, hi))
    mn2, mx2 = float(np.percentile(p2, lo)), float(np.percentile(p2, hi))

    corners = np.array([
        mn1 * ax1 + mn2 * ax2,
        mx1 * ax1 + mn2 * ax2,
        mx1 * ax1 + mx2 * ax2,
        mn1 * ax1 + mx2 * ax2,
    ])
    ctr    = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - ctr[1], corners[:, 0] - ctr[0])
    return corners[np.argsort(angles)]


# ══════════════════════════════════════════════════════════════════════════════
# Classifier
# ══════════════════════════════════════════════════════════════════════════════

def _normal_cluster_score(normals: np.ndarray, axis: np.ndarray) -> float:
    """
    Normalised Shannon entropy of the lateral-normal azimuth distribution.
      · ≈ 1.0  →  uniform azimuth distribution  →  cylinder
      · ≈ 0.0  →  normals cluster in 2–4 bins    →  cuboid
    Returns 1.0 (inconclusive → assume cylinder) when too few lateral
    normals are visible (e.g. top-down view of the end cap).
    """
    axial    = np.abs(normals @ axis)
    lat_mask = axial < 0.70
    if lat_mask.sum() < 10:
        return 1.0

    lat_n = normals[lat_mask]
    horiz = lat_n - np.outer(lat_n @ axis, axis)
    norms = np.linalg.norm(horiz, axis=1)
    good  = norms > 0.30
    if good.sum() < 10:
        return 1.0

    horiz_n = horiz[good] / norms[good, None]
    ref = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 else np.array([0., 1., 0.])
    e1  = np.cross(axis, ref);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(axis, e1)
    angles = np.arctan2(horiz_n @ e2, horiz_n @ e1)

    hist, _ = np.histogram(angles, bins=_NORMAL_HIST_BINS, range=(-np.pi, np.pi))
    p       = hist.astype(float) / (hist.sum() + 1e-8)
    entropy = -float(np.sum(p * np.log(p + 1e-10)))
    return entropy / np.log(_NORMAL_HIST_BINS)


def _classify(pts: np.ndarray, normals: np.ndarray, axis: np.ndarray) -> str:
    """
    Classify the isolated object as 'cylinder' or 'cuboid'.

    Two-cluster approach
    --------------------
    The point cloud is split by surface-normal orientation relative to the
    table axis into two groups, each projected onto the horizontal plane
    (bird's-eye view) for 2-D shape analysis.

    Cluster A — normals parallel to chessboard (|n·axis| > _CAP_NORMAL_DOT)
        Top / bottom face points.
          • rect_score ≥ _RECT_SCORE_CUBOID or right-angle corners → cuboid vote
          • rect_score ≤ _RECT_SCORE_CYLINDER                       → cylinder vote

    Cluster B — normals perpendicular to chessboard (|n·axis| < _PERP_DOT_THRESH)
        Lateral / side-wall points.  Their horizontal footprint is rectangular
        for a cuboid and circular for a cylinder.
          • rect_score ≥ _RECT_SCORE_CUBOID or right-angle corners → cuboid vote

    Decision
    --------
      Any cuboid vote                          → "cuboid"
      Cylinder vote from A, no cuboid votes   → "cylinder"
      Neither cluster conclusive              → fallback

    Fallback — lateral-normal entropy + full-cloud silhouette
    ---------------------------------------------------------
    Entropy of the lateral surface normals (cylinder → uniform azimuth;
    cuboid → 2–4 tight clusters) combined with the full silhouette
    rect_score and corner count.
    """
    ref = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 else np.array([0., 1., 0.])
    e1  = np.cross(axis, ref);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(axis, e1)

    cuboid_votes   = 0
    cylinder_votes = 0

    # ── Cluster A: top-face (normals ≈ parallel to table axis) ────────────
    top_mask = np.abs(normals @ axis) > _CAP_NORMAL_DOT
    if top_mask.sum() >= _TOP_FACE_MIN_PTS:
        top_pts = pts[top_mask]
        d       = top_pts - top_pts.mean(axis=0)
        top_2d  = np.column_stack([d @ e1, d @ e2])

        rs_a        = _rect_score_2d(top_2d)
        corners_a   = _detect_2d_corners(top_2d)
        n_corners_a = len(corners_a)

        print(f"[shape_fitter]  [A] top_pts={top_mask.sum():3d}  "
              f"rect={rs_a:.3f}  corners={n_corners_a}")

        if rs_a >= _RECT_SCORE_CUBOID or n_corners_a >= _MIN_CORNERS_CUBOID:
            cuboid_votes += 1
        elif rs_a <= _RECT_SCORE_CYLINDER:
            cylinder_votes += 1
        # else: ambiguous in A — let cluster B or fallback decide

    # ── Cluster B: side-wall (normals ≈ perpendicular to table axis) ───────
    side_mask = np.abs(normals @ axis) < _PERP_DOT_THRESH
    if side_mask.sum() >= _PERP_MIN_PTS:
        side_pts = pts[side_mask]
        d        = side_pts - side_pts.mean(axis=0)
        side_2d  = np.column_stack([d @ e1, d @ e2])

        rs_b        = _rect_score_2d(side_2d)
        corners_b   = _detect_2d_corners(side_2d)
        n_corners_b = len(corners_b)

        print(f"[shape_fitter]  [B] side_pts={side_mask.sum():3d}  "
              f"rect={rs_b:.3f}  corners={n_corners_b}")

        if rs_b >= _RECT_SCORE_CUBOID or n_corners_b >= _MIN_CORNERS_CUBOID:
            cuboid_votes += 1
        # Note: cylinder votes are NOT added from cluster B — the side-wall
        # footprint of a cylinder also projects as a somewhat circular ring,
        # so its negative signal is weak.  Cylinder verdict comes from A only.

    # ── Early decision ─────────────────────────────────────────────────────
    if cuboid_votes > 0:
        return "cuboid"
    if cylinder_votes > 0:
        return "cylinder"

    # ── Fallback: lateral-normal entropy + full-cloud silhouette ───────────
    d      = pts - pts.mean(axis=0)
    pts_2d = np.column_stack([d @ e1, d @ e2])

    entropy      = _normal_cluster_score(normals, axis)
    is_clustered = entropy < _NORMAL_CLUSTER_THRESH
    rs_full      = _rect_score_2d(pts_2d)
    corners_full = _detect_2d_corners(pts_2d)
    has_corners  = len(corners_full) >= _MIN_CORNERS_CUBOID

    print(f"[shape_fitter]  [fallback] entropy={entropy:.3f}  "
          f"rect={rs_full:.3f}  corners={len(corners_full)}")

    if is_clustered or has_corners or rs_full >= _RECT_SCORE_CUBOID:
        return "cuboid"
    return "cylinder"


# ══════════════════════════════════════════════════════════════════════════════
# Top-down classifier  (primary — no per-point normals required)
# ══════════════════════════════════════════════════════════════════════════════

def _classify_topdown(pts: np.ndarray,
                      axis: np.ndarray) -> tuple[str, float, int]:
    """
    Classify by projecting the full point cloud onto the chessboard plane
    (along ``axis`` = table_normal) and analysing the 2-D footprint.

    Compared with the normal-cluster approach, this:
      · requires no per-point surface normals
      · is robust to the 30° tilt of the D405 (projection direction is the
        table normal, not the camera axis, so the footprint is always overhead)
      · runs in < 2 ms for typical object clouds

    Returns
    -------
    (shape, rect_score, n_corners)
      shape : "cylinder" | "cuboid" | "unknown"
        "unknown" is returned when rect_score falls in the ambiguous gap
        (0.83 – 0.88) AND fewer than _MIN_CORNERS_CUBOID corners are found.
    """
    ref = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 \
          else np.array([0., 1., 0.])
    e1  = np.cross(axis, ref);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(axis, e1)

    d      = pts - pts.mean(axis=0)
    pts_2d = np.column_stack([d @ e1, d @ e2])

    rs        = _rect_score_2d(pts_2d)
    corners   = _detect_2d_corners(pts_2d)
    n_corners = len(corners)

    print(f"[shape_fitter]  [topdown]  pts={len(pts):4d}  "
          f"rect={rs:.3f}  corners={n_corners}")

    if rs >= _RECT_SCORE_CUBOID or n_corners >= _MIN_CORNERS_CUBOID:
        return "cuboid", rs, n_corners
    if rs <= _RECT_SCORE_CYLINDER:
        return "cylinder", rs, n_corners
    return "unknown", rs, n_corners


# ══════════════════════════════════════════════════════════════════════════════
# Cylinder geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _best_fit_cylinder(pts: np.ndarray,
                        normals: np.ndarray,
                        axis: np.ndarray):
    """
    Geometric 2-D circle fit in the plane ⊥ to axis.

    Two-stage:
      1. Algebraic least-squares for a robust initial estimate.
      2. Geometric refinement via scipy.optimize.least_squares that minimises
         the actual radial distance error  Σ (‖p_⊥ - centre‖ − r)².
         This is critical for partial arcs (~1/3 visible at 30°) where the
         algebraic fit is biased toward a larger, looser circle.

    Returns (axis_pt, radius, mean_radial_err) or (None, None, inf).
    """
    centroid = pts.mean(axis=0)
    ref = np.array([0., 0., 1.]) if abs(axis[2]) < 0.9 else np.array([1., 0., 0.])
    e1  = np.cross(axis, ref); e1 /= np.linalg.norm(e1)
    e2  = np.cross(axis, e1);  e2 /= np.linalg.norm(e2)
    d   = pts - centroid
    u, v = d @ e1, d @ e2

    # ── Stage 1: algebraic least-squares (fast initial estimate) ─────────────
    A  = np.column_stack([-2*u, -2*v, np.ones(len(u))])
    b  = -(u**2 + v**2)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    cu0, cv0, dv = x
    r_sq = cu0**2 + cv0**2 - dv
    if r_sq <= 0:
        return None, None, np.inf
    r0 = float(np.sqrt(r_sq))

    # ── Stage 2: geometric refinement ────────────────────────────────────────
    try:
        from scipy.optimize import least_squares as _lsq

        def _residuals(params):
            cu_, cv_, r_ = params
            dist = np.sqrt((u - cu_)**2 + (v - cv_)**2)
            return dist - abs(r_)

        res = _lsq(_residuals, [cu0, cv0, r0], method='lm',
                   max_nfev=200, ftol=1e-6, xtol=1e-6)
        cu, cv, r_fit = res.x
        r_fit = abs(r_fit)
        if r_fit > 0:
            cu0, cv0, r0 = cu, cv, r_fit
    except Exception:
        pass   # keep algebraic result if scipy unavailable or diverges

    r       = float(r0)
    axis_pt = centroid + cu0*e1 + cv0*e2
    along   = (pts - axis_pt) @ axis
    on_ax   = axis_pt + np.outer(along, axis)
    err     = float(np.mean(np.abs(np.linalg.norm(pts - on_ax, axis=1) - r)))
    return axis_pt, r, err


def _estimate_height(pts: np.ndarray,
                      normals: np.ndarray,
                      axis: np.ndarray,
                      centroid: np.ndarray):
    # Use full min/max — SOR already removed stray points so the extremes are
    # trustworthy.  Percentiles used to leave visible gaps at the top/bottom.
    proj  = (pts - centroid) @ axis
    h_min = float(proj.min())
    h_max = float(proj.max())
    cap   = np.abs(normals @ axis) > _CAP_NORMAL_DOT
    if cap.sum() >= _CAP_MIN_PTS:
        cp   = proj[cap];  span = h_max - h_min
        cm, cx = float(cp.min()), float(cp.max())
        if cm < h_min + 0.3 * span:
            h_min = cm
            print(f"[shape_fitter]  bottom cap {h_min*1e3:.0f}mm")
        if cx > h_max - 0.3 * span:
            h_max = cx
            print(f"[shape_fitter]  top cap    {h_max*1e3:.0f}mm")
    return h_min, h_max


# ══════════════════════════════════════════════════════════════════════════════
# Temporal convergence tracker  — DO NOT modify convergence constants
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
    """

    def __init__(self):
        self.orient    = None
        self._streak   = 0
        self._locked   = False
        self._flip_str = 0

        # Smoothed cylinder parameters
        self.axis    = None   # (3,) unit vector
        self.axis_pt = None   # (3,) point on axis
        self.radius  = None   # float (m)
        self.h_ctr   = None   # (h_min + h_max) / 2
        self.height  = None   # h_max - h_min

        # Shape voting — separate state so reset() does not clear it
        self.shape         = None   # "cylinder" | "cuboid" | None
        self._shape_streak = 0
        self._shape_locked = False
        self._shape_flip   = 0

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
        Blend one frame's raw cylinder fit into the running EMA estimate.

        Returns (axis, axis_pt, radius, h_min, h_max) — smoothed.
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
            self.h_ctr   = (1 - alpha) * self.h_ctr  + alpha * raw_h_ctr
            self.height  = (1 - alpha) * self.height + alpha * raw_h
            self.axis_pt = raw_axis_pt.copy()
            self.radius  = raw_r

        # Axis is always the chessboard table normal — no convergence drift
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

    def full_reset(self):
        """Full reset — clears cylinder parameters AND shape voting state.

        Call this when the object is lost (no detection for a frame) so that
        switching objects starts completely fresh with no residual commitment.
        """
        self.reset()
        self.shape         = None
        self._shape_streak = 0
        self._shape_locked = False
        self._shape_flip   = 0
        print("[shape_fitter]  *** full_reset() — shape vote cleared, "
              "ready for new object ***")


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

    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=r, height=height, resolution=20)
    mesh.rotate(R, center=np.zeros(3))
    mesh.translate(center)
    ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    ls.paint_uniform_color(_COLOR["cylinder"])
    return ls


def _build_cuboid(pts: np.ndarray, axis: np.ndarray,
                  normals: np.ndarray | None = None) -> o3d.geometry.LineSet:
    """
    Build a cuboid wireframe by fitting best-fit planes to each face.

    Approach — analogous to fitting a line through points, but in 3-D we fit a
    plane to each face group and intersect the 6 planes to get 8 corners:

      1. Establish 3 orthogonal face-normal directions:
           · vertical  = table_normal  (= axis)
           · horizontal h1, h2  from cv2.minAreaRect on the floor footprint
      2. For each face direction, split points into two face groups
         (lowest FACE_FRAC and highest FACE_FRAC of projected distances).
         If surface normals are available, additionally filter to points whose
         normal aligns with the expected face direction — this gives cleaner
         face membership, especially when two faces are partially occluded.
      3. Plane offset  = mean projection of the face group along its normal.
         This is the closed-form least-squares solution for a plane with
         known normal: d* = mean(p_i · n).
      4. 8 corners = all combinations of (lo/hi) × 3 axes.
         Edges = corner pairs that differ in exactly one axis bit (12 edges).
    """
    FACE_FRAC       = 0.20   # fraction of pts used as each face's candidates
    NORMAL_ALIGN_TH = 0.50   # surface-normal dot-product threshold for face membership

    # ── Orthonormal basis ─────────────────────────────────────────────────────
    n   = axis / np.linalg.norm(axis)
    ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    e1  = np.cross(n, ref);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(n, e1)

    # ── Horizontal footprint → minAreaRect for orientation ────────────────────
    mean_pt = pts.mean(axis=0)
    d       = pts - mean_pt
    u       = (d @ e1).astype(np.float32)
    v       = (d @ e2).astype(np.float32)
    pts_2d  = np.column_stack([u, v])
    pts_f   = pts_2d.astype(np.float32).reshape(-1, 1, 2)
    _, _, angle = cv2.minAreaRect(pts_f)
    rad = np.deg2rad(angle)
    h1  = np.cos(rad) * e1 + np.sin(rad) * e2    # horizontal face normal 1
    h2  = -np.sin(rad) * e1 + np.cos(rad) * e2   # horizontal face normal 2

    # ── Fit two planes per axis direction ─────────────────────────────────────
    def face_extents(face_n):
        """
        Return (d_lo, d_hi) — the two face-plane offsets along face_n.

        Each offset is the mean projection of that face's candidate points,
        i.e. the least-squares plane position given a fixed normal direction.
        """
        proj  = pts @ face_n
        n_pts = len(proj)
        n_k   = max(5, int(n_pts * FACE_FRAC))
        idx   = np.argsort(proj)
        lo_idx = idx[:n_k]
        hi_idx = idx[-n_k:]

        if normals is not None:
            # Tighten face membership using surface-normal alignment.
            # The face with outward normal  face_n → surface normals ≈ +face_n
            # The face with outward normal -face_n → surface normals ≈ -face_n
            lo_align = normals[lo_idx] @ (-face_n)   # should be positive for bottom face
            hi_align = normals[hi_idx] @   face_n    # should be positive for top face
            lo_good  = lo_idx[lo_align > NORMAL_ALIGN_TH]
            hi_good  = hi_idx[hi_align > NORMAL_ALIGN_TH]
            if len(lo_good) >= 3:
                lo_idx = lo_good
            if len(hi_good) >= 3:
                hi_idx = hi_good

        d_lo = float(proj[lo_idx].mean())
        d_hi = float(proj[hi_idx].mean())

        # Guard against degenerate (flat) case
        if d_hi - d_lo < 0.005:
            mid  = (d_lo + d_hi) / 2.0
            d_lo, d_hi = mid - 0.0025, mid + 0.0025

        return d_lo, d_hi

    v_lo,  v_hi  = face_extents(n)
    h1_lo, h1_hi = face_extents(h1)
    h2_lo, h2_hi = face_extents(h2)

    # ── Force equal side lengths (cube) ──────────────────────────────────────
    # Take the largest fitted extent across all three axes and apply it
    # symmetrically around each face-pair centre.
    v_size  = v_hi  - v_lo
    h1_size = h1_hi - h1_lo
    h2_size = h2_hi - h2_lo
    side    = max(v_size, h1_size, h2_size)

    v_ctr   = (v_lo  + v_hi)  / 2.0
    h1_ctr  = (h1_lo + h1_hi) / 2.0
    h2_ctr  = (h2_lo + h2_hi) / 2.0

    v_lo,  v_hi  = v_ctr  - side / 2, v_ctr  + side / 2
    h1_lo, h1_hi = h1_ctr - side / 2, h1_ctr + side / 2
    h2_lo, h2_hi = h2_ctr - side / 2, h2_ctr + side / 2

    # ── Build 8 corners ───────────────────────────────────────────────────────
    # Corner index i encodes: bit0 = v axis (0=lo,1=hi)
    #                          bit1 = h1 axis
    #                          bit2 = h2 axis
    v_vals  = [v_lo,  v_hi ]
    h1_vals = [h1_lo, h1_hi]
    h2_vals = [h2_lo, h2_hi]
    verts   = np.array([
        v_vals[i & 1] * n + h1_vals[(i >> 1) & 1] * h1 + h2_vals[(i >> 2) & 1] * h2
        for i in range(8)
    ])

    # Edges: connect any two corners that differ in exactly one axis (12 edges)
    edges = [[i, j]
             for i in range(8)
             for j in range(i + 1, 8)
             if bin(i ^ j).count('1') == 1]

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(verts)
    ls.lines  = o3d.utility.Vector2iVector(edges)
    ls.paint_uniform_color(_COLOR["cuboid"])
    return ls


# ══════════════════════════════════════════════════════════════════════════════
# Simple single-frame fit  (no tracker, no voting, no EMA)
# ══════════════════════════════════════════════════════════════════════════════

def fit_once(pts: np.ndarray,
             table_normal,
             shape_hint: str | None = None
             ) -> tuple[str | None, "o3d.geometry.LineSet | None"]:
    """
    Classify and fit the object in a single frame with no temporal state.

    Classification priority
    -----------------------
    1. ``shape_hint``       — direct YOLO result; used as-is
    2. ``_classify_topdown``— bird's-eye projection along table_normal
    3. ``"cuboid"``         — fallback when topdown is "unknown"

    Parameters
    ----------
    pts          : (N, 3) isolated object point cloud
    table_normal : (3,) unit normal of the table plane, or None
    shape_hint   : "cylinder" | "cuboid" | None

    Returns
    -------
    (shape_name, LineSet) or (None, None) if pts is too sparse after SOR
    """
    if len(pts) < 20:
        return None, None

    if shape_hint is None:
        return None, None   # no YOLO result → do not fit

    _t0 = time.perf_counter()

    # ── ① Voxel downsample ────────────────────────────────────────────────
    _pcd = o3d.geometry.PointCloud()
    _pcd.points = o3d.utility.Vector3dVector(pts)
    _pcd = _pcd.voxel_down_sample(_VOXEL_SIZE)
    pts = np.asarray(_pcd.points)
    _t1 = time.perf_counter()

    if len(pts) < 20:
        return None, None

    # ── ② Lightweight SOR ────────────────────────────────────────────────
    _pcd, _ = _pcd.remove_statistical_outlier(
        nb_neighbors=_SOR_NEIGHBORS, std_ratio=_SOR_STD_RATIO)
    pts = np.asarray(_pcd.points)
    _t2 = time.perf_counter()

    if len(pts) < 20:
        return None, None

    # ── ③ Main cluster only (on the small downsampled cloud — fast) ───────
    pts = _largest_cluster(pts)
    _t3 = time.perf_counter()

    if len(pts) < 20:
        return None, None

    axis  = table_normal if table_normal is not None else np.array([0., 0., 1.])
    shape = shape_hint
    print(f"[fit_once]  {shape}  pts={len(pts)}  "
          f"vox={(_t1-_t0)*1e3:.1f}ms  sor={(_t2-_t1)*1e3:.1f}ms  "
          f"dbscan={(_t3-_t2)*1e3:.1f}ms")

    # ── ④ Surface normals ─────────────────────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=_KNN_NORMAL))
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0., 0., 0.]))
    normals = np.asarray(pcd.normals)
    _t4 = time.perf_counter()
    print(f"[fit_once]  normals={(_t4-_t3)*1e3:.1f}ms")

    # ── Fit ────────────────────────────────────────────────────────────────
    if shape == "cuboid":
        ls = _build_cuboid(pts, axis, normals)
        _t5 = time.perf_counter()
        print(f"[fit_once]  cuboid fit={(_t5-_t4)*1e3:.1f}ms  "
              f"total={(_t5-_t0)*1e3:.1f}ms")
        return "cuboid", ls

    # Cylinder
    axis_pt, r, err = _best_fit_cylinder(pts, normals, axis)
    _t5 = time.perf_counter()
    if axis_pt is None:
        return "cuboid", _build_cuboid(pts, axis, normals)

    r        = float(np.clip(r, _R_MIN, _R_MAX))
    centroid = pts.mean(axis=0)
    h_min, h_max = _estimate_height(pts, normals, axis, centroid)
    ls = _build_cylinder(axis, axis_pt, r, h_min, h_max)
    _t6 = time.perf_counter()
    print(f"[fit_once]  cylinder r={r*1e3:.1f}mm h={(h_max-h_min)*1e3:.1f}mm "
          f"err={err*1e3:.2f}mm  circle_fit={(_t5-_t4)*1e3:.1f}ms  "
          f"total={(_t6-_t0)*1e3:.1f}ms")

    return "cylinder", ls


# ══════════════════════════════════════════════════════════════════════════════
# Per-frame entry point
# ══════════════════════════════════════════════════════════════════════════════

def fit_and_track(pts: np.ndarray, table_normal, tracker: ShapeTracker,
                  shape_hint: str | None = None):
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

    Classification pipeline
    -----------------------
    ① _classify_topdown  (primary — no per-point normals, ~2 ms)
         Project all pts onto the chessboard plane along table_normal;
         analyse the 2-D footprint with rect_score + corner count.
         Returns "cylinder" | "cuboid" | "unknown".

    ② _classify          (fallback — only when topdown is "unknown", ~25 ms)
         Normal-cluster approach: split into cap / side-wall groups,
         analyse each projection separately.

    Performance fast-paths
    ----------------------
    locked cuboid              → SOR + topdown check, skip normals  (~5 ms)
    locked cylinder            → SOR + topdown + normals, skip classify (~15 ms)
    topdown confident cuboid   → SOR + topdown, skip normals (~7 ms)
    topdown confident cylinder → SOR + topdown + normals for height  (~17 ms)
    topdown "unknown"          → SOR + topdown + normals + fallback  (~27 ms)
    """
    if len(pts) < 50:
        return None, None

    # ── ① Voxel downsample ────────────────────────────────────────────────
    _pcd = o3d.geometry.PointCloud()
    _pcd.points = o3d.utility.Vector3dVector(pts)
    _pcd = _pcd.voxel_down_sample(_VOXEL_SIZE)
    pts = np.asarray(_pcd.points)
    if len(pts) < 20:
        return None, None

    # ── ② Lightweight SOR ─────────────────────────────────────────────────
    _pcd, _ = _pcd.remove_statistical_outlier(
        nb_neighbors=_SOR_NEIGHBORS, std_ratio=_SOR_STD_RATIO)
    pts = np.asarray(_pcd.points)
    if len(pts) < 20:
        return None, None

    # ── ③ Main cluster only ───────────────────────────────────────────────
    pts = _largest_cluster(pts)
    if len(pts) < 20:
        return None, None

    _fallback_axis = table_normal if table_normal is not None \
                     else np.array([0., 0., 1.])

    # ── Fast path A: locked cuboid ────────────────────────────────────────
    if tracker._shape_locked and tracker.shape == "cuboid":
        # Reuse the already-downsampled _pcd from step ①
        _pcd.points = o3d.utility.Vector3dVector(pts)
        _pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=_KNN_NORMAL))
        _pcd.orient_normals_towards_camera_location(
            camera_location=np.array([0., 0., 0.]))
        return "cuboid", _build_cuboid(pts, _fallback_axis,
                                        np.asarray(_pcd.normals))

    # ── YOLO hint (highest priority — skips all geometric classifiers) ────────
    if shape_hint is not None and not tracker._shape_locked:
        shape = tracker.vote_shape(shape_hint)
        print(f"[shape_fitter]  YOLO→{shape_hint}  committed={shape}  "
              f"streak={tracker._shape_streak}  locked={tracker._shape_locked}")
        if shape == "cuboid":
            tracker.reset()
            # normals computed below — use placeholder; will be recomputed
            pass   # fall through to normals block
        # cylinder branch: still needs normals for fitting → fall through

    # ── Primary classifier: topdown projection (no normals) ────────────────
    td_shape, td_rs, td_nc = _classify_topdown(pts, _fallback_axis)

    # Confident cuboid from topdown → vote immediately
    if shape_hint is None and td_shape == "cuboid" and \
            not (tracker._shape_locked and tracker.shape == "cylinder"):
        shape = tracker.vote_shape("cuboid")
        print(f"[shape_fitter]  topdown→cuboid  committed={shape}  "
              f"streak={tracker._shape_streak}  locked={tracker._shape_locked}")
        if shape == "cuboid":
            tracker.reset()
            pass   # fall through to normals block

    # ── Normals (needed for cylinder fitting and/or fallback classify) ─────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=_KNN_NORMAL))
    pcd.orient_normals_towards_camera_location(
        camera_location=np.array([0., 0., 0.]))
    normals = np.asarray(pcd.normals)

    # ── Fast path B: locked cylinder — skip classify ───────────────────────
    if tracker._shape_locked and tracker.shape == "cylinder":
        shape = "cylinder"
    elif shape_hint is not None:
        # YOLO already voted above; shape is already set — just re-read it
        shape = tracker.shape if tracker.shape is not None else shape_hint
    else:
        # Use topdown result if confident; otherwise fall back to normal-cluster
        if td_shape != "unknown":
            raw_shape = td_shape
        else:
            raw_shape = _classify(pts, normals, _fallback_axis)

        shape = tracker.vote_shape(raw_shape)
        print(f"[shape_fitter]  topdown={td_shape}(rect={td_rs:.3f})  "
              f"raw={raw_shape}  committed={shape}  "
              f"streak={tracker._shape_streak}  locked={tracker._shape_locked}")

    # ── Cuboid branch ──────────────────────────────────────────────────────
    if shape == "cuboid":
        tracker.reset()
        return "cuboid", _build_cuboid(pts, _fallback_axis, normals)

    # ── Cylinder branch ────────────────────────────────────────────────────
    axis = _fallback_axis.copy() if hasattr(_fallback_axis, "copy") \
           else np.array(_fallback_axis)

    axis_pt, r, err = _best_fit_cylinder(pts, normals, axis)
    if axis_pt is None:
        tracker.reset()
        return "cuboid", _build_cuboid(pts, axis, normals)

    r = float(np.clip(r, _R_MIN, _R_MAX))

    centroid = pts.mean(axis=0)
    h_min, h_max = _estimate_height(pts, normals, axis, centroid)

    if not tracker._shape_locked:
        print(f"[shape_fitter]  raw  r={r*1e3:.1f}mm  "
              f"h={(h_max-h_min)*1e3:.1f}mm  err={err*1e3:.2f}mm")

    if table_normal is not None:
        s_axis, s_pt, s_r, s_hmin, s_hmax = tracker.update(
            "vertical", axis, axis_pt, r, h_min, h_max, err, table_normal)
    else:
        s_axis, s_pt, s_r, s_hmin, s_hmax = axis, axis_pt, r, h_min, h_max

    return "cylinder", _build_cylinder(s_axis, s_pt, s_r, s_hmin, s_hmax)
