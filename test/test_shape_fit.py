"""
test_shape_fit.py
=================
Curvature-based primitive fitting with temporal convergence.

Key design change vs previous version
--------------------------------------
Each frame used to fit independently → parameters jumped every frame.

Now a _ShapeTracker maintains a running smoothed estimate:

  · All parameters (axis, radius, center, height) are updated via EMA
    (Exponential Moving Average) — each new frame nudges the estimate,
    rather than replacing it.

  · Orientation (vertical / horizontal) is decided by the PCD aspect ratio
    (robust to curvature noise), then *locked* after N_LOCK consistent frames.
    Once locked, it takes N_UNLOCK consecutive opposing frames to flip —
    preventing noise-driven oscillation.

  · The confidence of each frame's fit (mean radial error) scales the EMA
    learning rate: a high-error frame contributes less than a clean one.

Pipeline
--------
  INIT    detect_table_plane()  →  table_normal

  PER FRAME
    ① Normals + curvature  →  κ₁, κ₂, raw_axis
    ② Classify             →  "cylinder" | "cuboid"
                              Primary  : κ₂ curvature signal
                              Secondary: rigid-angle vertex + straight-edge
                                         detection — cylinders have neither
    ③ Confirm orientation  →  aspect-ratio primary, curvature-angle fallback
    ④ Best fit             →  2D circle in cross-section plane
                              → axis_pt, radius, mean_err
    ⑤ Height               →  percentile extents + end-cap refinement
    ⑥ Tracker.update()     →  EMA blend of all params into running estimate
    ⑦ Build wireframe      →  cylinder : from smoothed tracker params
                              cuboid   : axis (table normal) + visible corners
                                         detected from PCD → mirror to complete
                                         the rectangle → 8 vertices

Tracker tuning
--------------
  ALPHA      EMA learning rate  (0.2 = 20% toward new frame each step)
             High-error frames get a lower effective alpha automatically.
  N_LOCK     Consecutive frames with same orientation before locking (6)
  N_UNLOCK   Consecutive opposing frames needed to break the lock (20)

Chessboard
----------
  11 columns × 8 rows  →  inner corners (10, 7),  square = 15 mm

Wireframe colours
-----------------
  cylinder → orange   [1.0, 0.5, 0.0]
  cuboid   → lavender [0.8, 0.6, 1.0]
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
    # "sphere":   [0.2, 0.8, 1.0],   # disabled
    "cylinder": [1.0, 0.5, 0.0],
    "cuboid":   [0.8, 0.6, 1.0],
}

# ── Chessboard ─────────────────────────────────────────────────────────────────
_BOARD_COLS   = 10          # inner corners  (11 col board  → 10)
_BOARD_ROWS   = 7           # inner corners  ( 8 row board  →  7)
_SQUARE_M     = 0.015       # 15 mm

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
_CORNER_ANGLE_MAX      = 105.0 # convex-hull vertex interior angle ≤ this → rigid corner
_MIN_CORNERS_CUBOID    = 2     # need ≥ this many detected corners to trust vertex method

# ── Tracker ────────────────────────────────────────────────────────────────────
_ALPHA     = 0.20           # EMA base learning rate
_N_LOCK    = 6              # frames before orientation locks
_N_UNLOCK  = 20             # opposing frames needed to break the lock


# ══════════════════════════════════════════════════════════════════════════════
# TABLE PLANE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_table_plane(board_cols=_BOARD_COLS, board_rows=_BOARD_ROWS):
    """
    Stream RealSense frames until chessboard is found; fit plane via SVD.
    Returns (table_normal, d)  or  (None, None) on ESC.
    """
    board_shape = (board_cols, board_rows)
    criteria    = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    pipe  = rs.pipeline()
    cfg   = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    align = rs.align(rs.stream.color)

    profile     = pipe.start(cfg)
    intr        = (profile.get_stream(rs.stream.color)
                          .as_video_stream_profile().get_intrinsics())
    depth_scale = (profile.get_device().first_depth_sensor().get_depth_scale())
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

    print(f"\n[table]  Board inner corners {board_cols}×{board_rows}, "
          f"square {_SQUARE_M*1000:.0f} mm")
    print("[table]  Show chessboard to camera.  ESC to skip.\n")

    try:
        while True:
            frames      = pipe.wait_for_frames()
            aligned     = align.process(frames)
            cf, df      = aligned.get_color_frame(), aligned.get_depth_frame()
            if not cf or not df:
                continue

            img   = np.asarray(cf.get_data())
            depth = np.asarray(df.get_data()).astype(np.float32) * depth_scale

            gray         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, board_shape, None)

            disp = img.copy()
            if found:
                cv2.drawChessboardCorners(disp, board_shape, corners, True)
            cv2.putText(disp,
                        "FOUND — hold still" if found else "Searching …",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 220, 0) if found else (0, 80, 255), 2)
            cv2.imshow("Table plane detection  [ESC to skip]", disp)
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                return None, None
            if not found:
                continue

            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            pts3 = []
            for (u, v) in corners.reshape(-1, 2):
                ui, vi = int(round(u)), int(round(v))
                if not (0 <= ui < depth.shape[1] and 0 <= vi < depth.shape[0]):
                    continue
                z = float(depth[vi, ui])
                if z < 0.05 or z > 3.0:
                    continue
                pts3.append([(u-cx)*z/fx, (v-cy)*z/fy, z])

            if len(pts3) < 6:
                continue

            pts3     = np.array(pts3)
            centroid = pts3.mean(axis=0)
            _, _, Vt = np.linalg.svd(pts3 - centroid)
            normal   = Vt[-1] / np.linalg.norm(Vt[-1])
            d        = -float(normal @ centroid)
            if float(normal @ centroid) < 0:
                normal, d = -normal, -d

            res = float(np.abs((pts3 - centroid) @ normal).mean())
            if res > 0.005:
                print(f"[table]  Residual {res*1000:.1f} mm — retry")
                continue

            print(f"[table]  ✓  normal={np.round(normal,3)}  "
                  f"residual={res*1000:.2f} mm\n")
            cv2.waitKey(400)
            cv2.destroyAllWindows()
            return normal, d
    finally:
        pipe.stop()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE ① — Local curvature + classify
# ══════════════════════════════════════════════════════════════════════════════

def _fit_local_curvature(p, n, neighbours):
    ref = np.array([0.,0.,1.]) if abs(n[2])<0.9 else np.array([1.,0.,0.])
    t1  = np.cross(n,ref); t1/=np.linalg.norm(t1)
    t2  = np.cross(n,t1);  t2/=np.linalg.norm(t2)
    d   = neighbours - p
    u,v,h = d@t1, d@t2, d@n
    A   = np.column_stack([u**2, u*v, v**2])
    if np.linalg.matrix_rank(A) < 3:
        return None
    (a,b,c),*_ = np.linalg.lstsq(A, h, rcond=None)
    II          = np.array([[2*a,b],[b,2*c]])
    evals,evecs = np.linalg.eigh(II)
    ev          = evecs[:,0]
    return float(evals[0]), float(evals[1]), ev[0]*t1+ev[1]*t2


def _aggregate_curvatures(pts, normals):
    tree = cKDTree(pts)
    k1s,k2s,axs = [],[],[]
    for i in range(len(pts)):
        _,idx = tree.query(pts[i], k=_KNN_CURV+1)
        r = _fit_local_curvature(pts[i], normals[i], pts[idx[1:]])
        if r is None: continue
        k1s.append(r[0]); k2s.append(r[1]); axs.append(r[2])
    if not k1s:
        return 0.,0.,None
    k1,k2 = float(np.median(k1s)), float(np.median(k2s))
    axes  = np.array(axs)
    signs = np.sign(axes@axes[0]); signs[signs==0]=1
    axes *= signs[:,None]
    m = axes.mean(axis=0); nrm=np.linalg.norm(m)
    return k1, k2, m/nrm if nrm>1e-6 else None


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
        _, idx  = tree.query(pts[i], k=k + 1)
        mean_n  = normals[idx[1:]].mean(axis=0)
        score[i] = 1.0 - np.linalg.norm(mean_n)   # 0 = flat-face, 1 = random
    return score > (1.0 - _EDGE_COHERENCE_THRESH)


def _detect_2d_corners(pts_2d: np.ndarray,
                       angle_max: float = _CORNER_ANGLE_MAX) -> np.ndarray:
    """
    Find rigid-corner vertices on the convex hull of a 2-D point set.

    A vertex is a rigid corner if its interior angle ≤ angle_max degrees.
    For a perfect rectangle every hull corner is ≈ 90°; noise may push it to
    ~100°.  A cylinder's convex hull has no such sharp interior angles.

    Parameters
    ----------
    pts_2d  : (N, 2) projected points — any centring / scale
    angle_max : threshold in degrees

    Returns
    -------
    (M, 2) array of detected corner positions, or empty array if none found.
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
    Given M detected corners, complete a rectangle by mirroring and return
    exactly 4 corners in CCW angular order.

    M == 4  → order and return as-is
    M == 3  → 4th corner = P0 + P2 − P1  (parallelogram / mirror rule)
    M == 2  → the pair defines one edge; extend perpendicularly by the cloud
              width in that direction (signed toward the cloud centroid)
    M other → fall back to cv2.minAreaRect of pts_2d
    """
    def _ccw(c: np.ndarray) -> np.ndarray:
        ctr    = c.mean(axis=0)
        angles = np.arctan2(c[:, 1] - ctr[1], c[:, 0] - ctr[0])
        return c[np.argsort(angles)]

    m = len(corners)

    if m == 4:
        return _ccw(corners)

    if m == 3:
        c  = _ccw(corners)
        p3 = c[0] + c[2] - c[1]          # mirror: complete the parallelogram
        return _ccw(np.vstack([c, p3]))

    if m == 2:
        p0, p1   = corners[0], corners[1]
        edge_v   = p1 - p0
        edge_len = np.linalg.norm(edge_v)
        if edge_len > 1e-9:
            eu   = edge_v / edge_len
            perp = np.array([-eu[1], eu[0]])
            mid  = (p0 + p1) / 2.0
            # Width = PCD extent perpendicular to the known edge
            proj  = (pts_2d - mid) @ perp
            width = float(proj.max() - proj.min())
            # Shift toward the cloud centroid side
            direction = np.sign((pts_2d.mean(axis=0) - mid) @ perp) or 1.0
            p2 = p1 + direction * width * perp
            p3 = p0 + direction * width * perp
            return _ccw(np.array([p0, p1, p2, p3]))

    # Fallback: minimum-area bounding rectangle
    pts_f = pts_2d.astype(np.float32).reshape(-1, 1, 2)
    box   = cv2.boxPoints(cv2.minAreaRect(pts_f)).astype(float)
    return _ccw(box)


def _classify(k1: float, k2: float,
              pts: np.ndarray, normals: np.ndarray,
              axis: np.ndarray) -> str:
    """
    Classify the isolated object as 'cylinder' or 'cuboid'.

    Primary signal   : curvature anisotropy.
                       Flat κ₂ → planar surface → cuboid-like.
    Secondary signal : rigid-angle vertex and edge detection.
                       A cylinder has a smoothly curved surface — no sharp
                       corners and no high-coherence-drop edge bands.
                       A cuboid always has both.

    Decision
    --------
    → "cylinder"  if κ₂ signals a curved surface   AND  no rigid corners found
                  AND  edge-point fraction is low
    → "cuboid"    in all other cases
    """
    curv_cylinder = abs(k2) >= _FLAT_THRESH

    # Project pts onto the plane ⊥ axis for corner detection
    ref = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 else np.array([0., 1., 0.])
    e1  = np.cross(axis, ref);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(axis, e1)
    d   = pts - pts.mean(axis=0)
    pts_2d = np.column_stack([d @ e1, d @ e2])

    corners_2d = _detect_2d_corners(pts_2d)
    has_corners = len(corners_2d) >= _MIN_CORNERS_CUBOID

    edge_flags = _edge_mask(pts, normals)
    edge_frac  = float(edge_flags.mean())

    print(f"[shape_fit]  edge_frac={edge_frac:.3f}  "
          f"corners={len(corners_2d)}  has_corners={has_corners}")

    if curv_cylinder and not has_corners and edge_frac < _EDGE_FRAC_CUBOID:
        return "cylinder"
    return "cuboid"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE ② — Confirm orientation (aspect ratio primary, curvature fallback)
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

    ref = np.array([1.,0.,0.]) if abs(table_normal[0])<0.9 else np.array([0.,1.,0.])
    p1  = np.cross(table_normal,ref); p1/=np.linalg.norm(p1)
    p2  = np.cross(table_normal,p1);  p2/=np.linalg.norm(p2)
    wide = float(max((d@p1).ptp(), (d@p2).ptp()))

    aspect = up / (wide + 1e-6)
    print(f"[shape_fit]  up={up*1e3:.0f}mm  wide={wide*1e3:.0f}mm  "
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

    # Snap axis
    if orient == "vertical":
        axis = table_normal.copy()
    else:
        axis = raw_axis - (raw_axis @ table_normal)*table_normal
        nrm  = np.linalg.norm(axis)
        axis = raw_axis if nrm < 1e-6 else axis/nrm

    if axis @ raw_axis < 0:
        axis = -axis
    return orient, axis


# ══════════════════════════════════════════════════════════════════════════════
# STAGE ③ — Best fit (2D circle) + height
# ══════════════════════════════════════════════════════════════════════════════

def _best_fit_cylinder(pts, normals, axis):
    """
    Least-squares 2D circle fit in the plane ⊥ to axis.
    Returns (axis_pt, radius, mean_radial_err) or (None, None, inf).
    """
    centroid = pts.mean(axis=0)
    ref = np.array([0.,0.,1.]) if abs(axis[2])<0.9 else np.array([1.,0.,0.])
    e1  = np.cross(axis,ref); e1/=np.linalg.norm(e1)
    e2  = np.cross(axis,e1);  e2/=np.linalg.norm(e2)
    d   = pts - centroid
    u,v = d@e1, d@e2
    A   = np.column_stack([-2*u, -2*v, np.ones(len(u))])
    b   = -(u**2 + v**2)
    x,*_ = np.linalg.lstsq(A, b, rcond=None)
    cu,cv,dv = x
    r_sq = cu**2+cv**2-dv
    if r_sq <= 0:
        return None, None, np.inf
    r       = float(np.sqrt(r_sq))
    axis_pt = centroid + cu*e1 + cv*e2
    along   = (pts-axis_pt)@axis
    on_ax   = axis_pt + np.outer(along, axis)
    err     = float(np.mean(np.abs(np.linalg.norm(pts-on_ax, axis=1) - r)))
    return axis_pt, r, err


def _estimate_height(pts, normals, axis, centroid):
    proj  = (pts-centroid)@axis
    h_min = float(np.percentile(proj, 1))
    h_max = float(np.percentile(proj, 99))
    cap   = np.abs(normals@axis) > _CAP_NORMAL_DOT
    if cap.sum() >= _CAP_MIN_PTS:
        cp  = proj[cap]; span = h_max-h_min
        cm,cx = float(cp.min()), float(cp.max())
        if cm < h_min+0.3*span: h_min=cm; print(f"[shape_fit]  bottom cap {h_min*1e3:.0f}mm")
        if cx > h_max-0.3*span: h_max=cx; print(f"[shape_fit]  top cap    {h_max*1e3:.0f}mm")
    return h_min, h_max


# ══════════════════════════════════════════════════════════════════════════════
# STAGE ④ — Temporal convergence tracker
# ══════════════════════════════════════════════════════════════════════════════

class _ShapeTracker:
    """
    Maintains a temporally-smoothed estimate of the cylinder parameters.

    Each call to update() blends one frame's raw fit into the running estimate
    via EMA, then returns the smoothed parameters used to build the wireframe.

    Orientation locking
    -------------------
    After N_LOCK consecutive frames agree on "vertical" or "horizontal", the
    orientation is locked.  While locked, the axis is forced to the locked
    direction regardless of what the current frame estimates.
    Unlocking requires N_UNLOCK consecutive opposing frames — making accidental
    flips very unlikely without a real change in the scene.
    """

    def __init__(self):
        self.orient     = None      # "vertical" | "horizontal" | None
        self._streak    = 0         # consecutive frames with current orient
        self._locked    = False
        self._flip_str  = 0         # consecutive frames with opposing orient

        # Smoothed parameters
        self.axis    = None         # (3,) unit vector
        self.axis_pt = None         # (3,) point on axis
        self.radius  = None         # float (m)
        self.h_ctr   = None         # (h_min+h_max)/2
        self.height  = None         # h_max-h_min

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
        # Adaptive alpha: high-error frames contribute less
        alpha = float(np.clip(_ALPHA / (1.0 + raw_err * 30), 0.04, 0.35))

        # ── EMA parameter blend ───────────────────────────────────────────────
        raw_h_ctr = (raw_h_min + raw_h_max) / 2.0
        raw_h     = raw_h_max - raw_h_min

        if self.axis is None:
            # First valid frame — initialise directly
            self.axis    = table_normal.copy()
            self.axis_pt = raw_axis_pt.copy()
            self.radius  = raw_r
            self.h_ctr   = raw_h_ctr
            self.height  = raw_h
        else:
            # Smooth height; position and radius taken raw each frame so the
            # wireframe stays as tight as possible to the object surface.
            self.h_ctr  = (1-alpha)*self.h_ctr  + alpha*raw_h_ctr
            self.height = (1-alpha)*self.height  + alpha*raw_h
            self.axis_pt = raw_axis_pt.copy()   # raw fit → no lag
            self.radius  = raw_r                # raw fit → no lag

        # ── Axis is ALWAYS the chessboard table normal — no convergence drift ─
        # This is set unconditionally after every update so that neither EMA
        # blending nor orientation detection can tilt the axis away from vertical.
        self.axis = table_normal.copy()

        h_min = self.h_ctr - self.height / 2.0
        h_max = self.h_ctr + self.height / 2.0
        return self.axis, self.axis_pt, self.radius, h_min, h_max

    def reset(self):
        self.__init__()


# ══════════════════════════════════════════════════════════════════════════════
# Wireframe builders
# ══════════════════════════════════════════════════════════════════════════════

def _build_cylinder(axis, axis_pt, r, h_min, h_max):
    r      = float(np.clip(r, _R_MIN, _R_MAX))
    height = float(np.clip(h_max-h_min, 0.005, 5.0))
    center = axis_pt + axis*(h_min+h_max)/2.0

    z  = np.array([0.,0.,1.])
    v  = np.cross(z, axis);  s = np.linalg.norm(v);  c = float(np.dot(z,axis))
    if s < 1e-6:
        R = np.eye(3) if c>0 else np.diag([1.,-1.,-1.])
    else:
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R  = np.eye(3) + vx + vx@vx*(1.0-c)/(s**2)

    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=r,height=height,resolution=20)
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
    2. Find the convex-hull vertices whose interior angle ≤ _CORNER_ANGLE_MAX —
       these are the actual cuboid corners visible to the camera.
    3. Complete the rectangle by mirroring any missing corners:
         · 4 found → use directly
         · 3 found → 4th = P0 + P2 − P1  (parallelogram rule)
         · 2 found → extend perpendicularly by the cloud width
         · 0–1     → fall back to cv2.minAreaRect
    4. Reconstruct 8 three-dimensional vertices (4 bottom + 4 top) and build
       the 12-edge line set directly — no intermediate TriangleMesh.
    """
    n   = axis.copy()
    ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    e1  = np.cross(n, ref);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(n, e1)

    # ── Project onto horizontal plane (centred on cloud mean) ─────────────
    mean_pt = pts.mean(axis=0)
    d       = pts - mean_pt
    u       = (d @ e1).astype(np.float32)
    v       = (d @ e2).astype(np.float32)
    w       = pts @ n                          # absolute height along axis
    pts_2d  = np.column_stack([u, v])          # (N, 2), centred

    h_min = float(np.percentile(w, 1))
    h_max = float(np.percentile(w, 99))
    h_min = h_min if h_max - h_min > 0.005 else h_min - 0.0025

    # ── Detect visible corners, then mirror to complete rectangle ──────────
    # Subsample to ≤ 500 pts for corner detection; hull angle test is O(hull).
    if len(pts) > 500:
        idx    = np.random.choice(len(pts), 500, replace=False)
        sub_2d = pts_2d[idx]
    else:
        sub_2d = pts_2d

    corners_2d   = _detect_2d_corners(sub_2d)
    rect_corners = _complete_rectangle_2d(corners_2d, pts_2d)  # (4, 2) CCW

    # ── Reconstruct 3D vertices ────────────────────────────────────────────
    # The horizontal base position of the mean point (strips the n-component).
    mean_horiz = mean_pt - (mean_pt @ n) * n

    verts = []
    for (cu, cv) in rect_corners:
        horiz = mean_horiz + cu * e1 + cv * e2
        verts.append(horiz + h_min * n)   # bottom corner
        verts.append(horiz + h_max * n)   # top corner
    # Layout: [bot_0, top_0, bot_1, top_1, bot_2, top_2, bot_3, top_3]
    verts = np.array(verts)               # (8, 3)

    # ── Build 12 edges (4 bottom + 4 top + 4 vertical) ────────────────────
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

def fit_and_track(pts: np.ndarray, table_normal, tracker: _ShapeTracker):
    """
    Run one frame through the full pipeline and update the tracker.

    Returns (shape_name, LineSet) or (None, None).
    """
    if len(pts) < 50:
        return None, None

    # Normals
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=_KNN_NORMAL))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.,0.,0.]))
    normals = np.asarray(pcd.normals)

    N = len(pts)
    if N > _SUBSAMPLE:
        idx    = np.random.choice(N, _SUBSAMPLE, replace=False)
        sp,sn  = pts[idx], normals[idx]
    else:
        sp,sn  = pts, normals

    # ① Curvature + classify
    # For classification we need the projection axis; use table_normal when
    # available (it's always our vertical), fall back to raw curvature axis.
    k1, k2, raw_axis = _aggregate_curvatures(sp, sn)
    _cls_axis = (table_normal.copy() if table_normal is not None
                 else (raw_axis if raw_axis is not None
                       else np.array([0., 0., 1.])))
    shape = _classify(k1, k2, sp, sn, _cls_axis)
    print(f"[shape_fit]  κ₁={k1:+.2f}  κ₂={k2:+.2f}  → {shape}")

    if shape == "cuboid" or raw_axis is None:
        tracker.reset()
        _cub_axis = table_normal if table_normal is not None else _cls_axis
        return "cuboid", _build_cuboid(pts, _cub_axis)

    # ② Axis is always the chessboard table normal — vertical to the platform.
    # Orientation detection is skipped; the axis is never derived from curvature
    # or aspect ratio so convergence cannot tilt it.
    if table_normal is not None:
        axis = table_normal.copy()
    else:
        axis = raw_axis     # fallback: no table plane detected yet
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

    print(f"[shape_fit]  raw  r={r*1e3:.1f}mm  "
          f"h={(h_max-h_min)*1e3:.1f}mm  err={err*1e3:.2f}mm")

    # ⑤ Tracker: blend into running estimate
    if table_normal is not None:
        s_axis,s_pt,s_r,s_hmin,s_hmax = tracker.update(
            orient, axis, axis_pt, r, h_min, h_max, err, table_normal
        )
    else:
        s_axis,s_pt,s_r,s_hmin,s_hmax = axis, axis_pt, r, h_min, h_max

    print(f"[shape_fit]  smooth r={s_r*1e3:.1f}mm  "
          f"h={(s_hmax-s_hmin)*1e3:.1f}mm  "
          f"locked={tracker._locked}  orient={tracker.orient}")

    ls = _build_cylinder(s_axis, s_pt, s_r, s_hmin, s_hmax)
    return "cylinder", ls


# ══════════════════════════════════════════════════════════════════════════════
# Visualizer callback
# ══════════════════════════════════════════════════════════════════════════════

def _make_overlay_callback(table_normal):
    tracker = _ShapeTracker()
    state   = {"ls": None, "label": None}

    def _on_frame(obj_verts: np.ndarray, vis: o3d.visualization.Visualizer):
        shape, new_ls = fit_and_track(obj_verts, table_normal, tracker)
        if new_ls is None:
            return

        if shape != state["label"]:
            print(f"[shape_fit]  *** shape → {shape} ***")

        # Re-centre the camera lookat on the object centroid every frame
        vis.get_view_control().set_lookat(obj_verts.mean(axis=0).tolist())

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

def run(board_cols=_BOARD_COLS, board_rows=_BOARD_ROWS, debug=False):
    table_normal, _ = detect_table_plane(board_cols, board_rows)

    isolator = ObjectIsolator(min_points=50)
    isolator.start()
    print("Waiting for camera …")
    isolator.ready.wait()
    print("Camera ready.\n")

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
    p.add_argument("--board-cols", type=int, default=_BOARD_COLS)
    p.add_argument("--board-rows", type=int, default=_BOARD_ROWS)
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    run(board_cols=a.board_cols, board_rows=a.board_rows, debug=a.debug)