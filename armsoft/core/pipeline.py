"""
armsoft/pipeline.py
====================
The model itself: one RGB-D frame in, one :class:`GraspResult` out.

    RGBDFrame
        │  ObjectIsolatorRGBD          armsoft/isolation.py
        ▼
    isolated object point cloud + BGR crop
        │  shape hint (YOLO or geometric)   armsoft/classifier.py
        ▼
    fit_once + ShapeEMA                armsoft/shape_fitter.py
        │
        ▼
    grasp_from_shape / joint_positions armsoft/grasp_geometry.py
        │
        ▼
    GraspResult  →  .to_dict() / .to_json()

There is no threading here: ``process()`` is synchronous and deterministic, so
the same input always gives the same output.  Threading, rendering and
transport are the caller's business (see ``run_inference.py`` and
``armsoft/output.py``).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from ..sources.base import RGBDFrame
from . import grasp_geometry as gg
from .isolation import ObjectIsolatorRGBD
from .shape_fitter import fit_once, ShapeEMA

#: Bump this whenever the output dict changes shape.
SCHEMA_VERSION = "arm-soft-grasp/1.0"

#: Default table normal in camera coordinates (+Y is down), i.e. "up".
DEFAULT_TABLE_NORMAL = np.array([0.0, -1.0, 0.0])

# Stability gate — a grasp is only reported stable once it has settled.
STABLE_FRAMES = 8
STABLE_POS_M  = 0.004
STABLE_JAW_M  = 0.003


@dataclass
class GraspResult:
    """
    Stable output record.  ``to_dict()`` is the contract other software
    (a ROS 2 bridge, an Arduino UNO Q firmware, a logger) should depend on:
    plain JSON scalars and flat float lists only — no numpy, no Open3D.

    status
        ``"ok"``         a grasp was computed.
        ``"no_object"``  no red object isolated in this frame.
        ``"no_shape"``   object seen, but no shape hint / fit available.
        ``"no_grasp"``   shape fitted, but no valid grasp could be derived.
    """

    status: str
    frame_index: int = 0
    timestamp: float = 0.0
    source: str = "unknown"
    shape: str | None = None
    shape_source: str | None = None
    n_object_points: int = 0

    # Grasp, camera frame (metres) — all None unless status == "ok"
    position_m: list[float] | None = None
    approach: list[float] | None = None
    closing: list[float] | None = None
    binormal: list[float] | None = None
    rotation: list[list[float]] | None = None
    jaw_opening_m: float | None = None

    # Object descriptors
    distance_m: float | None = None
    object_width_m: float | None = None
    object_height_m: float | None = None

    # Robot-facing command
    joint_names: list[str] = field(default_factory=lambda: list(gg.JOINT_NAMES))
    joint_positions: list[float] | None = None
    base_roll_rad: float | None = None
    elevation_deg: float | None = None
    bearing_deg: float | None = None
    hand_pose: int = 0

    # Stability / diagnostics
    stable: bool = False
    pos_std_m: float | None = None
    jaw_std_m: float | None = None
    gripper_points: list[list[float]] | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = SCHEMA_VERSION
        return d

    def to_json(self, indent: int | None = 2) -> str:
        import json
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class GraspPipeline:
    """
    Frame → grasp.  No camera, no robot middleware, no GPU required.

    Parameters
    ----------
    table_normal : (3,) array-like or None
        Unit normal of the work surface in camera coordinates.  ``None`` uses
        :data:`DEFAULT_TABLE_NORMAL`.  On the robot this comes from the
        chessboard calibration (``armsoft/table_plane.py``).
    classifier : ShapeClassifierBase or None
        Shape-hint provider.  ``None`` builds the geometric classifier.
    min_points : int
        Minimum isolated points for a frame to be considered valid.
    smooth : bool
        Apply the ``ShapeEMA`` temporal smoothing (recommended for streams,
        turn it off for single-frame batch inference).
    """

    def __init__(self, table_normal=None, classifier=None,
                 min_points: int = 50, smooth: bool = True):
        self.table_normal = (DEFAULT_TABLE_NORMAL if table_normal is None
                             else np.asarray(table_normal, float))
        n = np.linalg.norm(self.table_normal)
        if n > 1e-9:
            self.table_normal = self.table_normal / n

        if classifier is None:
            from .classifier import GeometricShapeClassifier
            classifier = GeometricShapeClassifier()
        self.classifier = classifier

        self.isolator = ObjectIsolatorRGBD(min_points=min_points)
        self._ema = ShapeEMA() if smooth else None
        self._trans_buf: deque = deque(maxlen=STABLE_FRAMES)
        self._jaw_buf: deque = deque(maxlen=STABLE_FRAMES)
        self.last_isolation = None      # exposed for viewers / debugging
        self.last_shape_ls = None       # Open3D LineSet of the fitted shape

    # ── API ──────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear temporal state (call when the object leaves the scene)."""
        if self._ema is not None:
            self._ema.reset()
        self._trans_buf.clear()
        self._jaw_buf.clear()
        self.last_shape_ls = None

    def process(self, frame: RGBDFrame) -> GraspResult:
        t0 = time.perf_counter()
        base = dict(frame_index=frame.index, timestamp=frame.timestamp,
                    source=frame.source)

        iso = self.isolator.isolate(frame)
        self.last_isolation = iso
        t_iso = time.perf_counter()

        if not iso.found:
            self.reset()
            return GraspResult(status="no_object", **base,
                               timing_ms={"isolation": (t_iso - t0) * 1e3})

        hint = self.classifier.predict(iso.crop_bgr, points=iso.object_points,
                                       table_normal=self.table_normal)
        t_clf = time.perf_counter()
        if hint is None:
            self.reset()
            return GraspResult(status="no_shape", n_object_points=len(iso.object_points),
                               shape_source=self.classifier.name, **base,
                               timing_ms={"isolation": (t_iso - t0) * 1e3,
                                          "classify": (t_clf - t_iso) * 1e3})

        shape, shape_ls = fit_once(iso.object_points, self.table_normal,
                                   shape_hint=hint)
        if self._ema is not None:
            shape, shape_ls = self._ema.update(shape, shape_ls)
        t_fit = time.perf_counter()
        self.last_shape_ls = shape_ls

        timing = {"isolation": (t_iso - t0) * 1e3,
                  "classify":  (t_clf - t_iso) * 1e3,
                  "fit":       (t_fit - t_clf) * 1e3}

        if shape_ls is None:
            return GraspResult(status="no_shape", shape=shape,
                               shape_source=self.classifier.name,
                               n_object_points=len(iso.object_points),
                               **base, timing_ms=timing)

        rot, trans, half_w = gg.grasp_from_shape(shape, self.table_normal, shape_ls)
        timing["grasp"] = (time.perf_counter() - t_fit) * 1e3
        timing["total"] = (time.perf_counter() - t0) * 1e3

        if rot is None:
            self._trans_buf.clear()
            self._jaw_buf.clear()
            return GraspResult(status="no_grasp", shape=shape,
                               shape_source=self.classifier.name,
                               n_object_points=len(iso.object_points),
                               **base, timing_ms=timing)

        # ── stability gate ───────────────────────────────────────────────────
        self._trans_buf.append(np.asarray(trans, float).copy())
        self._jaw_buf.append(float(half_w))
        pos_std = jaw_std = None
        stable = False
        if len(self._trans_buf) == STABLE_FRAMES:
            pos_std = float(np.std(np.array(self._trans_buf), axis=0).max())
            jaw_std = float(np.std(self._jaw_buf))
            stable = (pos_std < STABLE_POS_M) and (jaw_std < STABLE_JAW_M)

        verts = gg.wireframe_vertices(shape_ls)
        height_m = float(np.ptp(verts @ self.table_normal))
        dist, elev, bear, _hand = gg.object_params(rot, trans)

        return GraspResult(
            status="ok", shape=shape, shape_source=self.classifier.name,
            n_object_points=len(iso.object_points), **base,
            position_m=[float(x) for x in trans],
            approach=[float(x) for x in rot[:, 0]],
            closing=[float(x) for x in rot[:, 1]],
            binormal=[float(x) for x in rot[:, 2]],
            rotation=[[float(x) for x in row] for row in rot],
            jaw_opening_m=float(2.0 * half_w),
            distance_m=dist,
            object_width_m=float(2.0 * (half_w - gg.FINGER_CLEARANCE)),
            object_height_m=height_m,
            joint_positions=gg.joint_positions(rot, trans, self.table_normal),
            base_roll_rad=gg.base_roll(rot, self.table_normal),
            elevation_deg=elev, bearing_deg=bear,
            hand_pose=1 if stable else 0,
            stable=stable, pos_std_m=pos_std, jaw_std_m=jaw_std,
            gripper_points=[[float(x) for x in p]
                            for p in gg.gripper_skeleton(rot, trans, half_w)],
            timing_ms=timing,
        )
