"""
armsoft/classifier.py
======================
Shape-hint providers: "cylinder" | "cuboid" | None.

``YoloShapeClassifier``      the trained YOLOv8-cls model (``models/
                             shape_classifier.pt``), run on CPU by default.
``GeometricShapeClassifier`` no-neural-network fallback: it projects the object
                             cloud onto the table plane and analyses the 2-D
                             footprint (:mod:`armsoft.shape_fitter`).  Useful
                             when torch / ultralytics are unavailable — e.g. on
                             a very small target board.
``FixedShapeClassifier``     always returns the same label (tests, debugging).

All of them expose the same two-method interface::

    hint = clf.predict(crop_bgr, points=obj_points, table_normal=n)
    clf.name        # short string for logging / result metadata
"""

from __future__ import annotations

import numpy as np


class ShapeClassifierBase:
    name = "none"

    def predict(self, crop_bgr, points=None, table_normal=None) -> str | None:
        raise NotImplementedError


class YoloShapeClassifier(ShapeClassifierBase):
    """
    YOLOv8-classify wrapper — predicts "cylinder" or "cuboid" from the
    cropped colour image of the isolated object.

    The default device is **cpu**, so it runs on a plain development machine;
    pass ``device="cuda"`` if a GPU is available.
    """

    name = "yolo"

    def __init__(self, model_path: str, device: str = "cpu",
                 conf_thresh: float = 0.70, imgsz: int = 128,
                 verbose: bool = True):
        from ultralytics import YOLO  # noqa: PLC0415 — heavy, optional
        self._model = YOLO(model_path)
        self._device = device
        self._conf = float(conf_thresh)
        self._imgsz = int(imgsz)
        self.last_confidence: float | None = None
        self.last_label: str | None = None
        if verbose:
            print(f"[YoloShapeClassifier] '{model_path}' on {device}; "
                  f"classes={self._model.names} conf_thresh={conf_thresh}")

    def predict(self, crop_bgr, points=None, table_normal=None) -> str | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        res = self._model.predict(source=crop_bgr, device=self._device,
                                  imgsz=self._imgsz, verbose=False)
        probs = res[0].probs
        conf = float(probs.top1conf)
        name = self._model.names[int(probs.top1)]
        self.last_confidence, self.last_label = conf, name
        return name if conf >= self._conf else None


class GeometricShapeClassifier(ShapeClassifierBase):
    """
    Geometry-only shape hint — projects the object cloud onto the table plane
    and analyses the 2-D footprint (``shape_fitter._classify_topdown``).

    Requires no torch, no ultralytics and no trained weights.
    """

    name = "geometric"

    def __init__(self, voxel_size: float = 0.008):
        from . import shape_fitter  # noqa: PLC0415
        self._sf = shape_fitter
        self._voxel = float(voxel_size)
        self.last_label: str | None = None

    def predict(self, crop_bgr=None, points=None, table_normal=None) -> str | None:
        if points is None or len(points) < 20:
            return None
        import open3d as o3d  # noqa: PLC0415
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pts = np.asarray(pcd.voxel_down_sample(self._voxel).points)
        if len(pts) < 20:
            return None
        pts = self._sf._largest_cluster(pts)
        if len(pts) < 20:
            return None
        axis = (np.asarray(table_normal, float) if table_normal is not None
                else np.array([0., 0., 1.]))
        label, _score, _corners = self._sf._classify_topdown(pts, axis)
        if label not in ("cylinder", "cuboid"):
            return None
        self.last_label = label
        return label


class FixedShapeClassifier(ShapeClassifierBase):
    """Always returns ``label`` — for deterministic tests."""

    name = "fixed"

    def __init__(self, label: str = "cylinder"):
        self.label = label

    def predict(self, crop_bgr=None, points=None, table_normal=None) -> str | None:
        return self.label


def build_classifier(kind: str = "auto", model_path: str | None = None,
                     device: str = "cpu", **kwargs) -> ShapeClassifierBase:
    """
    kind
        ``"auto"``       YOLO when ultralytics + weights are available,
                         otherwise the geometric classifier.
        ``"yolo"``       force YOLO (raises when unavailable).
        ``"geometric"``  force the geometry-only classifier.
        ``"fixed"``      ``FixedShapeClassifier(label=...)``.
    """
    kind = (kind or "auto").lower()
    if kind == "fixed":
        return FixedShapeClassifier(kwargs.get("label", "cylinder"))
    if kind == "geometric":
        return GeometricShapeClassifier()
    if kind == "yolo":
        return YoloShapeClassifier(model_path, device=device)
    if kind != "auto":
        raise ValueError(f"unknown classifier {kind!r}")

    try:
        import ultralytics  # noqa: F401,PLC0415
        import os
        if model_path and os.path.exists(model_path):
            return YoloShapeClassifier(model_path, device=device)
        print(f"[classifier] weights not found at {model_path!r} — "
              f"using the geometric classifier.")
    except Exception as exc:
        print(f"[classifier] ultralytics unavailable ({exc}) — "
              f"using the geometric classifier.")
    return GeometricShapeClassifier()
