"""
armsoft/core/__init__.py
========================
The model: an RGB-D frame in, a grasp out.  Nothing here touches a camera, a
display or a transport — those live in :mod:`armsoft.sources` and
:mod:`armsoft.sinks`.

    isolation.py       find the object and turn it into a point cloud
    classifier.py      decide whether it is a cylinder or a cuboid
    shape_fitter.py    fit that primitive and smooth it over time
    grasp_geometry.py  turn the fitted shape into a grasp and joint targets
    pipeline.py        run all of the above — GraspPipeline / GraspResult
    table_plane.py     calibrate the work surface from a chessboard
"""

from .classifier import (build_classifier, FixedShapeClassifier,
                         GeometricShapeClassifier, ShapeClassifierBase,
                         YoloShapeClassifier)
from .isolation import Isolation, ObjectIsolatorRGBD
from .pipeline import GraspPipeline, GraspResult, SCHEMA_VERSION
from .table_plane import detect_table_plane

__all__ = ["build_classifier", "FixedShapeClassifier", "GeometricShapeClassifier",
           "ShapeClassifierBase", "YoloShapeClassifier",
           "Isolation", "ObjectIsolatorRGBD",
           "GraspPipeline", "GraspResult", "SCHEMA_VERSION",
           "detect_table_plane"]
