"""
armsoft — vision-based grasp planning for the n-pulse soft robotic hand.

One aligned RGB-D frame in, one grasp out.  The package is self-contained: it
needs no robot, no middleware, no GPU and — thanks to the built-in simulator —
no camera either.

    from armsoft import GraspPipeline, create_frame_source

    with create_frame_source("auto") as source:          # camera or simulator
        pipeline = GraspPipeline(table_normal=source.table_normal_hint)
        for frame in source.frames(limit=1):
            print(pipeline.process(frame).to_json())

See the project README for setup, the command-line tool and the I/O contract.

The package is laid out in three layers:

    armsoft.sources   where frames come from — camera, simulator, recording
    armsoft.core      the model — isolation, classifier, fit, grasp, pipeline
    armsoft.sinks     where results go — stdout, JSONL, ROS 2, serial, viewer

Everything commonly needed is re-exported here, so `from armsoft import X`
keeps working regardless of which layer X lives in.
"""

from .sources import (
    RGBDFrame,
    CameraIntrinsics,
    FrameSource,
    SimulatedSource,
    RealSenseSource,
    RecordedSource,
    create_frame_source,
    camera_available,
    save_frame,
)
from .core import (
    ObjectIsolatorRGBD,
    Isolation,
    build_classifier,
    YoloShapeClassifier,
    GeometricShapeClassifier,
    FixedShapeClassifier,
    GraspPipeline,
    GraspResult,
    SCHEMA_VERSION,
    detect_table_plane,
)

__version__ = "1.0.0"

__all__ = [
    "RGBDFrame", "CameraIntrinsics", "FrameSource", "SimulatedSource",
    "RealSenseSource", "RecordedSource", "create_frame_source",
    "camera_available", "save_frame",
    "ObjectIsolatorRGBD", "Isolation",
    "build_classifier", "YoloShapeClassifier", "GeometricShapeClassifier",
    "FixedShapeClassifier",
    "GraspPipeline", "GraspResult", "SCHEMA_VERSION",
    "detect_table_plane", "__version__",
]
