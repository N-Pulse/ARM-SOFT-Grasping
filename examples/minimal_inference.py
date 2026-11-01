"""
Smallest possible end-to-end use of the model.

    python examples/minimal_inference.py

Uses a real depth camera when one is connected, and the built-in simulator
otherwise, then prints the grasp as JSON.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from armsoft import GraspPipeline, create_frame_source, build_classifier

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "models", "shape_classifier.pt")

with create_frame_source("auto") as source:
    pipeline = GraspPipeline(
        table_normal=source.table_normal_hint,
        classifier=build_classifier("auto", model_path=WEIGHTS),
    )

    for frame in source.frames(limit=1):
        result = pipeline.process(frame)
        print(result.to_json())
        raise SystemExit(0 if result.ok else 1)
