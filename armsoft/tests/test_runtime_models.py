"""Runtime model registry contract tests without loading neural weights."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from armsoft.models import MODEL_SCHEMA, discover_runtime_models


class RuntimeModelRegistryTests(unittest.TestCase):
    def test_discovers_valid_model_and_ignores_incomplete_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); model = root / "hand-pose-v1"; model.mkdir()
            (model / "best.pt").write_bytes(b"weights")
            (model / "model.json").write_text(json.dumps({
                "schema": MODEL_SCHEMA, "id": "hand-pose-v1",
                "display_name": "Hand Pose v1", "task": "pose",
                "weights": "best.pt", "model_version": "1.0.0",
                "keypoint_count": 17,
            }))
            (root / "incomplete").mkdir()
            records = discover_runtime_models(root)
            self.assertEqual([item.id for item in records], ["hand-pose-v1"])
            self.assertEqual(records[0].task, "pose")
            self.assertEqual(records[0].keypoint_count, 17)

    def test_rejects_metadata_that_escapes_model_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); outside = root / "outside.pt"; outside.write_bytes(b"x")
            model = root / "unsafe"; model.mkdir()
            (model / "model.json").write_text(json.dumps({
                "schema": MODEL_SCHEMA, "id": "unsafe", "task": "detection",
                "weights": "../outside.pt",
            }))
            self.assertEqual(discover_runtime_models(root), [])


if __name__ == "__main__":
    unittest.main()
