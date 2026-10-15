"""Safety and storage invariants for the local Dataset Studio."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from armsoft.object_segmentation.annotations import FrameAnnotation
from armsoft.sources.base import CameraIntrinsics
from tools.dataset_web import State


class DatasetWebSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data = Path(self.temp.name) / "data"
        self.capture = self.data / "raw" / "object"
        self.capture.mkdir(parents=True)
        self.state = State("sim", self.capture)

    def tearDown(self):
        self.temp.cleanup()

    def test_storage_report_counts_hardlinks_once_physically(self):
        raw = self.capture / "demo" / "color"
        export = self.data / "exports" / "object_segmentation" / "v1" / "images" / "train"
        raw.mkdir(parents=True); export.mkdir(parents=True)
        source = raw / "frame_000000.png"; source.write_bytes(b"x" * 4096)
        os.link(source, export / "demo__frame_000000.png")
        report = self.state.storage_report()
        self.assertEqual(report["categories"]["raw"]["unique_bytes"], 4096)
        self.assertEqual(report["categories"]["exports"]["unique_bytes"], 4096)
        self.assertEqual(report["physical_bytes"], 4096)

    def test_session_delete_is_blocked_when_export_references_it(self):
        session = self.capture / "demo"; session.mkdir(); (session / "session.json").write_text("{}")
        export = self.data / "exports" / "object_segmentation" / "v1"; export.mkdir(parents=True)
        (export / "manifest.json").write_text(json.dumps({"items": [{"session": "demo"}]}))
        with self.assertRaisesRegex(ValueError, "dependent dataset"):
            self.state.delete_session("demo", "demo")
        self.assertTrue(session.exists())

    def test_dataset_rejects_annotation_pointing_outside_raw_session(self):
        project = self.data / "annotations" / "object_segmentation" / "demo"
        (project / "annotations").mkdir(parents=True); (project / "annotation_run.json").write_text("{}")
        outside = Path(self.temp.name) / "outside"; outside.mkdir(); (outside / "image.png").write_bytes(b"image")
        FrameAnnotation("frame_000000", str(outside), "image.png", 10, 10, [], "approved").save(project / "annotations" / "frame_000000.json")
        report = self.state.dataset_validate({"assignments": [{"session": "demo", "split": "train"}]})
        self.assertFalse(report["valid"])
        self.assertTrue(any("does not match" in error for error in report["errors"]))

    def test_annotation_delete_preserves_raw_and_derived(self):
        raw = self.capture / "demo"; raw.mkdir(); (raw / "session.json").write_text("{}")
        derived = self.data / "derived" / "object" / "demo" / "v1"; derived.mkdir(parents=True)
        annotations = self.data / "annotations" / "object_segmentation" / "demo"; annotations.mkdir(parents=True)
        (annotations / "annotation_run.json").write_text("{}")
        self.state.annotation_delete("demo", "demo")
        self.assertTrue(raw.exists()); self.assertTrue(derived.exists()); self.assertFalse(annotations.exists())

    def test_active_training_dataset_cannot_be_deleted(self):
        export = self.data / "exports" / "object_segmentation" / "v1"; export.mkdir(parents=True)
        (export / "manifest.json").write_text("{}")
        self.state.training_job = {"state": "running", "dataset": "v1", "run_name": "r1"}
        with self.assertRaisesRegex(ValueError, "Stop the training"):
            self.state.dataset_delete("v1", "v1")
        self.assertTrue(export.exists())

    def test_dot_paths_are_rejected_as_resource_names(self):
        for name in (".", "..", ".hidden"):
            with self.assertRaises(ValueError):
                self.state.delete_session(name, name)

    def test_live_test_requires_completed_local_model(self):
        with self.assertRaisesRegex(ValueError, "best.pt"):
            self.state.test_start({"run_name": "missing", "device": "cpu"})

    def test_failure_case_is_saved_as_normal_rgbd_session(self):
        intrinsics = CameraIntrinsics(4, 3, 2.0, 2.0, 2.0, 1.5)
        self.state.latest_frame = (
            np.zeros((3, 4, 3), dtype=np.uint8),
            np.ones((3, 4), dtype=np.float32),
            intrinsics, 7, 123.0, "test-camera",
        )
        self.state.test_job = {"run_name": "test_run", "objects": [], "failure_saved": 0, "failure_skipped_similar": 0}
        result = self.state.test_save_failure({"failure_type": "missed_detection", "note": "Expected the box"})
        session = self.capture / result["session"]
        self.assertTrue((session / "session.json").is_file())
        self.assertEqual(len(list((session / "color").glob("*.png"))), 1)
        self.assertEqual(len(list((session / "depth").glob("*.npy"))), 1)
        record = json.loads((session / "manifest.jsonl").read_text().splitlines()[0])
        self.assertEqual(record["failure_type"], "missed_detection")
        self.assertEqual(record["failure_note"], "Expected the box")

    def test_failure_collection_skips_near_duplicate_frames(self):
        intrinsics = CameraIntrinsics(4, 3, 2.0, 2.0, 2.0, 1.5)
        self.state.latest_frame = (np.zeros((3, 4, 3), dtype=np.uint8), np.ones((3, 4), dtype=np.float32), intrinsics, 1, 1.0, "test-camera")
        self.state.test_job = {"run_name": "test_run", "objects": [], "failure_saved": 0, "failure_skipped_similar": 0}
        first = self.state.test_save_failure({"failure_type": "bad_mask"})
        second = self.state.test_save_failure({"failure_type": "bad_mask"})
        self.assertFalse(first["skipped_similar"])
        self.assertTrue(second["skipped_similar"])
        self.assertEqual(len(list((self.capture / first["session"] / "color").glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
