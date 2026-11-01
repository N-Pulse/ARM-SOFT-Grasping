"""
armsoft/selftest.py
====================
Smoke test for the model — runs on any development machine, no hardware needed.

    python -m armsoft.tests.selftest

Checks, in order:
  1. every portable module imports;
  2. the simulator renders a geometrically correct RGB-D frame;
  3. isolation recovers the object cloud at the right size;
  4. the full pipeline produces a valid grasp for a cylinder and a cuboid;
  5. the result dict is JSON-serialisable and carries the expected keys.

Exit code 0 = all good.
"""

from __future__ import annotations

import json
import sys

import numpy as np

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    print("portable self-test\n" + "=" * 60)

    print("\n[1] imports")
    from ..sources import SimulatedSource, camera_available
    from ..core.isolation import ObjectIsolatorRGBD
    from ..core.classifier import GeometricShapeClassifier, FixedShapeClassifier
    from ..core.pipeline import GraspPipeline, SCHEMA_VERSION
    from ..core import grasp_geometry as gg
    check("modules import", True)
    check("camera probe does not raise", isinstance(camera_available(), bool),
          f"camera present = {camera_available()}")

    print("\n[2] simulator")
    src = SimulatedSource(shape="cylinder", n_frames=1, noise_m=0.0)
    frame = src.read()
    check("frame shape", frame.color_bgr.shape == (480, 640, 3)
          and frame.depth_m.shape == (480, 640))
    d = frame.depth_m[frame.depth_m > 0]
    check("depth range plausible", 0.1 < float(d.min()) < 0.9, f"min={d.min():.3f} m")

    print("\n[3] isolation")
    iso = ObjectIsolatorRGBD().isolate(frame)
    ext = iso.object_points.max(0) - iso.object_points.min(0)
    check("object isolated", iso.found, f"{len(iso.object_points)} points")
    check("object size ≈ 60×100 mm",
          abs(ext[0] - 0.06) < 0.01 and abs(ext[1] - 0.10) < 0.01,
          f"extent = {np.round(ext * 1e3, 1)} mm")

    print("\n[4] pipeline")
    for shape in ("cylinder", "cuboid"):
        src = SimulatedSource(shape=shape, n_frames=12, noise_m=0.0008, seed=1)
        pipe = GraspPipeline(table_normal=src.table_normal_hint,
                             classifier=FixedShapeClassifier(shape))
        results = [pipe.process(f) for f in src.frames()]
        ok = [r for r in results if r.ok]
        check(f"{shape}: grasp computed", len(ok) == len(results),
              f"{len(ok)}/{len(results)} frames")
        if ok:
            last = ok[-1]
            check(f"{shape}: shape label", last.shape == shape, last.shape or "-")
            check(f"{shape}: distance ≈ 0.35 m",
                  abs(last.distance_m - 0.35) < 0.05, f"{last.distance_m:.3f} m")
            check(f"{shape}: rotation orthonormal",
                  np.allclose(np.asarray(last.rotation).T @ np.asarray(last.rotation),
                              np.eye(3), atol=1e-6))
            check(f"{shape}: stability gate reached", any(r.stable for r in ok))

    print("\n[5] output contract")
    d = last.to_dict()
    required = {"schema", "status", "shape", "position_m", "rotation",
                "jaw_opening_m", "joint_names", "joint_positions",
                "base_roll_rad", "hand_pose", "stable", "distance_m"}
    check("all contract keys present", required <= set(d),
          f"missing: {sorted(required - set(d)) or 'none'}")
    check("schema version", d["schema"] == SCHEMA_VERSION, d["schema"])
    try:
        json.loads(json.dumps(d))
        check("JSON round-trip", True)
    except Exception as exc:
        check("JSON round-trip", False, str(exc))

    check("geometric classifier constructs",
          GeometricShapeClassifier().name == "geometric")
    check("gripper skeleton has 6 points",
          gg.gripper_skeleton(np.eye(3), np.zeros(3), 0.03).shape == (6, 3))

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
