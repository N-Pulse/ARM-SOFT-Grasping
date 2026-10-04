"""
porting/check_env.py
====================
Import *and exercise* every dependency of the pipeline, and print one line per
library: OK / MISSING / BROKEN, the version, and what was tested.

    python porting/check_env.py            # human-readable table
    python porting/check_env.py --json     # machine-readable
    python porting/check_env.py --minimal  # judge against the UNO Q target set

An import alone is not enough on ARM: Open3D can import and then crash in
DBSCAN, torchvision can import without its C++ ops, and so on.  Each probe below
calls the exact functions armsoft uses.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback


def _numpy():
    import numpy as np
    a = np.random.default_rng(0).normal(size=(200, 3))
    np.linalg.lstsq(np.c_[a, np.ones(200)], a[:, 0], rcond=None)
    return np.__version__, "linalg.lstsq"


def _scipy():
    import numpy as np
    import scipy
    from scipy.optimize import least_squares
    from scipy.spatial import cKDTree
    u = np.cos(np.linspace(0, 2, 50)) * 0.03
    v = np.sin(np.linspace(0, 2, 50)) * 0.03
    r = least_squares(lambda p: np.hypot(u - p[0], v - p[1]) - abs(p[2]),
                      [0.001, 0.001, 0.02], method="lm").x[2]
    assert abs(abs(r) - 0.03) < 1e-4
    cKDTree(np.c_[u, v]).query_ball_point([0, 0], 0.05)
    return scipy.__version__, "optimize.least_squares(lm), spatial.cKDTree"


def _cv2():
    import cv2
    import numpy as np
    img = np.zeros((64, 64, 3), np.uint8)
    cv2.circle(img, (32, 32), 10, (0, 0, 255), -1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.minAreaRect(cnt[0])
    cv2.convexHull(cnt[0])
    cv2.findChessboardCorners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (7, 6), None)
    return cv2.__version__, "HSV, contours, minAreaRect, convexHull, chessboard"


def _open3d():
    import numpy as np
    import open3d as o3d
    pts = np.random.default_rng(0).uniform(-0.05, 0.05, size=(3000, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(0.008)
    pcd.cluster_dbscan(eps=0.018, min_points=5, print_progress=False)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=6))
    pcd.orient_normals_towards_camera_location(np.zeros(3))
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.03, height=0.1)
    o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    return o3d.__version__, "voxel, DBSCAN, normals, cylinder LineSet"


def _torch():
    import torch
    x = torch.randn(1, 3, 128, 128)
    torch.nn.functional.conv2d(x, torch.randn(8, 3, 3, 3))
    return torch.__version__, f"conv2d; threads={torch.get_num_threads()}"


def _torchvision():
    import torch
    import torchvision
    torch.ops.torchvision.nms           # fails on a torch/torchvision mismatch
    return torchvision.__version__, "C++ ops (nms) registered"


def _ultralytics():
    import ultralytics
    return ultralytics.__version__, "import"


def _onnxruntime():
    import onnxruntime as ort
    return ort.__version__, "providers=" + ",".join(ort.get_available_providers())


def _onnx():
    import onnx
    return onnx.__version__, "import (export-time only)"


def _tflite():
    try:
        from ai_edge_litert.interpreter import Interpreter  # noqa: F401
        import ai_edge_litert
        return getattr(ai_edge_litert, "__version__", "?"), "ai_edge_litert.Interpreter"
    except ImportError:
        from tflite_runtime.interpreter import Interpreter  # noqa: F401
        import tflite_runtime
        return tflite_runtime.__version__, "tflite_runtime.Interpreter"


def _pyrealsense2():
    import pyrealsense2 as rs
    n = len(rs.context().query_devices())
    return getattr(rs, "__version__", "?"), f"{n} device(s)"


def _serial():
    import serial
    return serial.__version__, "import"


#  name, probe, role in the pipeline
PROBES = [
    ("numpy",        _numpy,        "required"),
    ("scipy",        _scipy,        "required  (cylinder fit)"),
    ("opencv",       _cv2,          "required  (isolation, fitting)"),
    ("open3d",       _open3d,       "required today; to be replaced"),
    ("onnxruntime",  _onnxruntime,  "classifier runtime (ONNX)"),
    ("torch",        _torch,        "optional  (--classifier yolo)"),
    ("torchvision",  _torchvision,  "optional  (--classifier yolo)"),
    ("ultralytics",  _ultralytics,  "optional  (--classifier yolo)"),
    ("onnx",         _onnx,         "export host only"),
    ("tflite",       _tflite,       "alternative classifier runtime"),
    ("pyrealsense2", _pyrealsense2, "optional  (live camera)"),
    ("pyserial",     _serial,       "optional  (--serial)"),
]


def main() -> int:
    as_json = "--json" in sys.argv
    rows = []
    for name, probe, role in PROBES:
        t0 = time.perf_counter()
        try:
            ver, what = probe()
            status = "OK"
        except ImportError as exc:
            status, ver, what = "MISSING", "-", str(exc).splitlines()[0]
        except Exception as exc:                      # imported, then failed
            status, ver, what = "BROKEN", "-", f"{type(exc).__name__}: {exc}"
            if "-v" in sys.argv:
                traceback.print_exc()
        rows.append(dict(lib=name, status=status, version=ver, role=role,
                         tested=what, seconds=round(time.perf_counter() - t0, 2)))

    env = dict(machine=platform.machine(), python=platform.python_version(),
               platform=platform.platform())
    if as_json:
        print(json.dumps(dict(env=env, libs=rows), indent=2))
        return 0
    print(f"machine={env['machine']}  python={env['python']}  {env['platform']}")
    for r in rows:
        print(f"  {r['status']:<8}{r['lib']:<13}{r['version']:<14}"
              f"{r['role']:<34}{r['tested']}  ({r['seconds']}s)")
    # --minimal: the UNO Q target set, where Open3D is replaced by open3d_lite
    core = (("numpy", "scipy", "opencv", "onnxruntime") if "--minimal" in sys.argv
            else ("numpy", "scipy", "opencv", "open3d"))
    core_ok = all(r["status"] == "OK" for r in rows if r["lib"] in core)
    print("core dependencies:", "OK" if core_ok else "NOT OK")
    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
