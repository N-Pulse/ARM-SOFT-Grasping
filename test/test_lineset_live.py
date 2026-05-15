"""
test_lineset_live.py

Live GraspNet grasp visualisation on the isolated object point cloud.
ObjectIsolator locks on one object; GraspInferenceThread runs GraspNet
in the background; the best grasp is drawn as a lineset fork over the
live Open3D point cloud viewer.

Point-cloud processing and display (frame reading, pts/cols selection,
geom_added / zoom_fitted flags, cv2 YOLO preview) is identical to
show_isolated_pcd() in helper/pcd_visualizer.py.  The only additions are
the lineset geometry and the grasp-inference thread.

Usage:
    python test_lineset_live.py --checkpoint /path/to/checkpoint.tar
    python test_lineset_live.py --checkpoint /path/to/checkpoint.tar --device cpu

Controls:
    Close the Open3D window  |  Ctrl+C  |  ESC / q in the YOLO preview
"""

import sys
import os
import queue
import signal
import threading
import time
import argparse

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "capture"))
sys.path.insert(0, _ROOT)   # exposes the helper package

import cv2
import numpy as np
import open3d as o3d
import torch

from capture.object_isolation import ObjectIsolator
from grasp_pipeline_graspnet import load_model, GraspInferenceThread
from helper.pcd_visualizer import auto_zoom


# ── Gripper geometry ──────────────────────────────────────────────────────────

FINGER_LENGTH = 0.085
PALM_DEPTH    = 0.06
GRIPPER_COLOR = [1.0, 0.4, 0.0]


def _gripper_lineset(rot, trans, width):
    approach  = rot[:, 0]
    closing   = rot[:, 1]
    half_w    = width / 2.0

    tip_center = trans + approach * FINGER_LENGTH
    palm_back  = trans - approach * PALM_DEPTH
    left_root  = trans      + closing * half_w
    right_root = trans      - closing * half_w
    left_tip   = tip_center + closing * half_w
    right_tip  = tip_center - closing * half_w

    pts = np.array([
        palm_back, trans, left_root, right_root, left_tip, right_tip,
    ], dtype=np.float64)

    lines = [[0,1],[1,2],[1,3],[2,4],[3,5]]

    ls        = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([GRIPPER_COLOR] * len(lines))
    return ls


def _empty_lineset():
    ls        = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.zeros((2, 3)))
    ls.lines  = o3d.utility.Vector2iVector([[0, 1]])
    ls.colors = o3d.utility.Vector3dVector([[0, 0, 0]])
    return ls


# ── Main ──────────────────────────────────────────────────────────────────────

def run(checkpoint, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — pass --device cpu")

    print("Loading GraspNet model...")
    model = load_model(checkpoint, device=device)

    isolator = ObjectIsolator(min_points=50)
    isolator.start()
    print("Waiting for YOLO to load...")
    isolator.ready.wait()
    print("YOLO ready — opening viewer.\n")
    print("Running — close the Open3D window or Ctrl+C to stop.\n")

    grasp_thread = GraspInferenceThread(model, device=device, interval=0.5)

    # ── Clean-exit flag — set by Ctrl+C or window close ──────────────────────
    _stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: _stop.set())

    # ── Window setup (same as show_isolated_pcd) ──────────────────────────────
    CV2_WIN = "YOLO Detection"
    cv2.namedWindow(CV2_WIN, cv2.WINDOW_NORMAL)

    vis = o3d.visualization.Visualizer()
    vis.create_window("GraspNet — Live Grasp", width=1280, height=720)
    opt = vis.get_render_option()
    opt.point_size       = 2.0
    opt.line_width       = 3.0
    opt.background_color = np.array([1.0, 1.0, 1.0])

    pcd         = o3d.geometry.PointCloud()
    lineset     = _empty_lineset()
    # Coordinate frame: X=red, Y=green, Z=blue (camera space, origin at camera)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05, origin=[0.0, 0.0, 0.0]
    )
    geom_added      = False
    zoom_fitted     = False   # set only after the first real isolated cloud
    last_grasp      = None    # track last grasp to avoid redundant updates
    lineset_ready   = False   # True once the real gripper lineset has been added

    RENDER_INTERVAL = 1.0 / 30.0   # cap renderer at 30 fps
    _last_render    = 0.0

    frame_ready = False

    while not _stop.is_set():
        # ── Pull latest frame (identical to show_isolated_pcd) ────────────────
        frame_ready = False
        try:
            verts, _raw_colors, full_colors, obj_verts, obj_colors, preview_bgr = \
                isolator._frame_queue.get_nowait()
            frame_ready = True

            # Point-cloud selection — same logic as show_isolated_pcd
            pts  = obj_verts  if len(obj_verts)  > 0 else verts
            cols = obj_colors if len(obj_colors) > 0 else full_colors

            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)

            if not geom_added:
                vis.add_geometry(pcd)
                vis.add_geometry(lineset)   # ← grasp overlay added once
                vis.add_geometry(coord_frame)
                geom_added = True
            else:
                vis.update_geometry(pcd)

            # Auto-zoom once on the first confirmed isolated cloud —
            # identical to show_isolated_pcd; camera is free afterwards
            if not zoom_fitted and len(obj_verts) > 0:
                iso_pcd = o3d.geometry.PointCloud()
                iso_pcd.points = o3d.utility.Vector3dVector(obj_verts)
                auto_zoom(vis, iso_pcd)
                zoom_fitted = True

            # cv2 YOLO preview — identical to show_isolated_pcd
            if preview_bgr is not None:
                cv2.imshow(CV2_WIN, preview_bgr)

            # ── GraspNet: feed isolated cloud ─────────────────────────────────
            if len(obj_verts) > 0:
                iso = o3d.geometry.PointCloud()
                iso.points = o3d.utility.Vector3dVector(obj_verts)
                iso.colors = o3d.utility.Vector3dVector(obj_colors)
                grasp_thread.update_pcd(iso)

        except queue.Empty:
            pass

        # Update grasp lineset only when a new result arrives
        grasp = grasp_thread.get_grasp()
        if grasp is not None and grasp is not last_grasp:
            last_grasp = grasp
            rot, trans, width = grasp
            new_ls = _gripper_lineset(rot, trans, width)
            if not lineset_ready and geom_added:
                # First real grasp: replace the empty placeholder with correct
                # topology (different point/line count) — remove then re-add.
                vis.remove_geometry(lineset, reset_bounding_box=False)
                lineset.points = new_ls.points
                lineset.lines  = new_ls.lines
                lineset.colors = new_ls.colors
                vis.add_geometry(lineset, reset_bounding_box=False)
                lineset_ready = True
            elif lineset_ready:
                lineset.points = new_ls.points
                lineset.lines  = new_ls.lines
                lineset.colors = new_ls.colors
                vis.update_geometry(lineset)

        # ── Service both GUI event loops — capped at 30 fps ─────────────────
        now = time.monotonic()
        if now - _last_render >= RENDER_INTERVAL:
            if not vis.poll_events():
                _stop.set()
                break
            vis.update_renderer()
            _last_render = now

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):   # ESC or q
            _stop.set()
            break

        # Yield when no new frame arrived to avoid spinning 100 % CPU
        if not frame_ready:
            time.sleep(0.005)

    # ── Ordered teardown ──────────────────────────────────────────────────────
    print("\n[test_lineset_live] shutting down...")
    grasp_thread.stop()
    isolator.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()
    print("[test_lineset_live] done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="GraspNet checkpoint (.tar)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.checkpoint, device=args.device)
