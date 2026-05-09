"""
test_obj_iso.py
---------------
Tests ObjectIsolator with a live Open3D point-cloud window + cv2 preview.
Uses O3DVisualizer for all rendering boilerplate.

Controls
--------
  Close the Open3D window  |  press 'q' in the cv2 window  |  Ctrl+C  → stop
"""

import sys
import os
import queue

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "capture"))

import cv2
import numpy as np
import open3d as o3d
from object_isolation import ObjectIsolator
from o3d_visualizer import O3DVisualizer


def run():
    isolator = ObjectIsolator(min_points=50)
    isolator.start()

    print("Waiting for YOLO to load...")
    isolator.ready.wait()
    print("YOLO ready — opening windows.\n")

    pcd        = o3d.geometry.PointCloud()
    first_frame = True
    CV2_WIN    = "YOLO Preview"

    print("Running — close the Open3D window, press 'q' in the cv2 window,")
    print("or Ctrl+C to stop.\n")

    with O3DVisualizer(
        title      = "Stage 1 — Isolated Object Point Cloud",
        width      = 1280,
        height     = 720,
        point_size = 3.0,
        bg_color   = (1.0, 1.0, 1.0),   # white background
        voxel_size = 0.004,
        sor_k      = 30,
        sor_std    = 1.5,
    ) as vis:

        try:
            while True:
                # ── Pull the latest frame ──────────────────────────────────
                try:
                    verts, full_colors, obj_verts, obj_colors, preview_bgr = \
                        isolator._frame_queue.get(timeout=0.1)

                    # Prefer isolated object; fall back to full scene
                    pts  = obj_verts  if len(obj_verts)  > 0 else verts
                    cols = obj_colors if len(obj_colors) > 0 else full_colors

                    # Build raw pcd, smooth it, copy back into persistent object
                    raw_pcd = o3d.geometry.PointCloud()
                    raw_pcd.points = o3d.utility.Vector3dVector(pts)
                    raw_pcd.colors = o3d.utility.Vector3dVector(cols)

                    smoothed       = vis.smooth_pcd(raw_pcd)
                    pcd.points     = smoothed.points
                    pcd.colors     = smoothed.colors

                    # Add on first frame, update thereafter
                    vis.add_or_update(pcd)

                    if first_frame:
                        vis.set_view(front=(0, 0, -1), up=(0, -1, 0), zoom=0.45)
                        first_frame = False

                    # Keep object centred in the window
                    vis.centre_on_pcd(pcd)

                    # cv2 preview
                    cv2.imshow(CV2_WIN, preview_bgr)

                except queue.Empty:
                    pass

                # ── Render tick ────────────────────────────────────────────
                if not vis.tick():
                    break

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            isolator.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run()