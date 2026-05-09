"""
test_obj_iso.py

Tests ObjectIsolator with a live Open3D visualiser + a cv2 preview window.
Uses O3DVisualizer helper — behaviour is identical to the original.

Controls:
    Close the Open3D window  |  press 'q' in the cv2 window  |  Ctrl+C
"""

import sys
import os
import queue

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "capture"))
sys.path.insert(0, os.path.join(_HERE, "..", "helper"))

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

    pcd         = o3d.geometry.PointCloud()
    first_frame = True
    CV2_WIN     = "YOLO Preview"

    print("Running — close the Open3D window, press 'q' in the cv2 window,")
    print("or Ctrl+C to stop.\n")

    with O3DVisualizer(
        title      = "Stage 1 — Isolated Object Point Cloud",
        width      = 1280,
        height     = 720,
        point_size = 3.0,
        bg_color   = (1.0, 1.0, 1.0),
        voxel_size = 0.004,
        sor_k      = 30,
        sor_std    = 1.5,
        # plain Visualizer (default) — matches original exactly
    ) as vis:

        try:
            while True:
                try:
                    verts, full_colors, obj_verts, obj_colors, preview_bgr = \
                        isolator._frame_queue.get(timeout=0.1)

                    pts  = obj_verts  if len(obj_verts)  > 0 else verts
                    cols = obj_colors if len(obj_colors) > 0 else full_colors

                    raw_pcd        = o3d.geometry.PointCloud()
                    raw_pcd.points = o3d.utility.Vector3dVector(pts)
                    raw_pcd.colors = o3d.utility.Vector3dVector(cols)

                    smoothed   = vis.smooth_pcd(raw_pcd)
                    pcd.points = smoothed.points
                    pcd.colors = smoothed.colors

                    vis.add_or_update(pcd)

                    # set_view called once on first frame, after add — same as
                    # the original's  if not geom_added: ... set_front/up/zoom
                    if first_frame:
                        vis.set_view(front=(0, 0, -1), up=(0, -1, 0), zoom=0.45)
                        first_frame = False

                    # lookat updated every frame — same as original
                    vis.centre_on_pcd(pcd)

                    cv2.imshow(CV2_WIN, preview_bgr)

                except queue.Empty:
                    pass

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