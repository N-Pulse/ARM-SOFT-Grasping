"""
test_obj_iso.py

Tests ObjectIsolator with a live Open3D visualiser + a cv2 preview window.
Uses O3DVisualizer wrapper (pass-through, identical Open3D call sequence).
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
from o3d_visualizer import O3DVisualizer, smooth_pcd


VOXEL_SIZE    = 0.004
SOR_NEIGHBORS = 30
SOR_STD_RATIO = 1.5


def run():
    isolator = ObjectIsolator(min_points=50)
    isolator.start()

    print("Waiting for YOLO to load...")
    isolator.ready.wait()
    print("YOLO ready — opening windows.\n")

    # ── Open3D point-cloud window ──────────────────────────────────────────
    vis = O3DVisualizer(
        title="Stage 1 — Isolated Object Point Cloud",
        width=1280, height=720,
    )
    vis.set_render_options(point_size=3.0, bg_color=(1.0, 1.0, 1.0))

    pcd        = o3d.geometry.PointCloud()
    geom_added = False

    CV2_WIN = "YOLO Preview"

    print("Running — close the Open3D window, press 'q' in the cv2 window,")
    print("or Ctrl+C to stop.\n")

    try:
        while True:
            # ── Pull the latest frame ──────────────────────────────────────
            try:
                verts, full_colors, obj_verts, obj_colors, preview_bgr = \
                    isolator._frame_queue.get(timeout=0.1)

                pts  = obj_verts  if len(obj_verts)  > 0 else verts
                cols = obj_colors if len(obj_colors) > 0 else full_colors

                # Build raw pcd then smooth it
                raw_pcd = o3d.geometry.PointCloud()
                raw_pcd.points = o3d.utility.Vector3dVector(pts)
                raw_pcd.colors = o3d.utility.Vector3dVector(cols)

                smoothed = smooth_pcd(
                    raw_pcd,
                    voxel_size = VOXEL_SIZE,
                    sor_k      = SOR_NEIGHBORS,
                    sor_std    = SOR_STD_RATIO,
                )

                pcd.points = smoothed.points
                pcd.colors = smoothed.colors

                # ── Add or update geometry ─────────────────────────────────
                if not geom_added:
                    vis.add_or_update(pcd)         # first call → add_geometry
                    geom_added = True
                    vis.set_view(front=(0, 0, -1), up=(0, -1, 0), zoom=0.45)
                else:
                    vis.add_or_update(pcd)         # subsequent → update_geometry

                # ── Auto-centre ────────────────────────────────────────────
                vis.centre_on_pcd(pcd)

                cv2.imshow(CV2_WIN, preview_bgr)

            except queue.Empty:
                pass

            # ── Pump Open3D events ─────────────────────────────────────────
            if not vis.tick():
                break

            # ── Pump cv2 events; 'q' quits ─────────────────────────────────
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        isolator.stop()
        vis.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()