"""
test_obj_iso.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "capture"))

import numpy as np
import open3d as o3d
from object_isolation import ObjectIsolator

def run():
    isolator = ObjectIsolator(min_points=50)
    isolator.start()

    print("Waiting for YOLO to load...")
    isolator.ready.wait()
    print("YOLO ready — opening window.")

    vis = o3d.visualization.Visualizer()
    vis.create_window("Stage 1 — Isolated Object Point Cloud", width=1280, height=720)
    pcd        = o3d.geometry.PointCloud()
    geom_added = False

    print("Running Stage 1 — YOLO segmentation + point cloud isolation.")
    print("Close the window or press Ctrl+C to stop.\n")

    try:
        while True:
            # mirror monolithic: block up to 100ms waiting for a frame
            result = isolator._frame_queue.get(timeout=0.1) if not isolator._frame_queue.empty() else None

            if result is not None:
                verts, full_colors, obj_verts, obj_colors, _ = result

                # use isolated object if available, else full scene
                if len(obj_verts) > 0:
                    pts, cols = obj_verts, obj_colors
                else:
                    pts, cols = verts, full_colors

                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(cols)

                if not geom_added:
                    vis.add_geometry(pcd)
                    ctr = vis.get_view_control()
                    ctr.set_lookat([0, 0, 0.3])
                    ctr.set_front([0, 0, -1])
                    ctr.set_up([0, -1, 0])
                    ctr.set_zoom(0.25)
                    geom_added = True
                else:
                    vis.update_geometry(pcd)

            if not vis.poll_events():
                break
            vis.update_renderer()

    except KeyboardInterrupt:
        pass
    finally:
        isolator.stop()
        vis.destroy_window()

if __name__ == "__main__":
    run()