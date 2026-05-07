"""
test_obj_iso.py

Tests ObjectIsolator with a live Open3D visualiser.
Waits for YOLO to finish loading before opening the window
so Open3D and YOLO don't fight over GPU memory.

Usage:
    python test_obj_iso.py

Controls:
    Close the Open3D window or press Ctrl+C to stop.
"""

import sys
import os
import queue
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "capture"))

import open3d as o3d
from object_isolation import ObjectIsolator


def run():
    isolator = ObjectIsolator(min_points=50)
    isolator.start()

    print("Waiting for YOLO to load...")
    isolator.ready.wait()
    print("YOLO ready — opening window.\n")

    vis = o3d.visualization.Visualizer()
    vis.create_window("Stage 1 — Isolated Object Point Cloud", width=1280, height=720)
    pcd        = o3d.geometry.PointCloud()
    geom_added = False

    print("Running — close the window or press Ctrl+C to stop.")

    try:
        while True:
            try:
                verts, full_colors, obj_verts, obj_colors, _ = \
                    isolator._frame_queue.get(timeout=0.1)

                pts  = obj_verts  if len(obj_verts)  > 0 else verts
                cols = obj_colors if len(obj_colors) > 0 else full_colors

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

            except queue.Empty:
                pass

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