"""
test_obj_iso.py

Tests ObjectIsolator with a live Open3D visualiser + cv2 YOLO preview.

Usage:
    python test_obj_iso.py

Controls:
    Press q in the cv2 window, or close the Open3D window to stop.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "capture"))

import cv2
import open3d as o3d
from object_isolation import ObjectIsolator


def run():
    vis        = o3d.visualization.Visualizer()
    vis.create_window("Stage 1 — Isolated Object Point Cloud", width=1280, height=720)
    pcd        = o3d.geometry.PointCloud()
    geom_added = False

    print("Starting ObjectIsolator...")
    print("Close the Open3D window or press q in the preview to stop.\n")

    with ObjectIsolator(min_points=50) as isolator:
        try:
            while True:
                result = isolator.get_full_frame()

                if result is not None:
                    full_pcd, iso_pcd, preview_bgr = result

                    # Open3D: show isolated object (fall back to full scene)
                    target_pcd = iso_pcd if iso_pcd is not None else full_pcd
                    pcd.points = target_pcd.points
                    pcd.colors = target_pcd.colors

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

                    # cv2: YOLO segmentation overlay
                    cv2.imshow("Stage 1 — YOLO Preview", preview_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if not vis.poll_events():
                    break
                vis.update_renderer()

        except KeyboardInterrupt:
            pass
        finally:
            vis.destroy_window()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run()