"""
test_obj_iso.py

Tests ObjectIsolator with a live Open3D visualiser + a cv2 preview window.

Improvements over the base version
------------------------------------
  Auto-centring  : the camera lookat is updated every frame to follow the
                   centroid of the current point cloud, so the object is
                   always in the middle of the window regardless of where
                   it sits in 3-D space.

  Smoothing      : two Open3D post-processing steps are applied before
                   rendering each frame:
                     1. Voxel down-sampling  — merges nearby points into one
                        representative point, removing jagged duplicate
                        clusters and giving uniform density.
                     2. Statistical outlier removal — drops isolated noisy
                        points that sit far from their neighbours.

Usage:
    python test_obj_iso.py

Controls:
    Close the Open3D window OR press 'q' in the cv2 window OR Ctrl+C to stop.
"""

import sys
import os
import queue

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "capture"))

import cv2
import numpy as np
import open3d as o3d
from object_isolation import ObjectIsolator

# ─── Smoothing parameters (tune these to taste) ───────────────────────────────

# Voxel size in metres.  Larger = fewer, chunkier points; smaller = more detail.
# 0.004 m (4 mm) is a good starting point for hand-sized objects at 10–40 cm.
VOXEL_SIZE = 0.004

# Statistical outlier removal.
# nb_neighbors : how many neighbours to consider for the mean-distance check
# std_ratio    : points further than (mean + std_ratio * std) are removed;
#                lower = more aggressive pruning
SOR_NEIGHBORS = 30
SOR_STD_RATIO = 1.5


def smooth_pcd(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Apply voxel down-sampling then statistical outlier removal.
    Returns a new PointCloud; the input is not modified.
    Returns the input unchanged if it has fewer than SOR_NEIGHBORS+1 points
    (outlier removal needs at least that many).
    """
    if len(pcd.points) < SOR_NEIGHBORS + 1:
        return pcd

    # Step 1 — merge nearby points into a single point per voxel cell
    down = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

    # Step 2 — drop points that are statistical outliers among their neighbours
    filtered, _ = down.remove_statistical_outlier(
        nb_neighbors=SOR_NEIGHBORS,
        std_ratio=SOR_STD_RATIO,
    )
    return filtered


def run():
    isolator = ObjectIsolator(min_points=50)
    isolator.start()

    print("Waiting for YOLO to load...")
    isolator.ready.wait()
    print("YOLO ready — opening windows.\n")

    # ── Open3D point-cloud window ──────────────────────────────────────────
    vis = o3d.visualization.Visualizer()
    vis.create_window("Stage 1 — Isolated Object Point Cloud", width=1280, height=720)

    # Increase point size so individual points are easier to see
    render_opt = vis.get_render_option()
    render_opt.point_size = 3.0
    render_opt.background_color = np.array([0.1, 0.1, 0.1])  # dark grey background

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

                # Prefer isolated object; fall back to full scene
                pts  = obj_verts  if len(obj_verts)  > 0 else verts
                cols = obj_colors if len(obj_colors) > 0 else full_colors

                # ── Build raw pcd then smooth it ───────────────────────────
                raw_pcd = o3d.geometry.PointCloud()
                raw_pcd.points = o3d.utility.Vector3dVector(pts)
                raw_pcd.colors = o3d.utility.Vector3dVector(cols)

                smoothed = smooth_pcd(raw_pcd)

                # Copy smoothed data back into the persistent pcd object so
                # Open3D's update_geometry works on the same reference
                pcd.points = smoothed.points
                pcd.colors = smoothed.colors

                # ── Add or update geometry ─────────────────────────────────
                if not geom_added:
                    vis.add_geometry(pcd)
                    geom_added = True

                    # Set a fixed viewing angle once on startup;
                    # lookat is updated every frame below
                    ctr = vis.get_view_control()
                    ctr.set_front([0, 0, -1])
                    ctr.set_up([0, -1, 0])
                    ctr.set_zoom(0.45)
                else:
                    vis.update_geometry(pcd)

                # ── Auto-centre: move lookat to the point-cloud centroid ───
                # This keeps the object in the middle of the window even when
                # the hand or robot moves it around in 3-D space.
                pts_np = np.asarray(pcd.points)
                if len(pts_np) > 0:
                    centroid = pts_np.mean(axis=0)
                    ctr = vis.get_view_control()
                    ctr.set_lookat(centroid.tolist())

                # ── cv2 preview ────────────────────────────────────────────
                cv2.imshow(CV2_WIN, preview_bgr)

            except queue.Empty:
                pass

            # ── Pump Open3D events ─────────────────────────────────────────
            if not vis.poll_events():
                break
            vis.update_renderer()

            # ── Pump cv2 events; 'q' quits ─────────────────────────────────
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        isolator.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()