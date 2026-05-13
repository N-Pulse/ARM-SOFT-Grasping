"""
test_obj_iso.py

Tests ObjectIsolator with a live Open3D visualiser + a cv2 preview window.

Changes from base version
--------------------------
  Point rendering  : point_size raised to 5.0 so points overlap visually
                     and gaps disappear without needing more geometry.

  Smoothing        : voxel down-sampling (VOXEL_SIZE=0.002) + statistical
                     outlier removal + radius outlier removal.
                     Voxel size is kept small (2 mm) to preserve density.

  View stability   : per-frame set_lookat removed — the camera view you
                     set manually is preserved across frames. View is
                     initialised once on startup only.

  Jetson load      : no normal estimation, no mesh reconstruction per frame.

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

# ─── Noise removal parameters ────────────────────────────────────────────────
# Voxel downsampling is intentionally NOT used here — it reduces point density
# and creates visible gaps. Instead, only outlier removal is applied so the
# full depth-image resolution is preserved.

# Statistical outlier removal
SOR_NEIGHBORS = 20     # kept light for Jetson speed
SOR_STD_RATIO = 1.2

# Radius outlier removal — removes isolated floating points
ROR_NB_POINTS = 6
ROR_RADIUS    = 0.012  # 1.2 cm

# ─── Rendering parameters ────────────────────────────────────────────────────
# With full point density preserved, point_size=2.0 gives a natural solid look
# without artificially inflating each point.
POINT_SIZE = 2.0


def smooth_pcd(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Apply statistical outlier removal + radius outlier removal.
    Voxel downsampling is deliberately skipped to preserve point density.
    Returns a new PointCloud; the input is not modified.
    """
    if len(pcd.points) < SOR_NEIGHBORS + 1:
        return pcd

    # Step 1 — drop statistical outliers
    filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=SOR_NEIGHBORS,
        std_ratio=SOR_STD_RATIO,
    )

    # Step 2 — drop isolated floating points
    filtered, _ = filtered.remove_radius_outlier(
        nb_points=ROR_NB_POINTS,
        radius=ROR_RADIUS,
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

    render_opt = vis.get_render_option()
    render_opt.point_size        = POINT_SIZE
    render_opt.background_color  = np.array([1.0, 1.0, 1.0])

    pcd        = o3d.geometry.PointCloud()
    geom_added = False
    view_set   = False   # ensure view is initialised only once

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

                # Copy smoothed data into the persistent pcd object
                pcd.points = smoothed.points
                pcd.colors = smoothed.colors

                # ── Add or update geometry ─────────────────────────────────
                if not geom_added:
                    vis.add_geometry(pcd)
                    geom_added = True
                else:
                    vis.update_geometry(pcd)

                # ── Set view once on first valid frame ─────────────────────
                # After this, the user can freely rotate/zoom without the
                # view being reset every frame.
                if not view_set and len(np.asarray(pcd.points)) > 0:
                    ctr = vis.get_view_control()
                    ctr.set_front([0, 0, -1])
                    ctr.set_up([0, -1, 0])
                    ctr.set_zoom(0.45)
                    view_set = True

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