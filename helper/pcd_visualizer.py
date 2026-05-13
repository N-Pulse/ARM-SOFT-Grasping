"""
helper/pcd_visualizer.py

Reusable Open3D point-cloud visualiser for an ObjectIsolator stream.

Typical usage
-------------
from helper.pcd_visualizer import show_isolated_pcd
from capture.object_isolation import ObjectIsolator

isolator = ObjectIsolator(min_points=50)
isolator.start()
show_isolated_pcd(isolator)        # blocks until the window is closed
isolator.stop()
"""

from __future__ import annotations

import math
import queue
from typing import TYPE_CHECKING

import cv2
import open3d as o3d

if TYPE_CHECKING:
    from capture.object_isolation import ObjectIsolator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auto_zoom(vis: o3d.visualization.Visualizer,
               pcd: o3d.geometry.PointCloud) -> None:
    """Centre the cloud then zoom in so it fills most of the window.

    Uses ``reset_view_point`` for centering, then derives a single zoom value
    from the bounding box and applies it via ``set_zoom`` — no camera-matrix
    manipulation required.

    How the zoom is computed
    ------------------------
    After ``reset_view_point`` Open3D positions the camera at:
        distance = max_extent * 0.5 / tan(30°)  ≈  max_extent * 0.866

    ``set_zoom(z)`` narrows the field of view to ``90 * z`` degrees without
    moving the camera, so the visible height becomes:
        visible_h = 2 * distance * tan(45 * z °)

    Solving for z such that the Y-axis extent of the bounding box fills 80 %
    of visible_h:
        z = atan( vert_extent / (0.8 * 2 * distance) ) * (180/π) / 45
    """
    vis.reset_view_point(True)

    bbox     = pcd.get_axis_aligned_bounding_box()
    extents  = bbox.get_extent()           # (dx, dy, dz)
    max_ext  = max(extents)
    vert_ext = extents[1]                  # Y extent = vertical in default view

    # Camera distance set by reset_view_point (Open3D internal formula).
    distance = max_ext * 0.866

    # Zoom for ~80 % vertical fill, clamped to a sensible range.
    zoom = math.degrees(math.atan(vert_ext / (1.6 * distance))) / 45.0
    zoom = max(0.05, min(zoom, 1.5))

    vis.get_view_control().set_zoom(zoom)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def show_isolated_pcd(
    isolator: "ObjectIsolator",
    *,
    title: str = "Stage 1 — Isolated Object Point Cloud",
    width: int = 1280,
    height: int = 720,
    frame_timeout: float = 0.1,
) -> None:
    """Spin up an Open3D window and stream frames from *isolator*.

    On the first frame the cloud is centred and auto-zoomed so the object
    fills most of the window.  After that the user can freely orbit, pan,
    and zoom; the camera is not reset on subsequent frames.

    The function blocks until the window is closed or a KeyboardInterrupt is
    received.  The isolator is **not** stopped here — the caller decides when
    to stop it so the same isolator can be reused with other tools.

    Parameters
    ----------
    isolator:
        A started ``ObjectIsolator`` instance.
    title:
        Window title.
    width / height:
        Initial window size in pixels.
    frame_timeout:
        Seconds to wait on the frame queue before polling the window again.
    """
    CV2_WIN = "YOLO Detection"

    vis = o3d.visualization.Visualizer()
    vis.create_window(title, width=width, height=height)
    cv2.namedWindow(CV2_WIN, cv2.WINDOW_NORMAL)

    pcd        = o3d.geometry.PointCloud()
    geom_added = False

    print("[pcd_visualizer] Window open — close the Open3D window or press Ctrl+C to stop.")

    try:
        while True:
            # --- pull the latest frame -----------------------------------------
            try:
                verts, full_colors, obj_verts, obj_colors, preview_bgr = \
                    isolator._frame_queue.get(timeout=frame_timeout)

                # ── Open3D point cloud ──────────────────────────────────────
                pts  = obj_verts  if len(obj_verts)  > 0 else verts
                cols = obj_colors if len(obj_colors) > 0 else full_colors

                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(cols)

                if not geom_added:
                    vis.add_geometry(pcd)
                    _auto_zoom(vis, pcd)
                    geom_added = True
                else:
                    vis.update_geometry(pcd)

                # ── cv2 YOLO preview ────────────────────────────────────────
                if preview_bgr is not None:
                    cv2.imshow(CV2_WIN, preview_bgr)

            except queue.Empty:
                pass

            # --- service both GUI event loops ----------------------------------
            if not vis.poll_events():
                break
            vis.update_renderer()

            # cv2 needs a brief pump; ESC or 'q' also closes everything
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):   # ESC or q
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        vis.destroy_window()
        print("[pcd_visualizer] Windows closed.")
