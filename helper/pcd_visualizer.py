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

def auto_zoom(vis: o3d.visualization.Visualizer,
              pcd: o3d.geometry.PointCloud) -> None:
    """Point the camera at the true centroid of the isolated cloud and zoom
    so its bounding box fills ~80 % of the window height.

    Steps
    -----
    1. ``reset_view_point(True)`` — establishes a sensible camera orientation
       and sets the internal distance:
           distance = max_extent * 0.5 / tan(30°)  ≈  max_extent * 0.866
    2. ``set_lookat(centroid)`` — re-targets the lookat to the mean of all
       points (not the geometric bbox centre), so the object is truly centred.
    3. ``set_zoom(z)`` — narrows the FOV to ``90 * z`` degrees without moving
       the camera.  The visible height then equals:
           visible_h = 2 * distance * tan(45 * z °)
       Solving for 80 % fill gives:
           z = atan( vert_extent / (1.6 * distance) ) * (180/π) / 45
    """
    vis.reset_view_point(True)

    # True centroid = mean of all isolated points (not the bbox centre).
    centroid = pcd.get_center()                # [x, y, z]

    bbox    = pcd.get_axis_aligned_bounding_box()
    extents = bbox.get_extent()                # [dx, dy, dz]
    max_ext = max(extents[0], extents[1], extents[2])
    # Use the larger of the two screen-facing axes (X and Y) so the whole
    # cloud fits inside 80 % regardless of whether it is wider or taller.
    screen_ext = max(extents[0], extents[1])

    # Distance formula matching Open3D's reset_view_point internals.
    distance = max_ext * 0.866

    # Zoom for 80 % fill, clamped to a usable range.
    zoom = math.degrees(math.atan(screen_ext / (1.6 * distance))) / 45.0
    zoom = max(0.05, min(zoom, 1.5))

    ctr = vis.get_view_control()
    ctr.set_lookat(centroid)   # centre on the point-cloud mean
    ctr.set_zoom(zoom)         # fill 80 % of the window


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
    on_new_frame=None,
    debug: bool = False,
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
    on_new_frame : callable(obj_verts, vis) | None
        Optional callback invoked every frame when isolated object points are
        available.  Ignored when ``debug=True``.
    debug : bool
        When True, always display the full scene point cloud (object in real
        colour, background grayed out) and auto-zoom on the first frame
        regardless of whether an object has been isolated.  The
        ``on_new_frame`` callback is skipped.  Useful for verifying camera
        coverage and depth filtering without needing a detected object.
    """
    CV2_WIN = "Camera Preview"

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(title, width=width, height=height)
    cv2.namedWindow(CV2_WIN, cv2.WINDOW_NORMAL)

    pcd         = o3d.geometry.PointCloud()
    geom_added  = False   # True after add_geometry; camera is set on that same frame
    zoom_fitted = False   # True after auto_zoom fires on first isolated cloud

    print("[pcd_visualizer] Window open — close the Open3D window or press Ctrl+C to stop.")
    if debug:
        print("[pcd_visualizer] DEBUG MODE — showing full point cloud.")

    try:
        while True:
            # --- pull the latest frame -----------------------------------------
            try:
                verts, raw_colors, full_colors, obj_verts, obj_colors, preview_bgr = \
                    isolator._frame_queue.get(timeout=frame_timeout)

                # ── Choose which points/colours to display ──────────────────
                if debug:
                    # Full scene with original, unmodified colours.
                    pts  = verts
                    cols = raw_colors
                else:
                    pts  = obj_verts  if len(obj_verts)  > 0 else verts
                    cols = obj_colors if len(obj_colors) > 0 else full_colors

                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(cols)

                if not geom_added:
                    vis.add_geometry(pcd)
                    if debug:
                        # Debug: fixed camera matching test_pointcloud_open3d.py
                        ctr = vis.get_view_control()
                        ctr.set_lookat([0, 0, 0.4])
                        ctr.set_front([0, 0, -1])
                        ctr.set_up([0, -1, 0])
                        ctr.set_zoom(0.2)
                    geom_added = True
                else:
                    vis.update_geometry(pcd)

                # Auto-zoom once onto the first isolated object cloud (normal mode).
                if not debug and not zoom_fitted and len(obj_verts) > 0:
                    iso_pcd = o3d.geometry.PointCloud()
                    iso_pcd.points = o3d.utility.Vector3dVector(obj_verts)
                    auto_zoom(vis, iso_pcd)
                    zoom_fitted = True

                # Optional per-frame callback (skipped in debug mode)
                if not debug and on_new_frame is not None and len(obj_verts) > 0:
                    on_new_frame(obj_verts, vis)

                # ── cv2 preview ─────────────────────────────────────────────
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
