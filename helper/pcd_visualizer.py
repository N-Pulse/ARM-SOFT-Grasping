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

import queue
from typing import TYPE_CHECKING

import cv2
import numpy as np
import open3d as o3d

if TYPE_CHECKING:
    from capture.object_isolation import ObjectIsolator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_camera_80pct(vis: o3d.visualization.Visualizer,
                      pcd: o3d.geometry.PointCloud,
                      width: int,
                      height: int) -> None:
    """Reposition the camera so the cloud's vertical span fills 80 % of the
    window height while keeping the cloud centred and fully visible.

    Strategy
    --------
    1. Call ``reset_view_point`` to centre the cloud and obtain a well-oriented
       camera (lookat = bbox centre, reasonable front/up).
    2. Read back the resulting PinholeCameraParameters — this gives us the
       exact rotation matrix R and intrinsic focal length fy.
    3. Project all 8 bounding-box corners onto the camera's vertical axis to
       get the true screen-space vertical span of the cloud (axis-aligned bbox
       projected extent, not the bounding sphere).
    4. Compute the depth ``d`` such that:
           (fy * vertical_span) / (d * height) = 0.80
       i.e. the span covers 80 % of the image height.
    5. Move the camera to ``bbox_centre + d * (-forward)``, keeping orientation.
    """
    # 1. Let Open3D centre the cloud and set a sensible orientation.
    vis.reset_view_point(True)
    vis.poll_events()
    vis.update_renderer()

    ctr = vis.get_view_control()
    try:
        params = ctr.convert_to_pinhole_camera_parameters()
    except Exception:
        # Fallback: reset_view_point already did something reasonable.
        return

    # Camera rotation and translation (world → camera).
    R = params.extrinsic[:3, :3]          # (3,3)
    t = params.extrinsic[:3, 3]           # (3,)

    # Camera position in world coords.
    cam_pos = -R.T @ t                    # (3,)

    # Camera forward axis in world coords (column 2 of R^T).
    forward = R.T[:, 2]                   # unit vector

    # Bbox centre.
    bbox   = pcd.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center())

    # 2. Project 8 bbox corners onto camera Y axis to get vertical screen span.
    #    Camera Y axis in world = column 1 of R^T.
    cam_y = R.T[:, 1]                     # unit vector (screen-down direction)
    corners = np.asarray(bbox.get_box_points())   # (8, 3)
    proj_y  = corners @ cam_y
    vert_span = float(proj_y.max() - proj_y.min())

    # Also project onto camera X axis (horizontal) so we can ensure the full
    # cloud fits both ways (use whichever constraint is tighter).
    cam_x     = R.T[:, 0]
    proj_x    = corners @ cam_x
    horiz_span = float(proj_x.max() - proj_x.min())

    # 3. Focal lengths from intrinsics.
    K  = params.intrinsic.intrinsic_matrix   # (3,3)
    fy = float(K[1, 1])
    fx = float(K[0, 0])

    TARGET = 0.80  # fill fraction

    # Required depth so vert_span fills TARGET of image height:
    #   fy * vert_span / d = TARGET * height   →   d = fy * vert_span / (TARGET * height)
    d_vert  = fy * vert_span  / (TARGET * height)
    # Same for horizontal:
    d_horiz = fx * horiz_span / (TARGET * width)

    # Use the larger distance so BOTH extents are within the window.
    d = max(d_vert, d_horiz)

    # 4. Reposition camera along the forward axis.
    new_cam_pos  = center - d * forward

    # 5. Rebuild extrinsic with new camera position, same rotation.
    new_t = -R @ new_cam_pos
    new_extrinsic        = params.extrinsic.copy()
    new_extrinsic[:3, 3] = new_t
    params.extrinsic     = new_extrinsic

    try:
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    except TypeError:
        # Older Open3D builds don't accept allow_arbitrary.
        ctr.convert_from_pinhole_camera_parameters(params)


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

    On the first frame the camera is positioned so the cloud is centred and
    its vertical extent fills 80 % of the window height (while keeping the
    whole cloud visible).  After that the user can freely orbit, pan, and zoom;
    the camera is not reset on subsequent frames.

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
                    _fit_camera_80pct(vis, pcd, width, height)
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
