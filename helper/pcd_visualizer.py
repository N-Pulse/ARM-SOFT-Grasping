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

import open3d as o3d

if TYPE_CHECKING:
    from capture.object_isolation import ObjectIsolator


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

    The view is auto-fitted on the first frame so the entire point cloud is
    visible and centred in the window.  After that the user can freely orbit,
    pan, and zoom; the view is not reset on subsequent frames.

    The function blocks until the window is closed or a KeyboardInterrupt
    is received.  The isolator is **not** stopped here — the caller decides
    when to stop it so the same isolator can be reused with other tools.

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
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, width=width, height=height)

    pcd        = o3d.geometry.PointCloud()
    geom_added = False

    print("[pcd_visualizer] Window open — close it or press Ctrl+C to stop.")

    try:
        while True:
            # --- pull the latest frame -----------------------------------------
            try:
                verts, full_colors, obj_verts, obj_colors, _ = \
                    isolator._frame_queue.get(timeout=frame_timeout)

                pts  = obj_verts  if len(obj_verts)  > 0 else verts
                cols = obj_colors if len(obj_colors) > 0 else full_colors

                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector(cols)

                if not geom_added:
                    vis.add_geometry(pcd)
                    # Auto-fit: centre the cloud and zoom so every point is
                    # inside the frustum.  reset_view_point computes the
                    # bounding-box centre as lookat and picks a zoom level
                    # that keeps the whole cloud visible — no manual tuning
                    # needed regardless of the cloud's actual position/scale.
                    vis.reset_view_point(True)
                    geom_added = True
                else:
                    vis.update_geometry(pcd)

            except queue.Empty:
                pass

            # --- service the GUI event loop ------------------------------------
            if not vis.poll_events():
                break
            vis.update_renderer()

    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()
        print("[pcd_visualizer] Window closed.")
