"""
o3d_visualizer.py
-----------------
Minimal pass-through wrapper around Open3D's Visualizer.
The internal Open3D object is exposed as `vis.raw` for any escape-hatch use.

Bundled helpers
---------------
  smooth_pcd(pcd)            voxel down-sample + statistical outlier removal
  vis.set_render_options()   point_size / bg / line_width
  vis.set_view()             front / up / zoom (call once)
  vis.centre_on_pcd(pcd)     set_lookat to centroid (every frame)
  vis.add_or_update(geom)    auto add on first call, update after
  vis.tick()                 poll_events + update_renderer
"""

import numpy as np
import open3d as o3d
from typing import Callable, Optional


# ── Standalone smoothing function (no visualizer needed) ──────────────────────

def smooth_pcd(
    pcd:        o3d.geometry.PointCloud,
    voxel_size: Optional[float] = 0.004,
    sor_k:      Optional[int]   = 30,
    sor_std:    float           = 1.5,
) -> o3d.geometry.PointCloud:
    """
    Voxel down-sample then statistical outlier removal.
    Returns a new PointCloud; input is not modified.
    Pass voxel_size=None / sor_k=None to skip a step.
    """
    result = pcd
    if voxel_size and voxel_size > 0:
        result = result.voxel_down_sample(voxel_size=voxel_size)
    if sor_k and sor_k > 0 and len(result.points) >= sor_k + 1:
        result, _ = result.remove_statistical_outlier(
            nb_neighbors=sor_k, std_ratio=sor_std)
    return result


# ── Pass-through wrapper ──────────────────────────────────────────────────────

class O3DVisualizer:
    """
    Pass-through wrapper.  Owns an o3d.visualization.Visualizer (or
    VisualizerWithKeyCallback) and exposes it via `self.raw`.
    """

    def __init__(
        self,
        title:             str   = "Open3D Viewer",
        width:             int   = 1280,
        height:            int   = 800,
        use_key_callbacks: bool  = False,
    ):
        if use_key_callbacks:
            self._vis = o3d.visualization.VisualizerWithKeyCallback()
        else:
            self._vis = o3d.visualization.Visualizer()

        self._vis.create_window(window_name=title, width=width, height=height)

        # Keep persistent references so the underlying C++ objects stay
        # bound to live Python wrappers for the lifetime of the window.
        self.render_option = self._vis.get_render_option()
        self.view_control  = self._vis.get_view_control()

        self._geoms: set = set()

    # Escape hatch — the underlying Open3D Visualizer
    @property
    def raw(self):
        return self._vis

    # ── render options (call once after construction, before add) ──────────────

    def set_render_options(
        self,
        point_size: float = 3.0,
        bg_color:   tuple = (1.0, 1.0, 1.0),
        line_width: float = 1.0,
    ) -> None:
        self.render_option.point_size       = point_size
        self.render_option.background_color = np.array(bg_color, dtype=np.float64)
        self.render_option.line_width       = line_width

    # ── geometry ───────────────────────────────────────────────────────────────

    def add_or_update(self, geom) -> None:
        """First call → add_geometry; subsequent calls → update_geometry."""
        if id(geom) not in self._geoms:
            self._vis.add_geometry(geom)
            self._geoms.add(id(geom))
        else:
            self._vis.update_geometry(geom)

    def remove(self, geom) -> None:
        self._vis.remove_geometry(geom)
        self._geoms.discard(id(geom))

    # ── camera ─────────────────────────────────────────────────────────────────

    def set_view(
        self,
        front: tuple = (0, 0, -1),
        up:    tuple = (0, -1, 0),
        zoom:  float = 0.45,
    ) -> None:
        """Set front/up/zoom — call once on first frame, after add."""
        self.view_control.set_front(list(front))
        self.view_control.set_up(list(up))
        self.view_control.set_zoom(zoom)

    def centre_on_pcd(self, pcd: o3d.geometry.PointCloud) -> None:
        """set_lookat to centroid — safe to call every frame."""
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return
        self.view_control.set_lookat(pts.mean(axis=0).tolist())

    # ── key callbacks (only with use_key_callbacks=True) ───────────────────────

    def on_key(self, key: str, callback: Callable) -> None:
        if not isinstance(self._vis, o3d.visualization.VisualizerWithKeyCallback):
            raise RuntimeError(
                "on_key() requires use_key_callbacks=True in the constructor.")
        self._vis.register_key_callback(ord(key.upper()), callback)

    # ── render loop ────────────────────────────────────────────────────────────

    def tick(self) -> bool:
        """poll_events + update_renderer.  False → window closed, exit loop."""
        if not self._vis.poll_events():
            return False
        self._vis.update_renderer()
        return True

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._vis.destroy_window()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()