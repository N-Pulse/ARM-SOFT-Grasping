"""
o3d_visualizer.py
-----------------
Thin wrapper around o3d.visualization.VisualizerWithKeyCallback that
bundles the patterns repeated across all our scripts:

  • window creation with render options
  • add / update geometry
  • smoothing  (voxel down-sample + statistical outlier removal)
  • auto-centre lookat to point-cloud centroid
  • key-callback registration
  • one-liner render tick  (poll + render)
"""

import numpy as np
import open3d as o3d
from typing import Callable, Optional


# ── Smoothing defaults (override via constructor or smooth_pcd kwargs) ─────────

_DEFAULT_VOXEL    = 0.004   # 4 mm voxels
_DEFAULT_SOR_K    = 30      # neighbours for outlier removal
_DEFAULT_SOR_STD  = 1.5     # std-ratio threshold


class O3DVisualizer:
    """
    Convenience wrapper around VisualizerWithKeyCallback.

    Parameters
    ----------
    title       : window title
    width, height : window size in pixels
    point_size  : rendered point radius
    bg_color    : background RGB in [0, 1]
    line_width  : line width for LineSets
    voxel_size  : voxel size (m) used by smooth_pcd; None = skip downsampling
    sor_k       : SOR neighbour count;  None = skip SOR
    sor_std     : SOR std-ratio threshold
    """

    def __init__(
        self,
        title:      str   = "Open3D Viewer",
        width:      int   = 1280,
        height:     int   = 800,
        point_size: float = 2.0,
        bg_color:   tuple = (0.10, 0.10, 0.10),
        line_width: float = 3.0,
        voxel_size: Optional[float] = _DEFAULT_VOXEL,
        sor_k:      Optional[int]   = _DEFAULT_SOR_K,
        sor_std:    float           = _DEFAULT_SOR_STD,
    ):
        self._voxel   = voxel_size
        self._sor_k   = sor_k
        self._sor_std = sor_std

        self._vis = o3d.visualization.VisualizerWithKeyCallback()
        self._vis.create_window(window_name=title, width=width, height=height)

        opt = self._vis.get_render_option()
        opt.point_size       = point_size
        opt.background_color = np.array(bg_color)
        opt.line_width       = line_width

        self._geoms: set = set()   # track which geometry objects have been added

    # ── geometry management ────────────────────────────────────────────────────

    def add(self, geom) -> None:
        """Add a geometry to the scene (idempotent — safe to call twice)."""
        if id(geom) not in self._geoms:
            self._vis.add_geometry(geom)
            self._geoms.add(id(geom))

    def update(self, geom) -> None:
        """
        Push in-place changes of an already-added geometry to the renderer.
        If the geometry has not been added yet, this calls add() first.
        """
        if id(geom) not in self._geoms:
            self.add(geom)
        else:
            self._vis.update_geometry(geom)

    def add_or_update(self, geom) -> None:
        """Convenience: add on first call, update on subsequent calls."""
        self.update(geom)   # update() already handles the first-time case

    def remove(self, geom) -> None:
        """Remove a geometry from the scene."""
        self._vis.remove_geometry(geom)
        self._geoms.discard(id(geom))

    # ── smoothing ──────────────────────────────────────────────────────────────

    def smooth_pcd(
        self,
        pcd:        o3d.geometry.PointCloud,
        voxel_size: Optional[float] = None,
        sor_k:      Optional[int]   = None,
        sor_std:    Optional[float] = None,
    ) -> o3d.geometry.PointCloud:
        """
        Return a smoothed copy of *pcd*:
          1. Voxel down-sample  (merges duplicate/nearby points)
          2. Statistical outlier removal  (drops isolated noise)

        Per-call arguments override the instance defaults.
        Pass voxel_size=0 / sor_k=0 to skip that step for this call.
        """
        voxel = voxel_size if voxel_size is not None else self._voxel
        k     = sor_k     if sor_k     is not None else self._sor_k
        std   = sor_std   if sor_std   is not None else self._sor_std

        result = pcd

        if voxel and voxel > 0:
            result = result.voxel_down_sample(voxel_size=voxel)

        min_pts = (k or 0) + 1
        if k and k > 0 and len(result.points) >= min_pts:
            result, _ = result.remove_statistical_outlier(
                nb_neighbors=k, std_ratio=std
            )

        return result

    # ── camera helpers ─────────────────────────────────────────────────────────

    def set_view(
        self,
        front:  tuple = (0, 0, -1),
        up:     tuple = (0, -1,  0),
        zoom:   float = 0.45,
        lookat: Optional[tuple] = None,
    ) -> None:
        """Set a fixed viewing angle.  Call once after the first add()."""
        ctr = self._vis.get_view_control()
        ctr.set_front(list(front))
        ctr.set_up(list(up))
        ctr.set_zoom(zoom)
        if lookat is not None:
            ctr.set_lookat(list(lookat))

    def centre_on_pcd(self, pcd: o3d.geometry.PointCloud) -> None:
        """Move the camera lookat to the centroid of *pcd* (auto-centre)."""
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return
        centroid = pts.mean(axis=0)
        self._vis.get_view_control().set_lookat(centroid.tolist())

    def centre_on_points(self, pts: np.ndarray) -> None:
        """Move the camera lookat to the centroid of a raw (N,3) array."""
        if len(pts) == 0:
            return
        self._vis.get_view_control().set_lookat(pts.mean(axis=0).tolist())

    # ── key callbacks ──────────────────────────────────────────────────────────

    def on_key(self, key: str, callback: Callable) -> None:
        """
        Register a key callback.
        *key* is a single character string, e.g. "R".
        *callback* receives the VisualizerWithKeyCallback instance and must
        return False (Open3D convention).
        """
        self._vis.register_key_callback(ord(key.upper()), callback)

    # ── render loop helpers ────────────────────────────────────────────────────

    def tick(self) -> bool:
        """
        Process one render frame.
        Returns False when the window has been closed (→ exit your loop).
        """
        if not self._vis.poll_events():
            return False
        self._vis.update_renderer()
        return True

    def close(self) -> None:
        """Destroy the window and release resources."""
        self._vis.destroy_window()

    # ── context manager ────────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()