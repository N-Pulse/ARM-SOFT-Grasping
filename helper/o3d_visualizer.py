"""
o3d_visualizer.py
-----------------
Thin wrapper around Open3D's Visualizer that bundles repeated patterns:

  • window creation with render options
  • add / update geometry  (first-time add auto-detected)
  • smoothing  (voxel down-sample + statistical outlier removal)
  • auto-centre lookat to point-cloud centroid
  • one-liner render tick  (poll + render)
  • optional key callbacks (switches to VisualizerWithKeyCallback)

Design goal: replicate the exact Open3D call sequence of the hand-written
scripts so rendering is pixel-identical — no surprises.
"""

import numpy as np
import open3d as o3d
from typing import Callable, Optional


_DEFAULT_VOXEL   = 0.004   # 4 mm
_DEFAULT_SOR_K   = 30
_DEFAULT_SOR_STD = 1.5


class O3DVisualizer:
    """
    Parameters
    ----------
    title             : window title
    width, height     : window size in pixels
    point_size        : rendered point radius
    bg_color          : background RGB in [0, 1]
    line_width        : line width for LineSets
    voxel_size        : voxel size (m) for smooth_pcd; None = skip
    sor_k             : SOR neighbour count; None = skip
    sor_std           : SOR std-ratio threshold
    use_key_callbacks : if True uses VisualizerWithKeyCallback (needed for
                        on_key()); if False uses plain Visualizer (default,
                        matches original test_obj_iso behaviour exactly)
    """

    def __init__(
        self,
        title:             str   = "Open3D Viewer",
        width:             int   = 1280,
        height:            int   = 800,
        point_size:        float = 2.0,
        bg_color:          tuple = (0.10, 0.10, 0.10),
        line_width:        float = 3.0,
        voxel_size:        Optional[float] = _DEFAULT_VOXEL,
        sor_k:             Optional[int]   = _DEFAULT_SOR_K,
        sor_std:           float           = _DEFAULT_SOR_STD,
        use_key_callbacks: bool            = False,
    ):
        self._voxel      = voxel_size
        self._sor_k      = sor_k
        self._sor_std    = sor_std
        self._point_size = point_size
        self._bg_color   = np.array(bg_color, dtype=np.float64)
        self._line_width = line_width

        if use_key_callbacks:
            self._vis = o3d.visualization.VisualizerWithKeyCallback()
        else:
            self._vis = o3d.visualization.Visualizer()

        self._vis.create_window(window_name=title, width=width, height=height)

        # Set render options BEFORE the first add_geometry — this is the order
        # that works reliably with plain Visualizer (matches original scripts).
        opt = self._vis.get_render_option()
        opt.point_size       = self._point_size
        opt.background_color = self._bg_color
        opt.line_width       = self._line_width

        self._geoms: set  = set()   # ids of added geometries
        self._view_set    = False   # has set_view() been called yet?

    # ── geometry management ────────────────────────────────────────────────────

    def add(self, geom) -> None:
        """Add a geometry (idempotent — safe to call twice)."""
        if id(geom) not in self._geoms:
            self._vis.add_geometry(geom)
            self._geoms.add(id(geom))

    def update(self, geom) -> None:
        """Push in-place changes to the renderer.  Auto-adds on first call."""
        if id(geom) not in self._geoms:
            self.add(geom)
        else:
            self._vis.update_geometry(geom)

    def add_or_update(self, geom) -> None:
        """First call → add; subsequent calls → update."""
        self.update(geom)

    def remove(self, geom) -> None:
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
        Return a smoothed copy of *pcd* (input is not modified):
          1. Voxel down-sample
          2. Statistical outlier removal
        Per-call args override instance defaults; pass 0/None to skip a step.
        """
        voxel = voxel_size if voxel_size is not None else self._voxel
        k     = sor_k      if sor_k      is not None else self._sor_k
        std   = sor_std    if sor_std    is not None else self._sor_std

        result = pcd

        if voxel and voxel > 0:
            result = result.voxel_down_sample(voxel_size=voxel)

        if k and k > 0 and len(result.points) >= k + 1:
            result, _ = result.remove_statistical_outlier(
                nb_neighbors=k, std_ratio=std)

        return result

    # ── camera helpers ─────────────────────────────────────────────────────────

    def set_view(
        self,
        front:  tuple = (0, 0, -1),
        up:     tuple = (0, -1,  0),
        zoom:   float = 0.45,
        lookat: Optional[tuple] = None,
    ) -> None:
        """
        Set front / up / zoom (and optionally lookat).
        Call ONCE after the first add() — identical to the original scripts.
        """
        ctr = self._vis.get_view_control()
        ctr.set_front(list(front))
        ctr.set_up(list(up))
        ctr.set_zoom(zoom)
        if lookat is not None:
            ctr.set_lookat(list(lookat))
        self._view_set = True

    def centre_on_pcd(self, pcd: o3d.geometry.PointCloud) -> None:
        """
        Move lookat to the centroid of *pcd*.
        Matches the original: calls set_lookat alone every frame, just like
          ctr.set_lookat(centroid.tolist())
        front/up/zoom are intentionally NOT touched here — that is the
        pattern that works with plain Visualizer.
        """
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return
        self._vis.get_view_control().set_lookat(pts.mean(axis=0).tolist())

    def centre_on_points(self, pts: np.ndarray) -> None:
        """Same as centre_on_pcd but takes a raw (N, 3) array."""
        if len(pts) == 0:
            return
        self._vis.get_view_control().set_lookat(pts.mean(axis=0).tolist())

    # ── key callbacks (only available when use_key_callbacks=True) ─────────────

    def on_key(self, key: str, callback: Callable) -> None:
        """
        Register a key callback.  Requires use_key_callbacks=True.
        callback(vis) must return False (Open3D convention).
        """
        if not isinstance(self._vis, o3d.visualization.VisualizerWithKeyCallback):
            raise RuntimeError(
                "on_key() requires use_key_callbacks=True in the constructor.")
        self._vis.register_key_callback(ord(key.upper()), callback)

    # ── render loop ────────────────────────────────────────────────────────────

    def tick(self) -> bool:
        """
        One render frame: poll events + update renderer.
        Returns False when the window is closed → exit your loop.
        """
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