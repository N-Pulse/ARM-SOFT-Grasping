"""
armsoft/viewer.py
==================
Optional Open3D live view — point cloud + fitted wireframe + gripper skeleton.

Purely a debugging aid: nothing in the model depends on it, and on a headless
machine you simply omit ``--viz``.
"""

from __future__ import annotations

import numpy as np

from ..core.grasp_geometry import GRIPPER_LINES

GRIPPER_COLOR = [1.0, 0.4, 0.0]


class LiveViewer:
    """Open3D window driven frame-by-frame from ``run_inference``."""

    def __init__(self, title: str = "ARM-SOFT grasp (portable)",
                 width: int = 1280, height: int = 720):
        import open3d as o3d
        self._o3d = o3d
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(title, width, height)
        self.pcd = o3d.geometry.PointCloud()
        self.grip = o3d.geometry.LineSet()
        self.grip.lines = o3d.utility.Vector2iVector(GRIPPER_LINES)
        self.grip.colors = o3d.utility.Vector3dVector([GRIPPER_COLOR] * len(GRIPPER_LINES))
        self._added = {"pcd": False, "grip": False, "shape": False}
        self._shape_ls = None

    def update(self, pipeline, result) -> bool:
        """Apply one frame; returns False when the window has been closed."""
        o3d = self._o3d
        iso = pipeline.last_isolation
        if iso is not None and len(iso.scene_points):
            self.pcd.points = o3d.utility.Vector3dVector(iso.scene_points)
            self.pcd.colors = o3d.utility.Vector3dVector(iso.scene_colors)
            if not self._added["pcd"]:
                self.vis.add_geometry(self.pcd)
                self._added["pcd"] = True
            else:
                self.vis.update_geometry(self.pcd)

        ls = pipeline.last_shape_ls
        if ls is not None:
            if self._shape_ls is None:
                self._shape_ls = ls
                self.vis.add_geometry(self._shape_ls, reset_bounding_box=False)
                self._added["shape"] = True
            else:
                self._shape_ls.points = ls.points
                self._shape_ls.lines = ls.lines
                self._shape_ls.colors = ls.colors
                self.vis.update_geometry(self._shape_ls)

        if result.ok and result.gripper_points:
            self.grip.points = o3d.utility.Vector3dVector(
                np.asarray(result.gripper_points))
            if not self._added["grip"]:
                self.vis.add_geometry(self.grip, reset_bounding_box=False)
                self._added["grip"] = True
            else:
                self.vis.update_geometry(self.grip)

        return self.vis.poll_events() and (self.vis.update_renderer() or True)

    def close(self) -> None:
        self.vis.destroy_window()
