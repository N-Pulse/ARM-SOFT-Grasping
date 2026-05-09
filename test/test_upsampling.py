"""
test_upsampling.py

Live Open3D reconstruction of an isolated object point cloud.
ObjectIsolator (YOLO + D405) locks on one object; a background thread
runs Poisson surface reconstruction and samples a denser point cloud
which is displayed in an Open3D viewer.

Controls:
    Close the Open3D window  |  Ctrl+C
"""

import sys
import os
import queue
import threading

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "capture"))

import numpy as np
import open3d as o3d
from object_isolation import ObjectIsolator


# ── Reconstruction settings ───────────────────────────────────────────────────

UPSAMPLE_N        = 5000    # points sampled from the reconstructed mesh
NORMAL_RADIUS     = 0.02    # m — hybrid normal-search radius
NORMAL_MAX_NN     = 30      # max neighbours for normal estimation
MIN_INPUT_POINTS  = 50      # skip reconstruction below this count


# ── Background reconstruction thread ─────────────────────────────────────────

class _ReconThread:
    """
    Accepts raw isolated point clouds via submit() and publishes the latest
    Poisson-reconstructed + uniformly sampled cloud via get().
    Keeps only the most recent pending input so stale frames are dropped.
    """

    def __init__(self):
        self._in  = queue.Queue(maxsize=1)
        self._out = queue.Queue(maxsize=1)
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, pcd: o3d.geometry.PointCloud) -> None:
        try:
            self._in.get_nowait()
        except queue.Empty:
            pass
        self._in.put(pcd)

    def get(self) -> "o3d.geometry.PointCloud | None":
        try:
            return self._out.get_nowait()
        except queue.Empty:
            return None

    def _loop(self) -> None:
        while True:
            pcd    = self._in.get()
            result = _reconstruct(pcd)
            if result is not None:
                try:
                    self._out.get_nowait()
                except queue.Empty:
                    pass
                self._out.put(result)


def _reconstruct(pcd: o3d.geometry.PointCloud) -> "o3d.geometry.PointCloud | None":
    """
    Estimate normals → Ball Pivoting Algorithm → uniform point sample.
    BPA works on partial/open surfaces (one-sided depth camera views),
    unlike Poisson which requires a watertight surface.
    Returns a denser PointCloud coloured light-blue, or None on failure.
    """
    if len(pcd.points) < MIN_INPUT_POINTS:
        return None

    pcd = o3d.geometry.PointCloud(pcd)   # work on a copy
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN
        )
    )
    pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

    # Set ball radii relative to average point spacing
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist  = np.mean(distances)
    radii     = o3d.utility.DoubleVector([avg_dist, avg_dist * 2, avg_dist * 4])

    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, radii
        )
    except Exception:
        return None

    if len(mesh.vertices) == 0:
        return None

    sampled = mesh.sample_points_uniformly(number_of_points=UPSAMPLE_N)
    sampled.paint_uniform_color([0.3, 0.7, 1.0])
    return sampled


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    isolator = ObjectIsolator(min_points=MIN_INPUT_POINTS)
    isolator.start()

    print("Waiting for YOLO to load...")
    isolator.ready.wait()
    print("YOLO ready — opening viewer.\n")
    print("Running — close the Open3D window or Ctrl+C to stop.\n")

    recon       = _ReconThread()
    display     = o3d.geometry.PointCloud()
    geom_added  = False
    frame_count = 0

    vis = o3d.visualization.Visualizer()
    vis.create_window("Upsampled Object Reconstruction", width=1280, height=720)
    opt = vis.get_render_option()
    opt.point_size       = 2.0
    opt.background_color = np.array([1.0, 1.0, 1.0])

    try:
        while True:
            try:
                full_verts, full_colors, obj_verts, obj_colors, _ = \
                    isolator._frame_queue.get(timeout=0.05)
            except queue.Empty:
                if not vis.poll_events():
                    break
                vis.update_renderer()
                continue

            frame_count += 1
            print(f"[upsampling] frames: {frame_count}  obj pts: {len(obj_verts)}")

            pts  = obj_verts  if len(obj_verts)  > 0 else full_verts
            cols = obj_colors if len(obj_colors) > 0 else full_colors

            raw_pcd        = o3d.geometry.PointCloud()
            raw_pcd.points = o3d.utility.Vector3dVector(pts)
            raw_pcd.colors = o3d.utility.Vector3dVector(cols)

            display.points = raw_pcd.points
            display.colors = raw_pcd.colors

            if len(obj_verts) > 0:
                recon.submit(raw_pcd)

            if not geom_added:
                vis.add_geometry(display)
                ctr = vis.get_view_control()
                ctr.set_front([0, 0, -1])
                ctr.set_up([0, -1, 0])
                ctr.set_zoom(0.45)
                geom_added = True
            else:
                vis.update_geometry(display)

            upsampled = recon.get()
            if upsampled is not None:
                display.points = upsampled.points
                display.colors = upsampled.colors
                vis.update_geometry(display)

            pts_arr = np.asarray(display.points)
            if len(pts_arr):
                vis.get_view_control().set_lookat(pts_arr.mean(axis=0).tolist())

            if not vis.poll_events():
                break
            vis.update_renderer()

    except KeyboardInterrupt:
        pass
    finally:
        isolator.stop()
        vis.destroy_window()


if __name__ == "__main__":
    run()
