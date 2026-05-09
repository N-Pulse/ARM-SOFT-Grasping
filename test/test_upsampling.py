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

POISSON_DEPTH     = 6       # octree depth — lower = faster/coarser
UPSAMPLE_N        = 5000    # points sampled from the reconstructed mesh
NORMAL_RADIUS     = 0.02    # m — hybrid normal-search radius
NORMAL_MAX_NN     = 30      # max neighbours for normal estimation
MIN_INPUT_POINTS  = 100     # skip reconstruction below this count


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
    Estimate normals → Poisson surface reconstruction → uniform point sample.
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

    try:
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=POISSON_DEPTH
        )
    except Exception:
        return None

    if len(mesh.vertices) == 0:
        return None

    # Poisson hallucinates geometry outside the real object extent — crop it
    pts      = np.asarray(pcd.points)
    lo, hi   = pts.min(axis=0) - 0.01, pts.max(axis=0) + 0.01
    mesh     = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))

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

    recon      = _ReconThread()
    display    = o3d.geometry.PointCloud()
    geom_added = False

    vis = o3d.visualization.Visualizer()
    vis.create_window("Upsampled Object Reconstruction", width=1280, height=720)
    opt = vis.get_render_option()
    opt.point_size       = 2.0
    opt.background_color = np.array([1.0, 1.0, 1.0])

    try:
        while True:
            frame = isolator.get_full_frame()
            if frame is not None:
                _, iso_pcd, _ = frame
                if iso_pcd is not None:
                    # Show the raw isolated cloud immediately as a fallback
                    display.points = iso_pcd.points
                    display.colors = iso_pcd.colors
                    recon.submit(iso_pcd)

                    if not geom_added:
                        vis.add_geometry(display)
                        ctr = vis.get_view_control()
                        ctr.set_front([0, 0, -1])
                        ctr.set_up([0, -1, 0])
                        ctr.set_zoom(0.45)
                        geom_added = True
                    else:
                        vis.update_geometry(display)

            # Swap to the upsampled cloud once reconstruction is ready
            upsampled = recon.get()
            if upsampled is not None and geom_added:
                display.points = upsampled.points
                display.colors = upsampled.colors
                vis.update_geometry(display)

            if geom_added:
                pts = np.asarray(display.points)
                if len(pts):
                    vis.get_view_control().set_lookat(pts.mean(axis=0).tolist())

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
