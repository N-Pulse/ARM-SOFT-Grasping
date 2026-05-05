import queue
import threading

import numpy as np
import open3d as o3d
import pyrealsense2 as rs


MIN_DEPTH = 0.07
MAX_DEPTH = 0.70


class PointcloudViewer:
    """
    Open3D live pointcloud window.  Must be created and ticked on the main thread
    (GLFW/OpenGL requirement).  Accepts key callbacks via register_key().
    """

    def __init__(self, title="RealSense Live Pointcloud", width=1280, height=720):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(title, width=width, height=height)
        self._pcd        = o3d.geometry.PointCloud()
        self._geom_added = False

    def register_key(self, key: int, callback):
        """Register an Open3D key callback (callback signature: fn(vis) -> bool)."""
        self.vis.register_key_callback(key, callback)

    def update(self, pcd: o3d.geometry.PointCloud):
        """Push a new pointcloud to the viewer."""
        if len(pcd.points) == 0:
            return
        self._pcd.points = pcd.points
        self._pcd.colors = pcd.colors
        if not self._geom_added:
            self.vis.add_geometry(self._pcd)
            ctr = self.vis.get_view_control()
            ctr.set_lookat([0, 0, 0.4])
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, -1, 0])
            ctr.set_zoom(0.2)
            self._geom_added = True
            print("[viewer] first frame — pointcloud visible")
        else:
            self.vis.update_geometry(self._pcd)

    def tick(self) -> bool:
        """Process events and redraw.  Returns False when the window is closed."""
        if not self.vis.poll_events():
            return False
        self.vis.update_renderer()
        return True

    def destroy(self):
        self.vis.destroy_window()


# ── Standalone entry point ────────────────────────────────────────────────────

def _capture_loop(frame_queue: queue.Queue, stop_event: threading.Event):
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile      = pipeline.start(cfg)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 4)

    align    = rs.align(rs.stream.color)
    spatial  = rs.spatial_filter()
    temporal = rs.temporal_filter()
    holes    = rs.hole_filling_filter()
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)
    pc_util = rs.pointcloud()

    try:
        while not stop_event.is_set():
            frames    = pipeline.wait_for_frames()
            aligned   = align.process(frames)
            depth_fr  = aligned.get_depth_frame()
            color_fr  = aligned.get_color_frame()
            if not depth_fr or not color_fr:
                continue

            depth_fr = spatial.process(depth_fr)
            depth_fr = temporal.process(depth_fr)
            depth_fr = holes.process(depth_fr)

            pc_util.map_to(color_fr)
            points    = pc_util.calculate(depth_fr)
            verts     = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            texcoords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

            depth  = np.linalg.norm(verts, axis=1)
            valid  = (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
            verts     = verts[valid]
            texcoords = texcoords[valid]

            bgr  = np.asanyarray(color_fr.get_data())
            h, w = bgr.shape[:2]
            u    = np.clip((texcoords[:, 0] * w).astype(int), 0, w - 1)
            v    = np.clip((texcoords[:, 1] * h).astype(int), 0, h - 1)
            colors = bgr[v, u, ::-1] / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(verts)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            frame_queue.put(pcd)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    frame_queue = queue.Queue(maxsize=1)
    stop_event  = threading.Event()

    t = threading.Thread(target=_capture_loop, args=(frame_queue, stop_event), daemon=True)
    t.start()
    print("Streaming pointcloud... close the window or press Ctrl+C to stop.")

    viewer = PointcloudViewer()
    try:
        while True:
            try:
                pcd = frame_queue.get(timeout=0.1)
                viewer.update(pcd)
            except queue.Empty:
                pass
            if not viewer.tick():
                break
    finally:
        stop_event.set()
        viewer.destroy()
