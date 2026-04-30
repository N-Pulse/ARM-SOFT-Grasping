import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import threading
import queue

# Configure streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pc = rs.pointcloud()  # realsense utility that computes pointclouds from depth frames
align = rs.align(rs.stream.color)

profile = pipeline.start(config)

# Depth sensor settings — reduce noise
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 4)

decimation    = rs.decimation_filter()
depth_to_disp = rs.disparity_transform(True)
spatial       = rs.spatial_filter()
temporal      = rs.temporal_filter()
disp_to_depth = rs.disparity_transform(False)
hole_filling  = rs.hole_filling_filter()

# Tune
decimation.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)
temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
temporal.set_option(rs.option.filter_smooth_delta, 20)

# Depth range clamp (meters) — D405 is optimized for close range (~7cm–50cm)
MIN_DEPTH = 0.07
MAX_DEPTH = 0.70

# maxsize=1: renderer always gets the latest frame, stale frames are dropped
frame_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()


def capture_loop():
    while not stop_event.is_set():
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        #depth_frame = decimation.process(depth_frame)
        #depth_frame = depth_to_disp.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        #depth_frame = disp_to_depth.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        # Generate point cloud with color texture
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)

        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        texcoords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

        # Remove invalid (zero) points and clamp to depth range
        depth = np.linalg.norm(verts, axis=1)
        mask = (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
        verts = verts[mask]
        texcoords = texcoords[mask]

        # Sample colors from the color frame
        color_image = np.asanyarray(color_frame.get_data())  # BGR, HxW
        h, w = color_image.shape[:2]
        u = np.clip((texcoords[:, 0] * w).astype(int), 0, w - 1)
        v = np.clip((texcoords[:, 1] * h).astype(int), 0, h - 1)
        colors = color_image[v, u, ::-1] / 255.0  # BGR→RGB, normalize

        # Drop stale frame if renderer hasn't consumed it yet, then push latest
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        frame_queue.put((verts, colors))


# Start capture+processing on a background thread
t = threading.Thread(target=capture_loop, daemon=True)
t.start()

# Open3D visualizer — must stay on main thread (GLFW/OpenGL requirement)
vis = o3d.visualization.Visualizer()
vis.create_window("RealSense Live Pointcloud", width=1280, height=720)
pcd = o3d.geometry.PointCloud()
geom_added = False

print("Streaming pointcloud... Close the window or press Ctrl+C to stop.")

try:
    while True:
        try:
            verts, colors = frame_queue.get(timeout=0.1)
            pcd.points = o3d.utility.Vector3dVector(verts)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            if not geom_added:
                vis.add_geometry(pcd)

                # Set a fixed viewpoint once — don't call reset_view_point again
                ctr = vis.get_view_control()
                ctr.set_lookat([0, 0, 0.4])   # look at ~40cm in front of camera
                ctr.set_front([0, 0, -1])      # camera looks in -Z direction
                ctr.set_up([0, -1, 0])         # Y is up (flipped for RealSense convention)
                ctr.set_zoom(0.2)              # tweak to taste

                geom_added = True
            else:
                vis.update_geometry(pcd)
        except queue.Empty:
            pass  # no new frame yet — keep the render loop alive

        if not vis.poll_events():
            break
        vis.update_renderer()

finally:
    stop_event.set()
    pipeline.stop()
    vis.destroy_window()
