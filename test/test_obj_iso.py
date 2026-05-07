"""
stage1_object_isolation.py

Captures live RGB-D frames from an Intel D405, runs YOLOv8 detection on the
color image (bounding boxes only — class labels ignored), selects the detection
whose center is closest to the image center, and outputs a point cloud masked
to that object alone.

Dependencies:
    pip install pyrealsense2 open3d ultralytics numpy opencv-python

Usage:
    python stage1_object_isolation.py

Output:
    - Live Open3D window showing the isolated object point cloud
    - get_latest_object_pcd() can be called from another module to
      consume the result programmatically (e.g. feed into Stage 2)
"""

import threading
import queue

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────────────────

DEPTH_WIDTH   = 640
DEPTH_HEIGHT  = 480
COLOR_WIDTH   = 640
COLOR_HEIGHT  = 480
FPS           = 30

MIN_DEPTH_M   = 0.07   # D405 reliable close-range floor
MAX_DEPTH_M   = 0.70   # beyond this depth quality drops

YOLO_MODEL    = "yolo11n.pt"   # nano — fast, good enough for bbox; auto-downloads
YOLO_CONF     = 0.35           # detection confidence threshold
YOLO_IOU      = 0.45           # NMS IoU threshold

IMG_CENTER    = np.array([COLOR_WIDTH / 2, COLOR_HEIGHT / 2])  # (320, 240)

# ─── RealSense setup ──────────────────────────────────────────────────────────

def build_pipeline():
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16,  FPS)
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)

    # D405 visual preset 4 = "Short Range" — reduces noise at close distances
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 4)

    align = rs.align(rs.stream.color)

    # Post-processing filters (same tuning as reference code)
    spatial  = rs.spatial_filter()
    temporal = rs.temporal_filter()
    holes    = rs.hole_filling_filter()
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    pc = rs.pointcloud()

    return pipeline, align, spatial, temporal, holes, pc


# ─── YOLO helpers ─────────────────────────────────────────────────────────────

def load_yolo():
    """Load YOLO model once; auto-downloads weights on first run."""
    model = YOLO(YOLO_MODEL)
    model.fuse()   # fuse Conv+BN layers for faster inference
    return model


def detect_boxes(model, bgr_image: np.ndarray) -> list[np.ndarray]:
    """
    Run YOLO on a BGR image.

    Returns a list of bounding boxes as [x1, y1, x2, y2] in pixel coords.
    Class labels are intentionally discarded — we only care about geometry.
    """
    results = model.predict(
        source=bgr_image,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        verbose=False,
    )
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            boxes.append(np.array([x1, y1, x2, y2]))
    return boxes


def select_central_box(boxes: list[np.ndarray]) -> np.ndarray | None:
    """
    From all detected boxes, return the one whose center pixel is
    closest to the image center (320, 240).

    This is the primary-object heuristic: the user is pointing the camera
    at what they want to grasp, so the centered object is the target.
    """
    if not boxes:
        return None

    best_box  = None
    best_dist = float("inf")

    for box in boxes:
        x1, y1, x2, y2 = box
        box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        dist = np.linalg.norm(box_center - IMG_CENTER)
        if dist < best_dist:
            best_box, best_dist = box, dist

    return best_box


# ─── Point cloud masking ──────────────────────────────────────────────────────

def mask_pcd_to_box(
    verts:     np.ndarray,   # (N, 3) — 3-D points in camera frame
    texcoords: np.ndarray,   # (N, 2) — UV in [0,1] mapping each point to a pixel
    colors:    np.ndarray,   # (N, 3) — RGB in [0,1]
    box:       np.ndarray,   # [x1, y1, x2, y2] in pixels
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only the 3-D points whose texture coordinate falls inside `box`.

    texcoords are in [0,1] UV space; we convert to pixel space first, then
    apply the bounding-box mask. This is purely a 2D index operation —
    no 3D geometry needed to isolate the object.
    """
    x1, y1, x2, y2 = box

    # UV → pixel
    u = (texcoords[:, 0] * COLOR_WIDTH).astype(int)
    v = (texcoords[:, 1] * COLOR_HEIGHT).astype(int)

    inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)

    return verts[inside], colors[inside]


# ─── Shared state (producer → consumer) ──────────────────────────────────────

_output_queue: queue.Queue = queue.Queue(maxsize=1)   # latest isolated pcd
_stop_event   = threading.Event()


def get_latest_object_pcd() -> o3d.geometry.PointCloud | None:
    """
    Non-blocking call for external modules (e.g. Stage 2).
    Returns the most recently isolated object point cloud, or None if not ready.
    """
    try:
        verts, colors = _output_queue.get_nowait()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    except queue.Empty:
        return None


# ─── Capture + inference loop (background thread) ─────────────────────────────

def capture_loop():
    pipeline, align, spatial, temporal, holes, pc_util = build_pipeline()
    model = load_yolo()

    try:
        while not _stop_event.is_set():

            # ── 1. Grab aligned RGB-D frame ───────────────────────────────
            frames   = pipeline.wait_for_frames()
            aligned  = align.process(frames)
            depth_fr = aligned.get_depth_frame()
            color_fr = aligned.get_color_frame()
            if not depth_fr or not color_fr:
                continue

            # Post-processing filters
            depth_fr = spatial.process(depth_fr)
            depth_fr = temporal.process(depth_fr)
            depth_fr = holes.process(depth_fr)

            # ── 2. Build full scene point cloud ───────────────────────────
            pc_util.map_to(color_fr)
            points_rs = pc_util.calculate(depth_fr)

            verts     = np.asanyarray(points_rs.get_vertices())\
                          .view(np.float32).reshape(-1, 3)
            texcoords = np.asanyarray(points_rs.get_texture_coordinates())\
                          .view(np.float32).reshape(-1, 2)

            # Depth range filter + remove invalid zero-points
            depth_vals = np.linalg.norm(verts, axis=1)
            valid = (depth_vals > MIN_DEPTH_M) & (depth_vals < MAX_DEPTH_M)
            verts     = verts[valid]
            texcoords = texcoords[valid]

            # Sample colors
            bgr   = np.asanyarray(color_fr.get_data())    # (H, W, 3) BGR
            h, w  = bgr.shape[:2]
            u = np.clip((texcoords[:, 0] * w).astype(int), 0, w - 1)
            v = np.clip((texcoords[:, 1] * h).astype(int), 0, h - 1)
            colors = bgr[v, u, ::-1] / 255.0              # BGR→RGB, [0,1]

            # ── 3. YOLO detection on RGB frame ────────────────────────────
            boxes      = detect_boxes(model, bgr)
            target_box = select_central_box(boxes)

            if target_box is None:
                # No detection — skip this frame, keep visualiser alive
                continue

            # ── 4. Mask point cloud to selected box ───────────────────────
            obj_verts, obj_colors = mask_pcd_to_box(
                verts, texcoords, colors, target_box
            )

            if obj_verts.shape[0] < 50:
                # Too few points — detection probably clipped an edge, skip
                continue

            # ── 5. Push to output queue (drop stale frame if not consumed) ─
            try:
                _output_queue.get_nowait()
            except queue.Empty:
                pass
            _output_queue.put((obj_verts, obj_colors))

    finally:
        pipeline.stop()


# ─── Visualiser (main thread) ─────────────────────────────────────────────────

def run():
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    vis = o3d.visualization.Visualizer()
    vis.create_window("Stage 1 — Isolated Object Point Cloud", width=1280, height=720)
    pcd        = o3d.geometry.PointCloud()
    geom_added = False

    print("Running Stage 1 — YOLO object isolation.")
    print("Close the window or press Ctrl+C to stop.\n")

    try:
        while True:
            try:
                verts, colors = _output_queue.get(timeout=0.1)
                pcd.points = o3d.utility.Vector3dVector(verts)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                if not geom_added:
                    vis.add_geometry(pcd)
                    ctr = vis.get_view_control()
                    ctr.set_lookat([0, 0, 0.3])
                    ctr.set_front([0, 0, -1])
                    ctr.set_up([0, -1, 0])
                    ctr.set_zoom(0.25)
                    geom_added = True
                else:
                    vis.update_geometry(pcd)

            except queue.Empty:
                pass

            if not vis.poll_events():
                break
            vis.update_renderer()

    except KeyboardInterrupt:
        pass
    finally:
        _stop_event.set()
        vis.destroy_window()


if __name__ == "__main__":
    run()