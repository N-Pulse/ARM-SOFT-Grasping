"""
test_obj_iso.py

Captures live RGB-D frames from an Intel D405, runs YOLOv8-seg detection on
the color image, selects the detection whose center is closest to the image
center, and outputs a point cloud masked to that object's segmentation contour
(not its bounding box).

Performance optimisations vs. original:
  - Segmentation model runs at imgsz=320 instead of 640
  - YOLO inference skipped every YOLO_SKIP_FRAMES frames (last mask reused)
  - Point cloud subsampled by SUBSAMPLE factor before masking
  - TensorRT engine auto-used if .engine file exists alongside .pt

Dependencies:
    pip install pyrealsense2 open3d ultralytics numpy opencv-python

Usage:
    python test_obj_iso.py

    # Optional: export TensorRT engine first (big speedup on Jetson)
    # yolo export model=yolo11n-seg.pt format=engine device=0 imgsz=320

Output:
    - Live Open3D window showing the isolated object point cloud
    - get_latest_object_pcd() for programmatic consumption (Stage 2)
"""

import os
import threading
import queue

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────────────────

DEPTH_WIDTH        = 640
DEPTH_HEIGHT       = 480
COLOR_WIDTH        = 640
COLOR_HEIGHT       = 480
FPS                = 30

MIN_DEPTH_M        = 0.07
MAX_DEPTH_M        = 0.70

# Auto-use TensorRT engine if it exists next to the .pt file
_PT_MODEL          = "yolo11n-seg.pt"
_ENGINE_MODEL      = "yolo11n-seg.engine"
YOLO_MODEL         = _ENGINE_MODEL if os.path.exists(_ENGINE_MODEL) else _PT_MODEL

YOLO_CONF          = 0.35
YOLO_IOU           = 0.45
YOLO_IMGSZ         = 320    # half of default — faster, still accurate enough

YOLO_SKIP_FRAMES   = 3      # run YOLO every N frames, reuse mask in between
SUBSAMPLE          = 2      # keep every Nth point cloud point (1 = no skip)

IMG_CENTER         = np.array([COLOR_WIDTH / 2, COLOR_HEIGHT / 2])

# ─── RealSense setup ──────────────────────────────────────────────────────────

def build_pipeline():
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH,  DEPTH_HEIGHT, rs.format.z16,  FPS)
    config.enable_stream(rs.stream.color, COLOR_WIDTH,  COLOR_HEIGHT, rs.format.bgr8, FPS)

    profile      = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 4)   # Short Range

    align    = rs.align(rs.stream.color)
    spatial  = rs.spatial_filter()
    temporal = rs.temporal_filter()
    holes    = rs.hole_filling_filter()
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    return pipeline, align, spatial, temporal, holes, rs.pointcloud()


# ─── YOLO helpers ─────────────────────────────────────────────────────────────

def load_yolo() -> YOLO:
    print(f"[YOLO] loading '{YOLO_MODEL}' ...")
    model = YOLO(YOLO_MODEL)
    if not YOLO_MODEL.endswith(".engine"):
        model.fuse()
    return model


def detect_masks(model, bgr_image: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Run YOLO segmentation on a BGR image.

    Returns a list of (mask, box) pairs where:
      mask : (H, W) bool array — True inside the object silhouette
      box  : [x1, y1, x2, y2] ints — bounding box (used only for centre selection)
    """
    results = model.predict(
        source=bgr_image,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        verbose=False,
    )
    detections = []
    for result in results:
        if result.masks is None:
            continue
        for mask_tensor, box in zip(result.masks.data, result.boxes.xyxy):
            mask_np = mask_tensor.cpu().numpy()
            mask_resized = cv2.resize(
                mask_np,
                (bgr_image.shape[1], bgr_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            detections.append((mask_resized, np.array([x1, y1, x2, y2])))
    return detections


def select_central(detections):
    """Return (mask, box) for the detection closest to the image centre."""
    if not detections:
        return None, None
    best_mask, best_box, best_dist = None, None, float("inf")
    for mask, box in detections:
        x1, y1, x2, y2 = box
        dist = np.linalg.norm(
            np.array([(x1 + x2) / 2, (y1 + y2) / 2]) - IMG_CENTER
        )
        if dist < best_dist:
            best_mask, best_box, best_dist = mask, box, dist
    return best_mask, best_box


# ─── Point cloud masking ──────────────────────────────────────────────────────

def mask_pcd_to_seg(
    verts:       np.ndarray,   # (N, 3) camera-frame 3-D points
    texcoords:   np.ndarray,   # (N, 2) UV in [0, 1]
    colors:      np.ndarray,   # (N, 3) RGB in [0, 1]
    seg_mask:    np.ndarray,   # (H, W) bool — True = object pixel
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only points whose UV coordinate maps to a True pixel in seg_mask.
    This cuts along the object's segmentation contour, not its bbox.
    """
    h, w = seg_mask.shape
    u = np.clip((texcoords[:, 0] * w).astype(int), 0, w - 1)
    v = np.clip((texcoords[:, 1] * h).astype(int), 0, h - 1)
    inside = seg_mask[v, u]
    return verts[inside], colors[inside]


# ─── Shared state ─────────────────────────────────────────────────────────────

_output_queue: queue.Queue = queue.Queue(maxsize=1)
_stop_event   = threading.Event()


def get_latest_object_pcd() -> o3d.geometry.PointCloud | None:
    """Non-blocking — returns latest isolated object pcd or None."""
    try:
        verts, colors = _output_queue.get_nowait()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    except queue.Empty:
        return None


# ─── Capture + inference loop ─────────────────────────────────────────────────

def capture_loop():
    pipeline, align, spatial, temporal, holes, pc_util = build_pipeline()
    model = load_yolo()

    frame_idx     = 0
    last_mask     = None   # cached segmentation mask from last YOLO run
    last_box      = None

    try:
        while not _stop_event.is_set():

            # ── 1. Grab aligned RGB-D frame ───────────────────────────────
            frames   = pipeline.wait_for_frames()
            aligned  = align.process(frames)
            depth_fr = aligned.get_depth_frame()
            color_fr = aligned.get_color_frame()
            if not depth_fr or not color_fr:
                continue

            depth_fr = spatial.process(depth_fr)
            depth_fr = temporal.process(depth_fr)
            depth_fr = holes.process(depth_fr)

            bgr = np.asanyarray(color_fr.get_data())

            # ── 2. YOLO segmentation (every YOLO_SKIP_FRAMES frames) ──────
            frame_idx += 1
            if frame_idx % YOLO_SKIP_FRAMES == 0:
                detections = detect_masks(model, bgr)
                last_mask, last_box = select_central(detections)

            if last_mask is None:
                continue   # no detection yet — wait for first YOLO hit

            # ── 3. Build point cloud (subsampled) ─────────────────────────
            pc_util.map_to(color_fr)
            points_rs = pc_util.calculate(depth_fr)

            verts     = np.asanyarray(points_rs.get_vertices()) \
                          .view(np.float32).reshape(-1, 3)[::SUBSAMPLE]
            texcoords = np.asanyarray(points_rs.get_texture_coordinates()) \
                          .view(np.float32).reshape(-1, 2)[::SUBSAMPLE]

            # Depth range filter
            depth_vals = np.linalg.norm(verts, axis=1)
            valid      = (depth_vals > MIN_DEPTH_M) & (depth_vals < MAX_DEPTH_M)
            verts      = verts[valid]
            texcoords  = texcoords[valid]

            # Sample colors
            h, w = bgr.shape[:2]
            u = np.clip((texcoords[:, 0] * w).astype(int), 0, w - 1)
            v = np.clip((texcoords[:, 1] * h).astype(int), 0, h - 1)
            colors = bgr[v, u, ::-1] / 255.0   # BGR → RGB, [0, 1]

            # ── 4. Mask along segmentation contour ────────────────────────
            obj_verts, obj_colors = mask_pcd_to_seg(
                verts, texcoords, colors, last_mask
            )

            if obj_verts.shape[0] < 50:
                continue

            # ── 5. Push (drop stale frame) ────────────────────────────────
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

    print("Running Stage 1 — YOLO segmentation + point cloud isolation.")
    print(f"  model       : {YOLO_MODEL}")
    print(f"  imgsz       : {YOLO_IMGSZ}")
    print(f"  skip frames : {YOLO_SKIP_FRAMES}")
    print(f"  subsample   : 1 in every {SUBSAMPLE} points")
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