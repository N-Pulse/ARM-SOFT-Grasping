"""
stage1.py

Importable module for Stage 1 — object isolation from a live D405 feed.

Usage from another file:
    from stage1 import ObjectIsolator

    isolator = ObjectIsolator()
    isolator.start()

    pcd = isolator.get_pcd()   # returns o3d.PointCloud or None
    isolator.stop()

Or as a context manager:
    with ObjectIsolator() as isolator:
        pcd = isolator.get_pcd()
"""

import queue
import threading

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────────────────

DEPTH_WIDTH  = 640
DEPTH_HEIGHT = 480
COLOR_WIDTH  = 640
COLOR_HEIGHT = 480
FPS          = 30

MIN_DEPTH_M  = 0.07
MAX_DEPTH_M  = 0.70

YOLO_MODEL   = "yolo11n-seg.pt"
YOLO_CONF    = 0.35
YOLO_IOU     = 0.45

IMG_CENTER   = np.array([COLOR_WIDTH / 2, COLOR_HEIGHT / 2])

# ─── Internal helpers (not exported) ─────────────────────────────────────────

def _build_pipeline():
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH,  DEPTH_HEIGHT, rs.format.z16,  FPS)
    config.enable_stream(rs.stream.color, COLOR_WIDTH,  COLOR_HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)
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

    return pipeline, align, spatial, temporal, holes, rs.pointcloud()


def _detect_masks(model, bgr_image):
    results = model.predict(source=bgr_image, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
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


def _select_central(detections):
    if not detections:
        return None, None
    best_mask, best_box, best_dist = None, None, float("inf")
    for mask, box in detections:
        x1, y1, x2, y2 = box
        dist = np.linalg.norm(np.array([(x1 + x2) / 2, (y1 + y2) / 2]) - IMG_CENTER)
        if dist < best_dist:
            best_mask, best_box, best_dist = mask, box, dist
    return best_mask, best_box


# ─── Public class ─────────────────────────────────────────────────────────────

class ObjectIsolator:
    """
    Runs D405 capture + YOLO segmentation on a background thread.
    Exposes a single method, get_pcd(), for the rest of the pipeline.

    Parameters
    ----------
    min_points : int
        Minimum number of points required before a frame is considered valid.
        Frames below this threshold are dropped silently.
    """

    def __init__(self, min_points: int = 50):
        self._min_points  = min_points
        self._frame_queue = queue.Queue(maxsize=1)
        self._stop_event  = threading.Event()
        self._thread      = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Start the background capture + inference thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # already running
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the background thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_full_frame(self):
        """
        Return (full_scene_pcd, isolated_pcd_or_None, preview_bgr) or None.

        full_scene_pcd  : entire depth-filtered scene; target is in real colour,
                          background is greyed out (0.35 grey) when a target is found.
        isolated_pcd    : only the target object, or None if not found / too few points.
        preview_bgr     : annotated BGR image (YOLO mask + bounding box overlay).

        Non-blocking.
        """
        try:
            full_verts, full_colors, obj_verts, obj_colors, preview_bgr = \
                self._frame_queue.get_nowait()
        except queue.Empty:
            return None

        full_pcd = o3d.geometry.PointCloud()
        full_pcd.points = o3d.utility.Vector3dVector(full_verts)
        full_pcd.colors = o3d.utility.Vector3dVector(full_colors)

        iso_pcd = None
        if len(obj_verts) > 0:
            iso_pcd = o3d.geometry.PointCloud()
            iso_pcd.points = o3d.utility.Vector3dVector(obj_verts)
            iso_pcd.colors = o3d.utility.Vector3dVector(obj_colors)

        return full_pcd, iso_pcd, preview_bgr

    def get_pcd(self) -> o3d.geometry.PointCloud | None:
        """
        Return the latest isolated object point cloud, or None if not ready.

        The point cloud is in camera frame coordinates.
        Points are guaranteed to:
          - fall within the 7–70 cm depth range
          - belong to the single object whose bbox center is closest to the
            image center (320, 240)

        Non-blocking — call this in your own loop.
        """
        result = self.get_full_frame()
        if result is None:
            return None
        _, iso_pcd, _ = result
        return iso_pcd

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self):
        pipeline, align, spatial, temporal, holes, pc_util = _build_pipeline()
        model = YOLO(YOLO_MODEL)
        model.fuse()

        try:
            while not self._stop_event.is_set():
                frames   = pipeline.wait_for_frames()
                aligned  = align.process(frames)
                depth_fr = aligned.get_depth_frame()
                color_fr = aligned.get_color_frame()
                if not depth_fr or not color_fr:
                    continue

                depth_fr = spatial.process(depth_fr)
                depth_fr = temporal.process(depth_fr)
                depth_fr = holes.process(depth_fr)

                bgr        = np.asanyarray(color_fr.get_data())
                detections = _detect_masks(model, bgr)
                target_mask, target_box = _select_central(detections)

                # Full-scene pointcloud (always built so the live view never stalls)
                pc_util.map_to(color_fr)
                points_rs = pc_util.calculate(depth_fr)
                verts     = np.asanyarray(points_rs.get_vertices()).view(np.float32).reshape(-1, 3)
                texcoords = np.asanyarray(points_rs.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

                depth_vals = np.linalg.norm(verts, axis=1)
                valid      = (depth_vals > MIN_DEPTH_M) & (depth_vals < MAX_DEPTH_M)
                verts      = verts[valid]
                texcoords  = texcoords[valid]

                h, w = bgr.shape[:2]
                u = np.clip((texcoords[:, 0] * w).astype(int), 0, w - 1)
                v = np.clip((texcoords[:, 1] * h).astype(int), 0, h - 1)
                colors = bgr[v, u, ::-1] / 255.0

                # Display colors: target in real colour, rest greyed out
                if target_mask is not None:
                    inside      = target_mask[v, u]
                    full_colors = np.full_like(colors, 0.35)
                    full_colors[inside] = colors[inside]
                    obj_verts_raw  = verts[inside]
                    obj_colors_raw = colors[inside]
                else:
                    full_colors    = colors
                    obj_verts_raw  = np.zeros((0, 3), np.float32)
                    obj_colors_raw = np.zeros((0, 3), np.float32)

                # cv2 preview with YOLO overlay
                preview_bgr = bgr.copy()
                if target_mask is not None and target_box is not None:
                    overlay = np.zeros_like(preview_bgr)
                    overlay[target_mask] = (0, 255, 0)
                    preview_bgr = cv2.addWeighted(preview_bgr, 0.7, overlay, 0.3, 0)
                    x1, y1, x2, y2 = target_box
                    cv2.rectangle(preview_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(preview_bgr, "target", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Only expose isolated object when it has enough points for inference
                if obj_verts_raw.shape[0] >= self._min_points:
                    obj_verts, obj_colors = obj_verts_raw, obj_colors_raw
                else:
                    obj_verts  = np.zeros((0, 3), np.float32)
                    obj_colors = np.zeros((0, 3), np.float32)

                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self._frame_queue.put((verts, full_colors, obj_verts, obj_colors, preview_bgr))

        finally:
            pipeline.stop()