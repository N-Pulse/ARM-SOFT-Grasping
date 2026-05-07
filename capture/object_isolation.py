"""
object_isolation.py

Importable ObjectIsolator class — captures live RGB-D frames from an Intel
D405, runs YOLO segmentation, and exposes the isolated object point cloud
(masked along the segmentation contour, not the bounding box).

Performance optimisations:
  - YOLO runs at imgsz=320
  - Inference skipped every YOLO_SKIP_FRAMES frames (last mask reused)
  - Point cloud subsampled by SUBSAMPLE factor
  - TensorRT engine auto-used if .engine file exists alongside .pt

Usage:
    from object_isolation import ObjectIsolator

    isolator = ObjectIsolator()
    isolator.start()
    pcd = isolator.get_pcd()   # o3d.PointCloud or None
    isolator.stop()

    # or as a context manager:
    with ObjectIsolator() as isolator:
        pcd = isolator.get_pcd()
"""

import os
import queue
import threading

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────────────────

DEPTH_WIDTH      = 640
DEPTH_HEIGHT     = 480
COLOR_WIDTH      = 640
COLOR_HEIGHT     = 480
FPS              = 30

MIN_DEPTH_M      = 0.07
MAX_DEPTH_M      = 0.70

# Resolve model paths relative to this file so YOLO doesn't fall back to
# downloading the weights when the script is launched from a different cwd.
_THIS_DIR        = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT    = os.path.dirname(_THIS_DIR)


def _find_model(filename: str) -> str:
    """Search for `filename` next to this script, in the project root, and cwd."""
    for base in (_THIS_DIR, _PROJECT_ROOT, os.getcwd()):
        candidate = os.path.join(base, filename)
        if os.path.exists(candidate):
            return candidate
    # Fall back to the bare filename (lets ultralytics download/resolve it).
    return filename


_PT_MODEL        = _find_model("yolo11n-seg.pt")
_ENGINE_MODEL    = _find_model("yolo11n-seg.engine")
YOLO_MODEL       = _ENGINE_MODEL if os.path.exists(_ENGINE_MODEL) else _PT_MODEL

YOLO_CONF        = 0.35
YOLO_IOU         = 0.45
YOLO_IMGSZ       = 320

YOLO_SKIP_FRAMES = 3    # run YOLO every N frames, reuse mask in between
SUBSAMPLE        = 2    # keep every Nth point cloud point

IMG_CENTER       = np.array([COLOR_WIDTH / 2, COLOR_HEIGHT / 2])

# ─── Internal helpers ─────────────────────────────────────────────────────────

def _build_pipeline():
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH,  DEPTH_HEIGHT, rs.format.z16,  FPS)
    config.enable_stream(rs.stream.color, COLOR_WIDTH,  COLOR_HEIGHT, rs.format.bgr8, FPS)

    profile      = pipeline.start(config)
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


def _select_central(detections):
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


def _mask_pcd_to_seg(verts, texcoords, colors, seg_mask):
    """Keep only points whose UV falls inside the segmentation mask."""
    h, w = seg_mask.shape
    u = np.clip((texcoords[:, 0] * w).astype(int), 0, w - 1)
    v = np.clip((texcoords[:, 1] * h).astype(int), 0, h - 1)
    inside = seg_mask[v, u]
    return verts[inside], colors[inside]


# ─── Public class ─────────────────────────────────────────────────────────────

class ObjectIsolator:
    """
    Runs D405 capture + YOLO segmentation on a background thread.
    Point cloud is masked along the object's segmentation contour.

    Parameters
    ----------
    min_points : int
        Minimum number of points for a frame to be considered valid.
    """

    def __init__(self, min_points: int = 50):
        self.ready = threading.Event()   # set after YOLO warm-up
        self._min_points  = min_points
        self._frame_queue = queue.Queue(maxsize=1)
        self._stop_event  = threading.Event()
        self._thread      = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_full_frame(self):
        """
        Returns (full_pcd, iso_pcd, preview_bgr) or None. Non-blocking.

        full_pcd    : entire scene; target in real colour, background greyed out.
        iso_pcd     : isolated object only, or None if not found.
        preview_bgr : annotated BGR image with YOLO mask overlay.
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
        Returns the latest isolated object point cloud, or None if not ready.
        Non-blocking — call this in your own loop.
        """
        result = self.get_full_frame()
        if result is None:
            return None
        _, iso_pcd, _ = result
        return iso_pcd

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self):
        try:
            pipeline, align, spatial, temporal, holes, pc_util = _build_pipeline()
        except Exception as exc:
            print(f"[ObjectIsolator] FATAL: could not start RealSense pipeline: {exc}")
            return

        try:
            print(f"[ObjectIsolator] loading '{YOLO_MODEL}' ...")
            model = YOLO(YOLO_MODEL)
            if not YOLO_MODEL.endswith(".engine"):
                model.fuse()
            print("[ObjectIsolator] YOLO loaded ✓ — warming up...")

            model.predict(source=np.zeros((320, 320, 3), dtype=np.uint8),
                          imgsz=YOLO_IMGSZ, verbose=False)
            print("[ObjectIsolator] warm-up done ✓ — streaming frames...")
            self.ready.set()
        except Exception as exc:
            print(f"[ObjectIsolator] FATAL: could not load YOLO model: {exc}")
            pipeline.stop()
            return

        import time as _time
        print("[ObjectIsolator] camera + YOLO ready, streaming frames...")

        frame_idx  = 0
        last_mask  = None
        last_box   = None
        _last_log  = 0.0

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

                bgr = np.asanyarray(color_fr.get_data())

                # YOLO every YOLO_SKIP_FRAMES frames
                frame_idx += 1
                if frame_idx % YOLO_SKIP_FRAMES == 0:
                    detections = _detect_masks(model, bgr)
                    last_mask, last_box = _select_central(detections)

                # Build subsampled point cloud
                pc_util.map_to(color_fr)
                points_rs = pc_util.calculate(depth_fr)
                verts     = np.asanyarray(points_rs.get_vertices()) \
                              .view(np.float32).reshape(-1, 3)[::SUBSAMPLE]
                texcoords = np.asanyarray(points_rs.get_texture_coordinates()) \
                              .view(np.float32).reshape(-1, 2)[::SUBSAMPLE]

                depth_vals = np.linalg.norm(verts, axis=1)
                valid      = (depth_vals > MIN_DEPTH_M) & (depth_vals < MAX_DEPTH_M)
                verts      = verts[valid]
                texcoords  = texcoords[valid]

                h, w = bgr.shape[:2]
                u = np.clip((texcoords[:, 0] * w).astype(int), 0, w - 1)
                v = np.clip((texcoords[:, 1] * h).astype(int), 0, h - 1)
                colors = bgr[v, u, ::-1] / 255.0

                # Full scene: target in colour, background grey
                if last_mask is not None:
                    inside      = last_mask[v, u]
                    full_colors = np.full_like(colors, 0.35)
                    full_colors[inside] = colors[inside]
                    obj_verts_raw  = verts[inside]
                    obj_colors_raw = colors[inside]
                else:
                    full_colors    = colors
                    obj_verts_raw  = np.zeros((0, 3), np.float32)
                    obj_colors_raw = np.zeros((0, 3), np.float32)

                # cv2 preview
                preview_bgr = bgr.copy()
                if last_mask is not None and last_box is not None:
                    overlay = np.zeros_like(preview_bgr)
                    overlay[last_mask] = (0, 255, 0)
                    preview_bgr = cv2.addWeighted(preview_bgr, 0.7, overlay, 0.3, 0)
                    x1, y1, x2, y2 = last_box
                    cv2.rectangle(preview_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(preview_bgr, "target", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if obj_verts_raw.shape[0] >= self._min_points:
                    obj_verts, obj_colors = obj_verts_raw, obj_colors_raw
                else:
                    obj_verts  = np.zeros((0, 3), np.float32)
                    obj_colors = np.zeros((0, 3), np.float32)

                now = _time.monotonic()
                if now - _last_log >= 1.0:
                    print(f"[ObjectIsolator] scene pts: {len(verts)}  "
                          f"obj pts: {len(obj_verts)}  "
                          f"target: {'yes' if last_mask is not None else 'no'}")
                    _last_log = now

                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self._frame_queue.put(
                    (verts, full_colors, obj_verts, obj_colors, preview_bgr)
                )

        except Exception as exc:
            import traceback
            print(f"[ObjectIsolator] FATAL: {exc}")
            traceback.print_exc()
        finally:
            pipeline.stop()