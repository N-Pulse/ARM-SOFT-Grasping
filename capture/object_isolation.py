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

_THIS_DIR        = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT    = os.path.dirname(_THIS_DIR)


def _find_model(filename: str) -> str:
    for base in (_THIS_DIR, _PROJECT_ROOT, os.getcwd()):
        candidate = os.path.join(base, filename)
        if os.path.exists(candidate):
            return candidate
    return filename


_PT_MODEL        = _find_model("yolo11n-seg.pt")
_ENGINE_MODEL    = _find_model("yolo11n-seg.engine")
YOLO_MODEL       = _ENGINE_MODEL if os.path.exists(_ENGINE_MODEL) else _PT_MODEL

YOLO_CONF        = 0.35
YOLO_IOU         = 0.45
YOLO_IMGSZ       = 320

YOLO_SKIP_FRAMES = 3
SUBSAMPLE        = 2

# Detections whose segmentation mask covers more than this fraction of the
# image are discarded — they are almost certainly background surfaces such as
# a table or floor rather than a graspable object.
MAX_MASK_FILL    = 0.80

# ─── Pre-YOLO foreground filtering ───────────────────────────────────────────
# Step 1 — depth-discontinuity removal
# Pixels whose neighbourhood contains a depth jump larger than this value
# (in raw D405 depth units; default scale = 1 mm/unit, so 30 ≈ 3 cm) are
# treated as surface-edge noise and blanked out before YOLO sees the image.
DEPTH_GAP_UNITS     = 30
DEPTH_KERNEL_SIZE   = 5     # square neighbourhood used for min/max depth check

# Step 2 — white-background removal
# Pixels where max(B,G,R) > WHITE_BRIGHTNESS_MIN  AND
#           max(B,G,R) - min(B,G,R) < WHITE_SAT_MAX
# are considered "white / near-white" and blanked out.
# 230 ≈ 90 % of 255 — only catches truly white surfaces; gray (~200 or below)
# is left untouched.  Raise toward 245 to be even more conservative.
WHITE_BRIGHTNESS_MIN = 170
WHITE_SAT_MAX        = 30

# ─── Object-lock parameters ───────────────────────────────────────────────────
# Once an object is selected, the lock is held for this many seconds as long as
# the camera is not moving.  Any motion above FRAME_MOTION_MAD breaks the lock
# immediately so a new target can be selected.
LOCK_DURATION_S  = 60.0   # seconds to keep a locked target
FRAME_MOTION_MAD = 8.0    # mean-abs-diff threshold (0-255) to declare camera moved
FRAME_MOTION_SIZE = (80, 60)  # downscale resolution for cheap motion check

IMG_CENTER       = np.array([COLOR_WIDTH / 2, COLOR_HEIGHT / 2])

# ─── Red-object colour segmentation ──────────────────────────────────────────
# YOLO is bypassed entirely.  Instead the pipeline finds the red blob closest
# to the image centre and uses its contour mask for point-cloud isolation.
#
# Red occupies two hue bands in HSV because the hue channel wraps at 180°:
#   low  band : 0 – RED_HUE_HIGH1   (orange-red)
#   high band : RED_HUE_LOW2 – 179  (magenta-red)
RED_HUE_HIGH1 =  10   # upper hue of the low  red band  (0–10)
RED_HUE_LOW2  = 160   # lower hue of the high red band  (160–179)
RED_SAT_MIN   =  80   # minimum HSV saturation (reject washed-out colours)
RED_VAL_MIN   =  50   # minimum HSV value      (reject near-black pixels)
RED_MIN_AREA  = 500   # minimum contour area in pixels  (reject noise specks)

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


def _build_foreground_mask(bgr: np.ndarray, depth_frame,
                           exclude_box=None) -> np.ndarray:
    """Return a bool mask (H, W) that is True where a pixel is likely foreground.

    Two stages — applied before red detection so noisy background is removed:

    1. **Depth-discontinuity removal** — pixels whose neighbourhood depth span
       exceeds ``DEPTH_GAP_UNITS`` sit on a surface boundary and are masked out,
       along with any pixel that has no depth reading.

    2. **White / near-white background removal** — pixels with high brightness
       and low colour saturation are masked out.  This step is skipped inside
       ``exclude_box`` (the last known object bounding box) so that white parts
       of the object itself are never erased.

    Parameters
    ----------
    exclude_box : array-like [x1, y1, x2, y2] | None
        If provided, white-mask removal is suppressed inside this rectangle.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (DEPTH_KERNEL_SIZE, DEPTH_KERNEL_SIZE)
    )

    # ── 1. Depth discontinuity ────────────────────────────────────────────
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    invalid = depth == 0

    depth_max = cv2.dilate(depth, kernel)

    depth_erode_in = depth.copy()
    depth_erode_in[invalid] = 65535.0
    depth_min = cv2.erode(depth_erode_in, kernel)
    depth_min[depth_min >= 65535.0] = 0.0

    depth_gap_mask = (depth_max - depth_min > DEPTH_GAP_UNITS) | invalid

    # ── 2. White / near-white background ─────────────────────────────────
    brightness = bgr.max(axis=2).astype(np.float32)
    saturation = (bgr.max(axis=2) - bgr.min(axis=2)).astype(np.float32)
    white_mask = (brightness > WHITE_BRIGHTNESS_MIN) & (saturation < WHITE_SAT_MAX)

    # Don't remove white pixels inside the previously detected object box —
    # the object itself may have white markings or surfaces.
    if exclude_box is not None:
        x1, y1, x2, y2 = exclude_box
        white_mask[y1:y2, x1:x2] = False

    # Foreground = neither a depth-edge pixel nor a white-background pixel
    return ~(depth_gap_mask | white_mask)


def _apply_foreground_mask(bgr: np.ndarray,
                           fg_mask: np.ndarray) -> np.ndarray:
    """Black out pixels outside *fg_mask* so YOLO ignores them."""
    out = bgr.copy()
    out[~fg_mask] = 0
    return out


def _detect_red_mask(bgr: np.ndarray):
    """Find the red blob closest to the image centre.

    Uses HSV colour thresholding (two hue bands because red wraps at 180°),
    followed by morphological open+close to remove noise and fill small holes.
    The contour closest to the image centre (by bounding-box centroid) is
    returned as a boolean mask and its axis-aligned bounding box.

    Returns ``(mask_bool, box_xyxy_int)`` or ``(None, None)`` when no
    sufficiently large red region is found.
    """
    h, w = bgr.shape[:2]
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Two inRange calls cover the wrap-around red hue bands.
    mask_lo = cv2.inRange(hsv,
                          (0,             RED_SAT_MIN, RED_VAL_MIN),
                          (RED_HUE_HIGH1, 255,         255))
    mask_hi = cv2.inRange(hsv,
                          (RED_HUE_LOW2, RED_SAT_MIN, RED_VAL_MIN),
                          (179,          255,         255))
    red_raw = cv2.bitwise_or(mask_lo, mask_hi)

    # Morphological cleanup: open removes speckle, close fills interior holes.
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red_raw = cv2.morphologyEx(red_raw, cv2.MORPH_OPEN,  kernel, iterations=2)
    red_raw = cv2.morphologyEx(red_raw, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(red_raw, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    best_mask, best_box, best_dist = None, None, float("inf")
    for cnt in contours:
        if cv2.contourArea(cnt) < RED_MIN_AREA:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)

        # Skip blobs that flood most of the frame (likely background).
        if (cw * ch) / (w * h) > MAX_MASK_FILL:
            continue

        cx, cy = x + cw / 2.0, y + ch / 2.0
        dist    = np.linalg.norm(np.array([cx, cy]) - IMG_CENTER)
        if dist < best_dist:
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(m, [cnt], -1, 255, cv2.FILLED)
            best_mask = m.astype(bool)
            best_box  = np.array([x, y, x + cw, y + ch])
            best_dist = dist

    return best_mask, best_box


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
        self.ready        = threading.Event()   # set after YOLO warm-up
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
            full_verts, _raw_colors, full_colors, obj_verts, obj_colors, preview_bgr = \
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

        # Colour-based detection — no model to load; signal ready immediately.
        print("[ObjectIsolator] colour-based (red) detection — ready.")
        self.ready.set()

        import time as _time
        print("[ObjectIsolator] camera + YOLO ready, streaming frames...")

        frame_idx       = 0
        last_mask       = None
        last_box        = None
        last_label      = None
        lock_end_time   = 0.0     # monotonic timestamp when current lock expires
        need_redetect   = True    # force YOLO on the very next eligible frame
        prev_gray_small = None    # previous small greyscale for motion check
        _last_log       = 0.0

        try:
            while not self._stop_event.is_set():

                # ── 1. Grab aligned RGB-D frame ───────────────────────────
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

                # ── 2. Pre-YOLO foreground filtering ──────────────────────
                # Build a pixel-level foreground mask (depth gaps + white BG)
                # and blank those regions before YOLO sees the image.
                fg_mask   = _build_foreground_mask(bgr, depth_fr, exclude_box=last_box)
                yolo_bgr  = _apply_foreground_mask(bgr, fg_mask)

                # ── 3. Motion check + lock / re-detect logic ───────────────
                now = _time.monotonic()

                # Cheap per-frame motion estimate on a tiny greyscale image.
                # Use the *original* bgr so masking artefacts don't pollute
                # the motion signal.
                gray_small = cv2.resize(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                    FRAME_MOTION_SIZE,
                )
                if prev_gray_small is not None:
                    mad = float(np.mean(
                        np.abs(gray_small.astype(np.float32) -
                               prev_gray_small.astype(np.float32))
                    ))
                    camera_moved = mad >= FRAME_MOTION_MAD
                else:
                    camera_moved = False
                prev_gray_small = gray_small

                # Break lock on motion or expiry.
                if camera_moved:
                    if last_mask is not None:
                        print("[ObjectIsolator] camera moved — re-selecting target")
                    need_redetect = True
                    lock_end_time = 0.0
                elif now >= lock_end_time and last_mask is not None:
                    print("[ObjectIsolator] lock expired — re-selecting target")
                    need_redetect = True
                    lock_end_time = 0.0

                # Determine whether to run YOLO this frame.
                # - Always run when a re-detect is needed.
                # - Skip when locked (camera stable + within 60 s).
                frame_idx += 1
                locked = (last_mask is not None) and (now < lock_end_time)
                run_yolo = need_redetect or (
                    not locked and frame_idx % YOLO_SKIP_FRAMES == 0
                )

                if run_yolo:
                    new_mask, new_box = _detect_red_mask(bgr)
                    new_label = "red" if new_mask is not None else None
                    if new_mask is not None:
                        if last_label != new_label:
                            print(f"[ObjectIsolator] locked onto red object "
                                  f"for {LOCK_DURATION_S:.0f} s")
                        last_mask, last_box, last_label = new_mask, new_box, new_label
                        lock_end_time = now + LOCK_DURATION_S
                        need_redetect = False
                    elif need_redetect:
                        # No red object in view; clear previous result.
                        last_mask = last_box = last_label = None

                # ── 4. Build subsampled point cloud ───────────────────────
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

                # ── 5. Mask along segmentation contour ────────────────────
                if last_mask is not None:
                    inside         = last_mask[v, u]
                    full_colors    = np.full_like(colors, 0.35)
                    full_colors[inside] = colors[inside]
                    obj_verts_raw  = verts[inside]
                    obj_colors_raw = colors[inside]
                else:
                    full_colors    = colors
                    obj_verts_raw  = np.zeros((0, 3), np.float32)
                    obj_colors_raw = np.zeros((0, 3), np.float32)

                # ── 7. cv2 preview with YOLO overlay ──────────────────────
                # Tint foreground-masked regions dark red so it is clear what
                # the pre-processing removed before YOLO ran.
                preview_bgr = bgr.copy()
                preview_bgr[~fg_mask] = (preview_bgr[~fg_mask] * 0.25).astype(np.uint8)
                if last_box is not None:
                    overlay = np.zeros_like(preview_bgr)
                    overlay[last_mask] = (0, 255, 0)
                    preview_bgr = cv2.addWeighted(preview_bgr, 0.7, overlay, 0.3, 0)
                    x1, y1, x2, y2 = last_box
                    cv2.rectangle(preview_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Class label with a filled background for readability
                    label      = last_label if last_label else "target"
                    font       = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.65
                    thickness  = 2
                    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    lx, ly = x1, max(y1 - 4, th + baseline)  # keep label inside frame
                    cv2.rectangle(preview_bgr,
                                  (lx, ly - th - baseline),
                                  (lx + tw, ly + baseline),
                                  (0, 255, 0), cv2.FILLED)
                    cv2.putText(preview_bgr, label, (lx, ly),
                                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

                if obj_verts_raw.shape[0] >= self._min_points:
                    obj_verts, obj_colors = obj_verts_raw, obj_colors_raw
                else:
                    obj_verts  = np.zeros((0, 3), np.float32)
                    obj_colors = np.zeros((0, 3), np.float32)

                # ── Lock-status banner at the top of the preview ──────────
                locked_now = (last_mask is not None) and (now < lock_end_time)
                if locked_now:
                    secs_left  = max(0.0, lock_end_time - now)
                    status_txt = f"LOCKED  {secs_left:4.0f}s"
                    banner_col = (0, 200, 0)    # green
                else:
                    status_txt = "SEARCHING..."
                    banner_col = (0, 140, 255)  # orange

                font_s = cv2.FONT_HERSHEY_SIMPLEX
                (bw, bh), bl = cv2.getTextSize(status_txt, font_s, 0.6, 2)
                cv2.rectangle(preview_bgr, (0, 0), (bw + 10, bh + bl + 8),
                              banner_col, cv2.FILLED)
                cv2.putText(preview_bgr, status_txt, (5, bh + 4),
                            font_s, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

                if now - _last_log >= 1.0:
                    print(f"[ObjectIsolator] scene pts: {len(verts)}  "
                          f"obj pts: {len(obj_verts)}  "
                          f"target: {last_label if last_label else 'none'}  "
                          f"{'LOCKED' if locked_now else 'searching'}")
                    _last_log = now

                # ── 6. Push to queue (drop stale frame) ───────────────────
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self._frame_queue.put(
                    (verts, colors, full_colors, obj_verts, obj_colors, preview_bgr)
                )

        except Exception as exc:
            import traceback
            print(f"[ObjectIsolator] FATAL: {exc}")
            traceback.print_exc()
        finally:
            pipeline.stop()