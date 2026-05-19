"""
object_isolation.py

Importable ObjectIsolator class — captures live RGB-D frames from an Intel
D405, isolates a red object via HSV colour segmentation, and exposes the
masked point cloud on a background thread.

Pipeline per frame
------------------
1. Grab aligned RGB-D frame from the RealSense D405.
2. Build a foreground mask (depth-discontinuity removal + white-BG removal).
3. Run HSV red-colour segmentation; pick the blob closest to frame centre.
4. Build a subsampled point cloud; mask it to the red contour.
5. Push (verts, raw_colors, full_colors, obj_verts, obj_colors, preview_bgr)
   to a single-slot queue for the visualiser to consume.

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

import queue
import threading
import time

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

# ─── Camera configuration ─────────────────────────────────────────────────────

DEPTH_WIDTH  = 640
DEPTH_HEIGHT = 480
COLOR_WIDTH  = 640
COLOR_HEIGHT = 480
FPS          = 30

MIN_DEPTH_M  = 0.07
MAX_DEPTH_M  = 0.70
SUBSAMPLE    = 2

IMG_CENTER   = np.array([COLOR_WIDTH / 2, COLOR_HEIGHT / 2])

# ─── Foreground filtering ─────────────────────────────────────────────────────

# Depth-discontinuity removal: pixels whose neighbourhood depth span exceeds
# this value (raw D405 units; 1 unit ≈ 1 mm, so 30 ≈ 3 cm) are treated as
# surface-boundary noise and blanked out.
DEPTH_GAP_UNITS   = 30
DEPTH_KERNEL_SIZE = 5

# White-background removal: pixels where max(B,G,R) > threshold AND
# max-min < saturation cap are considered near-white and removed.
WHITE_BRIGHTNESS_MIN = 170
WHITE_SAT_MAX        = 30

# Blobs whose bounding box covers more than this fraction of the frame are
# rejected as background (table, floor, etc.).
MAX_MASK_FILL = 0.80

# ─── Red-object colour segmentation ──────────────────────────────────────────
# Red wraps around in HSV, so two hue bands are needed:
#   low  band : 0 – RED_HUE_HIGH1   (orange-red)
#   high band : RED_HUE_LOW2 – 179  (magenta-red)

RED_HUE_HIGH1 =  10   # upper hue of the low  red band
RED_HUE_LOW2  = 160   # lower hue of the high red band
RED_SAT_MIN   =  80   # minimum HSV saturation (rejects washed-out colours)
RED_VAL_MIN   =  50   # minimum HSV value      (rejects near-black pixels)
RED_MIN_AREA  = 500   # minimum contour area in pixels (rejects noise specks)


# ─── Cluster filtering ────────────────────────────────────────────────────────
# After colour-mask isolation, stray reflections or background leakage can
# leave disconnected point patches.  DBSCAN groups points into connected
# regions; keeping only the largest removes those fragments.

CLUSTER_EPS        = 0.02   # 2 cm — max neighbour distance within a cluster
CLUSTER_MIN_POINTS = 10     # fragments smaller than this are discarded


def keep_largest_cluster(verts: np.ndarray,
                         colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return only the points in the largest DBSCAN cluster.

    If no cluster is found or the input is too small, the original arrays are
    returned unchanged.
    """
    if len(verts) < CLUSTER_MIN_POINTS:
        return verts, colors

    tmp = o3d.geometry.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(verts)
    labels = np.array(tmp.cluster_dbscan(
        eps=CLUSTER_EPS,
        min_points=CLUSTER_MIN_POINTS,
        print_progress=False,
    ))

    valid = labels >= 0
    if not valid.any():
        return verts, colors

    largest = np.bincount(labels[valid]).argmax()
    mask    = labels == largest
    return verts[mask], colors[mask]


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
    """Return a bool mask (H, W) — True where a pixel is likely foreground.

    Stage 1 — depth-discontinuity removal: pixels on surface boundaries or
    with no depth reading are masked out.

    Stage 2 — white-background removal: near-white pixels are masked out,
    except inside ``exclude_box`` (the current object bounding box) so white
    parts of the object itself are preserved.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (DEPTH_KERNEL_SIZE, DEPTH_KERNEL_SIZE)
    )

    depth   = np.ascontiguousarray(np.asanyarray(depth_frame.get_data()), dtype=np.uint16)
    invalid = depth == 0

    depth_max      = cv2.dilate(depth, kernel).astype(np.float32)
    depth_erode_in = depth.copy()
    depth_erode_in[invalid] = 65535
    depth_min = cv2.erode(depth_erode_in, kernel).astype(np.float32)
    depth_min[depth_min >= 65535.0] = 0.0
    depth = depth.astype(np.float32)

    depth_gap_mask = (depth_max - depth_min > DEPTH_GAP_UNITS) | invalid

    brightness = bgr.max(axis=2).astype(np.float32)
    saturation = (bgr.max(axis=2) - bgr.min(axis=2)).astype(np.float32)
    white_mask = (brightness > WHITE_BRIGHTNESS_MIN) & (saturation < WHITE_SAT_MAX)

    if exclude_box is not None:
        x1, y1, x2, y2 = exclude_box
        white_mask[y1:y2, x1:x2] = False

    return ~(depth_gap_mask | white_mask)


def _detect_red_mask(bgr: np.ndarray):
    """Find the red blob closest to the image centre.

    Returns (mask_bool, box_xyxy_int) or (None, None) if nothing found.
    """
    h, w = bgr.shape[:2]
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask_lo = cv2.inRange(hsv,
                          (0,             RED_SAT_MIN, RED_VAL_MIN),
                          (RED_HUE_HIGH1, 255,         255))
    mask_hi = cv2.inRange(hsv,
                          (RED_HUE_LOW2, RED_SAT_MIN, RED_VAL_MIN),
                          (179,          255,         255))
    red_raw = cv2.bitwise_or(mask_lo, mask_hi)

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
        if (cw * ch) / (w * h) > MAX_MASK_FILL:
            continue
        cx, cy = x + cw / 2.0, y + ch / 2.0
        dist = np.linalg.norm(np.array([cx, cy]) - IMG_CENTER)
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
    Captures D405 RGB-D frames on a background thread, segments the red object
    closest to the frame centre, and exposes its point cloud via a queue.

    Parameters
    ----------
    min_points : int
        Minimum number of isolated points for a frame to be considered valid.
    classifier : ShapeClassifier | None
        Optional YOLOv8 shape classifier.  When provided, it is called each
        frame on the bounding-box crop and the result is included as
        ``shape_hint`` in the frame tuple (7th element).  Pass None to skip
        YOLO and rely on the geometric classifier in shape_fitter.py.
    """

    def __init__(self, min_points: int = 50, classifier=None):
        self.ready        = threading.Event()
        self._min_points  = min_points
        self._classifier  = classifier
        self._frame_queue = queue.Queue(maxsize=1)
        self._stop_event  = threading.Event()
        self._thread      = None

        # Latest frame data — readable from outside (e.g. data collector).
        # Written only by the background thread; reads are best-effort / non-blocking.
        self.last_box:   np.ndarray | None = None  # [x1, y1, x2, y2] pixel coords
        self.last_bgr:   np.ndarray | None = None  # raw BGR frame (no overlays)
        self.last_shape: str | None        = None  # latest shape label for preview

        # YOLO rate-limiting state (background thread only)
        self._yolo_shape:     str | None = None   # sticky last accepted result
        self._yolo_last_time: float      = 0.0    # monotonic time of last YOLO call
        self._yolo_had_obj:   bool       = False  # whether object was present last frame

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

        full_pcd    : entire scene; object in real colour, background greyed out.
        iso_pcd     : isolated red object only, or None if not detected.
        preview_bgr : annotated BGR image showing the red mask overlay.

        Note: shape_hint (7th queue element) is consumed internally and
        accessible via isolator.shape_hint after each call.
        """
        try:
            full_verts, _raw_colors, full_colors, obj_verts, obj_colors, \
                preview_bgr, shape_hint = self._frame_queue.get_nowait()
        except queue.Empty:
            return None

        self.shape_hint = shape_hint   # expose for FitWorker / callbacks

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
        """Returns the latest isolated object point cloud, or None. Non-blocking."""
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

        print("[ObjectIsolator] camera ready — colour-based (red) detection active.")
        self.ready.set()

        last_mask   = None
        last_box    = None
        _last_log   = 0.0
        _prev_box   = None   # bbox from previous frame, for DBSCAN skip check
        self.shape_hint: str | None = None    # latest YOLO classification result

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
                self.last_bgr = bgr   # expose raw frame before any drawing

                # ── 2. Foreground filtering ───────────────────────────────
                fg_mask = _build_foreground_mask(bgr, depth_fr,
                                                 exclude_box=last_box)

                # ── 3. Red-object detection ───────────────────────────────
                last_mask, last_box = _detect_red_mask(bgr)
                self.last_box = last_box   # expose for data-collector / outside callers

                # ── 3b. YOLO shape classification — rate-limited ─────────────
                # Run YOLO only when:
                #   · Object just appeared  (no red segment → red segment)
                #   · 60 s have elapsed since the last successful classification
                # All other frames reuse the last accepted result (sticky hint).
                # This avoids running heavy inference every frame and is the
                # primary source of lag reduction.
                _YOLO_INTERVAL = 60.0   # seconds between periodic re-checks

                has_obj_now = last_box is not None
                now         = time.monotonic()

                if self._classifier is not None and has_obj_now:
                    newly_appeared   = not self._yolo_had_obj
                    periodic_recheck = (now - self._yolo_last_time) >= _YOLO_INTERVAL

                    if newly_appeared or periodic_recheck:
                        x1, y1, x2, y2 = last_box
                        crop = bgr[y1:y2, x1:x2]
                        if crop.size > 0:
                            result = self._classifier.predict(crop)
                            if result is not None:
                                self._yolo_shape     = result
                                self._yolo_last_time = now
                                reason = "new detection" if newly_appeared else "60 s check"
                                print(f"[ObjectIsolator]  YOLO re-classify ({reason})"
                                      f" → {result}")

                self._yolo_had_obj = has_obj_now
                # Expose sticky hint: keeps last known class while object is
                # present; resets to None only when the object disappears.
                shape_hint = self._yolo_shape if has_obj_now else None
                if not has_obj_now:
                    self._yolo_shape = None   # force re-classify on next appearance

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

                # ── 5. Mask point cloud to red contour ────────────────────
                if last_mask is not None:
                    inside         = last_mask[v, u]
                    full_colors    = np.full_like(colors, 0.35)
                    full_colors[inside] = colors[inside]

                    obj_verts_raw, obj_colors_raw = keep_largest_cluster(
                        verts[inside], colors[inside]
                    )
                    _prev_box = last_box
                else:
                    full_colors    = colors
                    obj_verts_raw  = np.zeros((0, 3), np.float32)
                    obj_colors_raw = np.zeros((0, 3), np.float32)
                    _prev_box       = None
                    self.last_shape = None   # clear label when object is lost

                # ── 6. cv2 preview ────────────────────────────────────────
                preview_bgr = bgr.copy()
                preview_bgr[~fg_mask] = (preview_bgr[~fg_mask] * 0.25).astype(np.uint8)
                if last_box is not None:
                    overlay = np.zeros_like(preview_bgr)
                    overlay[last_mask] = (0, 255, 0)
                    preview_bgr = cv2.addWeighted(preview_bgr, 0.7, overlay, 0.3, 0)
                    x1, y1, x2, y2 = last_box
                    cv2.rectangle(preview_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Show YOLO hint if available, otherwise last known shape,
                    # otherwise "?" while waiting for the first fit result.
                    display_label = (shape_hint if shape_hint is not None
                                     else (self.last_shape
                                           if self.last_shape is not None
                                           else "?"))
                    cv2.putText(preview_bgr, display_label, (x1, max(y1 - 6, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                                cv2.LINE_AA)

                if obj_verts_raw.shape[0] >= self._min_points:
                    obj_verts, obj_colors = obj_verts_raw, obj_colors_raw
                else:
                    obj_verts  = np.zeros((0, 3), np.float32)
                    obj_colors = np.zeros((0, 3), np.float32)

                now = time.monotonic()
                if now - _last_log >= 1.0:
                    print(f"[ObjectIsolator] scene pts: {len(verts)}  "
                          f"obj pts: {len(obj_verts)}  "
                          f"red: {'detected' if last_mask is not None else 'none'}")
                    _last_log = now

                # ── 7. Push to queue (drop stale frame) ───────────────────
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self._frame_queue.put(
                    (verts, colors, full_colors, obj_verts, obj_colors,
                     preview_bgr, shape_hint)
                )

        except Exception as exc:
            import traceback
            print(f"[ObjectIsolator] FATAL: {exc}")
            traceback.print_exc()
        finally:
            pipeline.stop()
