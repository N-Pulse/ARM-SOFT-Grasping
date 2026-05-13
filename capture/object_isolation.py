"""
object_isolation.py

Importable ObjectIsolator class — captures live RGB-D frames from an Intel
D405, runs YOLO segmentation, and exposes the isolated object point cloud
(masked along the segmentation contour, not the bounding box).

Key behaviours
--------------
  - Only graspable table objects are detected:
      · Persons, vehicles, and dining tables are excluded via
        YOLO_EXCLUDE_CLASSES.
      · Detections whose mask covers more than YOLO_MAX_MASK_RATIO of the
        frame are discarded — this prevents the table surface itself from
        being locked as a target even when YOLO mislabels it.
      · An optional whitelist (YOLO_ALLOWED_CLASSES) can restrict detection
        to specific COCO classes.
  - Valid depth range is 7 cm – 50 cm (arm's-reach objects only).
  - Once a target is chosen it is locked for TARGET_LOCK_SECONDS (60 s).
    The lock tracks the same object by bbox-centre proximity; it only breaks
    when the object drifts more than TARGET_LOCK_MAX_DRIFT pixels or
    disappears entirely.

Performance optimisations
--------------------------
  - YOLO runs at imgsz=320
  - Inference skipped every YOLO_SKIP_FRAMES frames (last mask reused)
  - Point cloud subsampled by SUBSAMPLE factor
  - TensorRT engine auto-used if .engine file exists alongside .pt

Usage
-----
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

# ─── Camera & stream configuration ───────────────────────────────────────────

DEPTH_WIDTH  = 640
DEPTH_HEIGHT = 480
COLOR_WIDTH  = 640
COLOR_HEIGHT = 480
FPS          = 30

# Valid depth range in metres — discard points outside this window
MIN_DEPTH_M  = 0.07   # 7 cm  — closest usable range for D405
MAX_DEPTH_M  = 0.50   # 50 cm — only consider objects within arm's reach

# ─── Model discovery ─────────────────────────────────────────────────────────

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)


def _find_model(filename: str) -> str:
    """Search common locations for a model file; fall back to filename as-is."""
    for base in (_THIS_DIR, _PROJECT_ROOT, os.getcwd()):
        candidate = os.path.join(base, filename)
        if os.path.exists(candidate):
            return candidate
    return filename


_PT_MODEL     = _find_model("yolo11n-seg.pt")
_ENGINE_MODEL = _find_model("yolo11n-seg.engine")
# Prefer a compiled TensorRT engine when available (faster on Jetson)
YOLO_MODEL    = _ENGINE_MODEL if os.path.exists(_ENGINE_MODEL) else _PT_MODEL

# ─── YOLO inference settings ─────────────────────────────────────────────────

YOLO_CONF        = 0.35   # minimum detection confidence
YOLO_IOU         = 0.45   # NMS IoU threshold
YOLO_IMGSZ       = 320    # inference resolution (smaller = faster)
YOLO_SKIP_FRAMES = 3      # run YOLO every N frames; reuse last mask in between

# ─── Class filtering ─────────────────────────────────────────────────────────
# COCO class reference:
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml

YOLO_EXCLUDE_CLASSES = {
    0,                        # person
    1, 2, 3, 4, 5, 6, 7, 8,  # bicycle, car, motorcycle, airplane, bus, train, truck, boat
    60,                       # dining table — stops the table surface being locked as target
}

# Optional whitelist — set to None to allow every non-excluded class.
# Uncomment and populate to restrict detection to specific graspable objects.
# YOLO_ALLOWED_CLASSES = {
#     39, 41, 42, 43, 44, 45,                  # bottle, cup, fork, knife, spoon, bowl
#     46, 47, 48, 49, 50, 51, 52, 53, 54, 55,  # common food items
#     63, 64, 65, 66, 67,                      # laptop, mouse, remote, keyboard, phone
#     73, 74, 75, 76, 77,                      # book, clock, vase, scissors, teddy bear
# }
YOLO_ALLOWED_CLASSES = None   # None = no whitelist restriction

# Reject any detection whose mask covers more than this fraction of the image.
# A graspable object should never fill 40 %+ of the frame; anything larger is
# almost certainly the table surface or background, even if YOLO mislabels it
# as something other than "dining table".
YOLO_MAX_MASK_RATIO = 0.40

# ─── Target locking ──────────────────────────────────────────────────────────

TARGET_LOCK_SECONDS   = 60.0   # once chosen, keep the same target for this long
TARGET_LOCK_MAX_DRIFT = 150    # pixels — max bbox-centre shift before lock breaks

# ─── Point cloud settings ────────────────────────────────────────────────────

SUBSAMPLE  = 1    # keep every Nth point (reduces compute & memory)

# Pixel coordinates of the image centre — used for initial target selection
IMG_CENTER = np.array([COLOR_WIDTH / 2, COLOR_HEIGHT / 2])


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _build_pipeline():
    """
    Initialise the RealSense D405 pipeline with depth + colour streams,
    spatial/temporal/hole-filling filters, alignment, and a point-cloud
    utility object.

    Returns
    -------
    pipeline, align, spatial, temporal, holes, rs.pointcloud
    """
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH,  DEPTH_HEIGHT, rs.format.z16,  FPS)
    config.enable_stream(rs.stream.color, COLOR_WIDTH,  COLOR_HEIGHT, rs.format.bgr8, FPS)

    profile      = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 4)  # preset 4 = Short Range

    # Align depth pixels to colour pixels
    align = rs.align(rs.stream.color)

    # Post-processing filters to reduce noise and fill holes
    spatial  = rs.spatial_filter()
    temporal = rs.temporal_filter()
    holes    = rs.hole_filling_filter()
    holes.set_option(rs.option.holes_fill, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    return pipeline, align, spatial, temporal, holes, rs.pointcloud()


def _detect_masks(model, bgr_image):
    """
    Run YOLO segmentation on *bgr_image* and return a list of valid detections.

    Each detection is a tuple:
        (mask_resized, box_xyxy, bbox_center)

        mask_resized : bool ndarray (H, W) — True inside the segmented object
        box_xyxy     : int ndarray (4,)    — [x1, y1, x2, y2]
        bbox_center  : float ndarray (2,)  — [(x1+x2)/2, (y1+y2)/2]

    A detection is dropped if any of the following are true:
        · Its class ID is in YOLO_EXCLUDE_CLASSES  (person, table, vehicles…)
        · YOLO_ALLOWED_CLASSES is set and the class ID is not in it
        · Its mask covers more than YOLO_MAX_MASK_RATIO of the image area
          (catches the table surface even when YOLO mislabels it)

    Parameters
    ----------
    model      : loaded Ultralytics YOLO model
    bgr_image  : uint8 ndarray (H, W, 3) in BGR order
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

        for mask_tensor, box, cls_tensor in zip(
            result.masks.data,
            result.boxes.xyxy,
            result.boxes.cls,
        ):
            cls_id = int(cls_tensor.item())

            # ── Class filtering ────────────────────────────────────────────
            if cls_id in YOLO_EXCLUDE_CLASSES:
                continue   # explicitly excluded (person, table, vehicles…)
            if YOLO_ALLOWED_CLASSES is not None and cls_id not in YOLO_ALLOWED_CLASSES:
                continue   # not in the graspable-object whitelist

            # Resize segmentation mask to full image resolution
            mask_np = mask_tensor.cpu().numpy()
            mask_resized = cv2.resize(
                mask_np,
                (bgr_image.shape[1], bgr_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            # ── Size filter — reject masks that cover too much of the frame ─
            # A graspable object should never fill 40 %+ of the image.
            # This catches the table surface even when YOLO mislabels it
            # as something other than "dining table" (class 60).
            mask_ratio = mask_resized.sum() / mask_resized.size
            if mask_ratio > YOLO_MAX_MASK_RATIO:
                continue

            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            bbox_center     = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

            detections.append((mask_resized, np.array([x1, y1, x2, y2]), bbox_center, cls_id))

    return detections


def _select_central(detections, locked_center=None):
    """
    Choose the best detection from *detections*.

    Strategy
    --------
    1. If *locked_center* is provided (active lock), find the detection whose
       bbox centre is closest to it.  If that distance is within
       TARGET_LOCK_MAX_DRIFT, return it — this keeps tracking the same object.
    2. If no valid locked detection is found (or no lock is active), fall back
       to the detection whose bbox centre is closest to the image centre.

    Parameters
    ----------
    detections    : list of (mask, box, center) tuples from _detect_masks
    locked_center : float ndarray (2,) or None

    Returns
    -------
    (mask, box, center) of the chosen detection, or (None, None, None).
    """
    if not detections:
        return None, None, None, None

    if locked_center is not None:
        # Try to re-acquire the locked target by proximity
        best = min(detections, key=lambda d: np.linalg.norm(d[2] - locked_center))
        if np.linalg.norm(best[2] - locked_center) <= TARGET_LOCK_MAX_DRIFT:
            return best   # (mask, box, center, cls_id) — lock holds

        # Locked target has drifted too far or vanished; fall through
        # to the image-centre strategy below

    # No active lock (or lock broken): pick the detection closest to frame centre
    best = min(detections, key=lambda d: np.linalg.norm(d[2] - IMG_CENTER))
    return best


# ─── Public class ─────────────────────────────────────────────────────────────

class ObjectIsolator:
    """
    Runs D405 capture + YOLO segmentation on a background thread.
    Point cloud is masked along the object's segmentation contour.

    Target locking
    --------------
    The first graspable object found (closest to the image centre) is locked
    as the target.  For the next TARGET_LOCK_SECONDS (60 s) the isolator
    continues tracking *that same object* even if another object enters the
    scene and is closer to the centre.  The lock is released only when:
      - the object's bbox centre moves more than TARGET_LOCK_MAX_DRIFT pixels
        from where it was last seen, or
      - the object is no longer detected at all.

    Parameters
    ----------
    min_points : int
        Minimum number of points for a frame to be considered valid.
    """

    def __init__(self, min_points: int = 50):
        self.ready        = threading.Event()   # set once YOLO warm-up completes
        self._min_points  = min_points
        self._frame_queue = queue.Queue(maxsize=1)
        self._stop_event  = threading.Event()
        self._thread      = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Start the background capture/inference thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the background thread to stop and wait for it to exit."""
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
        Non-blocking.  Returns the latest processed frame as a 3-tuple:

            (full_pcd, iso_pcd, preview_bgr)

            full_pcd    : o3d.PointCloud of the entire scene.
                          Target region is rendered in its real colour;
                          background is greyed out.
            iso_pcd     : o3d.PointCloud of the isolated object only,
                          or None if no target is currently locked.
            preview_bgr : annotated BGR image with YOLO mask overlay and
                          bounding box.

        Returns None if no frame is available yet.
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

    def get_pcd(self) -> "o3d.geometry.PointCloud | None":
        """
        Non-blocking convenience method.
        Returns the latest isolated object point cloud, or None.
        """
        result = self.get_full_frame()
        if result is None:
            return None
        _, iso_pcd, _ = result
        return iso_pcd

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self):
        """Main capture + inference loop; runs on the background thread."""

        # ── Initialise RealSense pipeline ─────────────────────────────────
        try:
            pipeline, align, spatial, temporal, holes, pc_util = _build_pipeline()
        except Exception as exc:
            print(f"[ObjectIsolator] FATAL: could not start RealSense pipeline: {exc}")
            return

        # ── Load and warm up YOLO ─────────────────────────────────────────
        try:
            print(f"[ObjectIsolator] loading '{YOLO_MODEL}' ...")
            model = YOLO(YOLO_MODEL)
            if not YOLO_MODEL.endswith(".engine"):
                model.fuse()   # fuse Conv+BN layers for faster CPU/GPU inference
            print("[ObjectIsolator] YOLO loaded ✓ — warming up...")
            # One dummy inference to initialise CUDA kernels before real-time use
            model.predict(
                source=np.zeros((320, 320, 3), dtype=np.uint8),
                imgsz=YOLO_IMGSZ,
                verbose=False,
            )
            print("[ObjectIsolator] warm-up done ✓")
            self.ready.set()
        except Exception as exc:
            print(f"[ObjectIsolator] FATAL: could not load YOLO model: {exc}")
            pipeline.stop()
            return

        import time as _time
        print("[ObjectIsolator] camera + YOLO ready, streaming frames...")

        frame_idx = 0
        last_mask    = None   # most recent segmentation mask (H, W) bool
        last_box     = None   # most recent bbox [x1, y1, x2, y2]
        last_cls_id  = None   # COCO class ID of the locked target

        # ── Target-lock state ─────────────────────────────────────────────
        locked_center = None   # bbox centre of the locked target (ndarray or None)
        lock_start    = None   # monotonic timestamp when the lock was acquired
        _last_log     = 0.0    # rate-limit console prints to ~1 Hz

        try:
            while not self._stop_event.is_set():

                # ── 1. Grab an aligned RGB-D frame ────────────────────────
                frames   = pipeline.wait_for_frames()
                aligned  = align.process(frames)
                depth_fr = aligned.get_depth_frame()
                color_fr = aligned.get_color_frame()
                if not depth_fr or not color_fr:
                    continue

                # Apply depth post-processing filters
                depth_fr = spatial.process(depth_fr)
                depth_fr = temporal.process(depth_fr)
                depth_fr = holes.process(depth_fr)

                bgr = np.asanyarray(color_fr.get_data())

                # ── 2. Run YOLO every YOLO_SKIP_FRAMES frames ─────────────
                #       Between inference frames the previous mask is reused,
                #       which keeps CPU/GPU load manageable.
                frame_idx += 1
                if frame_idx % YOLO_SKIP_FRAMES == 0:
                    detections = _detect_masks(model, bgr)
                    now_t      = _time.monotonic()

                    # Determine whether the current lock is still within its time window
                    lock_active = (
                        lock_start is not None
                        and (now_t - lock_start) < TARGET_LOCK_SECONDS
                    )

                    # Pass the locked centre only when the lock is still valid
                    lc = locked_center if lock_active else None
                    new_mask, new_box, new_center, new_cls_id = _select_central(
                        detections, locked_center=lc
                    )

                    if new_mask is not None:
                        last_mask     = new_mask
                        last_box      = new_box
                        locked_center = new_center
                        last_cls_id   = new_cls_id

                        # Start a fresh lock if there is none or the previous one expired
                        if not lock_active:
                            lock_start = now_t
                            print(
                                f"[ObjectIsolator] new target locked at "
                                f"centre {new_center.astype(int).tolist()} — "
                                f"lock duration {TARGET_LOCK_SECONDS:.0f} s"
                            )
                    else:
                        # No valid detection — clear everything so we start
                        # fresh on the next frame where YOLO runs
                        last_mask     = None
                        last_box      = None
                        locked_center = None
                        lock_start    = None
                        last_cls_id   = None

                # No target yet — nothing to publish
                if last_mask is None:
                    continue

                # ── 3. Build a subsampled point cloud from the depth frame ─
                pc_util.map_to(color_fr)
                points_rs = pc_util.calculate(depth_fr)

                # Reshape the packed vertex / texcoord arrays into (N, 3) / (N, 2)
                verts = (
                    np.asanyarray(points_rs.get_vertices())
                    .view(np.float32)
                    .reshape(-1, 3)[::SUBSAMPLE]
                )
                texcoords = (
                    np.asanyarray(points_rs.get_texture_coordinates())
                    .view(np.float32)
                    .reshape(-1, 2)[::SUBSAMPLE]
                )

                # Remove points outside the valid depth range
                depth_vals = np.linalg.norm(verts, axis=1)
                valid      = (depth_vals > MIN_DEPTH_M) & (depth_vals < MAX_DEPTH_M)
                verts      = verts[valid]
                texcoords  = texcoords[valid]

                # Look up the RGB colour for every 3-D point via its texture coordinate
                h, w = bgr.shape[:2]
                u = np.clip((texcoords[:, 0] * w).astype(int), 0, w - 1)
                v = np.clip((texcoords[:, 1] * h).astype(int), 0, h - 1)
                colors = bgr[v, u, ::-1] / 255.0   # BGR → RGB, normalise to [0, 1]

                # ── 4. Separate target points from background ──────────────
                #       Points whose pixel falls inside the segmentation mask
                #       keep their real colour; everything else is greyed out.
                inside      = last_mask[v, u]
                full_colors = np.full_like(colors, 0.35)   # uniform grey background
                full_colors[inside] = colors[inside]

                obj_verts_raw  = verts[inside]
                obj_colors_raw = colors[inside]

                # ── 5. Build annotated preview image ──────────────────────
                preview_bgr = bgr.copy()
                if last_box is not None:
                    # Semi-transparent green overlay on the segmented region
                    overlay = np.zeros_like(preview_bgr)
                    overlay[last_mask] = (0, 255, 0)
                    preview_bgr = cv2.addWeighted(preview_bgr, 0.7, overlay, 0.3, 0)

                    # Bounding box and label
                    x1, y1, x2, y2 = last_box
                    cv2.rectangle(preview_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Show class name + remaining lock time in the label
                    cls_name = model.names[last_cls_id] if last_cls_id is not None else "?"
                    if lock_start is not None:
                        remaining = max(
                            0.0, TARGET_LOCK_SECONDS - (_time.monotonic() - lock_start)
                        )
                        label = f"{cls_name}  lock {remaining:.0f}s"
                    else:
                        label = cls_name
                    cv2.putText(
                        preview_bgr, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )

                # ── 6. Discard frames with too few object points ───────────
                if obj_verts_raw.shape[0] >= self._min_points:
                    obj_verts, obj_colors = obj_verts_raw, obj_colors_raw
                else:
                    obj_verts  = np.zeros((0, 3), np.float32)
                    obj_colors = np.zeros((0, 3), np.float32)

                # ── 7. Rate-limited console log (~1 Hz) ───────────────────
                now = _time.monotonic()
                if now - _last_log >= 1.0:
                    lock_remaining = (
                        max(0.0, TARGET_LOCK_SECONDS - (now - lock_start))
                        if lock_start is not None else 0.0
                    )
                    print(
                        f"[ObjectIsolator] scene pts: {len(verts):5d}  "
                        f"obj pts: {len(obj_verts):5d}  "
                        f"locked: {'yes' if locked_center is not None else 'no '}  "
                        f"lock remaining: {lock_remaining:5.1f}s"
                    )
                    _last_log = now

                # ── 8. Push to queue (drop stale frame if consumer is slow) ─
                try:
                    self._frame_queue.get_nowait()   # evict the old frame
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