from __future__ import annotations

import sys
import os
import datetime
import importlib.util
from pathlib import Path
from typing import Optional, Tuple
import subprocess
import time

import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs


# ────────────────────── Configuration ────────────────────────────────
MIN_POINTS_THRESHOLD = 100   
SAVE_DEBUG_CLOUDS = True     
# ────────────────────── Project paths ────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
for sub in ("partial_pcd", "shape_identification"):
    p = BASE_DIR / sub
    if p.exists():
        sys.path.insert(0, str(p))

# ────────────────────── Utility loader ───────────────────────────────

def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

# ────────────────────── YOLO Object Selector ──────────────────────────

class YOLOSelector:
    """Object selector based on YOLO"""
    
    def __init__(self):
        print("[INFO] Loading YOLO model for object selection...")
        self.model = YOLO('yolov8n.pt')  
        
        # Configuration
        self.ignored_classes = ["dining table", "person"]  # Classes to ignore
        self.confidence_threshold = 0.7
        self.camera_angle = "face"  
        
        # Selection state
        self.selected_box = None
        self.selected_class = None
        self.selected_confidence = 0.0
        
        print(f"[YOLO] Model loaded. Mode: {self.camera_angle}")
    
    def set_camera_angle(self, angle: str):
        """Change camera angle (face/left/right)"""
        if angle in ["face", "gauche", "droite"]:
            self.camera_angle = angle
            print(f"[YOLO] Mode changed: {angle}")
        else:
            print(f"[YOLO] Invalid angle: {angle}")
    
    def cycle_camera_angle(self):
        """Cycle between face -> left -> right -> face"""
        angles = ["face", "gauche", "droite"]
        current_idx = angles.index(self.camera_angle)
        next_idx = (current_idx + 1) % len(angles)
        self.set_camera_angle(angles[next_idx])
    
    def process(self, frame: np.ndarray):
        """Process a frame to detect and select an object"""
        height, width = frame.shape[:2]
        center_x = width // 2
        
        # YOLO detection
        results = self.model(frame, verbose=False)
        boxes = results[0].boxes
        
        if boxes is None:
            self.selected_box = None
            return
        
        # Filter detections
        filtered_boxes = []
        for box in boxes:
            confidence = box.conf.item()
            class_id = int(box.cls)
            class_name = self.model.names[class_id]
            
            # Ignore irrelevant classes and low confidence detections
            if class_name not in self.ignored_classes and confidence >= self.confidence_threshold:
                filtered_boxes.append(box)
        
        # Selection according to camera angle
        self.selected_box = None
        
        if len(filtered_boxes) > 0:
            if self.camera_angle == "face":
                # Select the most centered object horizontally
                min_distance_to_center = float('inf')
                for box in filtered_boxes:
                    x1, _, x2, _ = map(int, box.xyxy[0])
                    box_center_x = (x1 + x2) // 2
                    distance_to_center = abs(box_center_x - center_x)
                    if distance_to_center < min_distance_to_center:
                        min_distance_to_center = distance_to_center
                        self.selected_box = box
            
            elif self.camera_angle == "gauche":
                # Select the leftmost object
                min_x = float('inf')
                for box in filtered_boxes:
                    x1, _, _, _ = map(int, box.xyxy[0])
                    if x1 < min_x:
                        min_x = x1
                        self.selected_box = box
            
            elif self.camera_angle == "droite":
                # Select the rightmost object
                max_x = -float('inf')
                for box in filtered_boxes:
                    _, _, x2, _ = map(int, box.xyxy[0])
                    if x2 > max_x:
                        max_x = x2
                        self.selected_box = box
        
        # Update selected object info
        if self.selected_box is not None:
            self.selected_confidence = self.selected_box.conf.item()
            class_id = int(self.selected_box.cls)
            self.selected_class = self.model.names[class_id]
    
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw overlay with the selected object"""
        overlay = frame.copy()
        
        # Draw selected object
        if self.selected_box is not None:
            x1, y1, x2, y2 = map(int, self.selected_box.xyxy[0])
            
            # Selection rectangle (green)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Label with class and confidence
            label = f"{self.selected_class}: {self.selected_confidence:.2f}"
            cv2.putText(overlay, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Show current mode and instructions
        mode_text = f"Mode: {self.camera_angle.upper()} | 'a': change | 'c': capture"
        cv2.putText(overlay, mode_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show selection status
        if self.selected_box is not None:
            status_text = f"SELECTED OBJECT: {self.selected_class}"
            color = (0, 255, 0)  # Green
        else:
            status_text = "No object selected (manual capture possible)"
            color = (0, 255, 255)  # Yellow
        
        cv2.putText(overlay, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return overlay
    
    def has_selection(self) -> bool:
        """Check if an object is selected"""
        return self.selected_box is not None
    
    def get_selection_info(self) -> Tuple[str, float]:
        """Return selected object info"""
        if self.selected_box is not None:
            return self.selected_class, self.selected_confidence
        return "None", 0.0

# ────────────────────── shape_id ────────────────
shape_mod = load_module("shape_id", BASE_DIR / "shape_identification" / "shape_id.py")
classify_shape = shape_mod.classify  # type: ignore[attr-defined]

import numpy as _np
if not hasattr(shape_mod, "_np2_patch"):
    def _tube_stats_np2(pts_c: _np.ndarray, vec: _np.ndarray):
        t = pts_c @ vec
        r = _np.linalg.norm(pts_c - _np.outer(t, vec), axis=1)
        m = r.mean() or 1e-6
        return r.std() / m, _np.ptp(t) / (2 * m)
    shape_mod.tube_stats = _tube_stats_np2  # type: ignore[attr-defined]
    shape_mod._np2_patch = True

# ────────────────────── YOLOv9 handle model ─────────────────────────
HANDLE_MODEL = (BASE_DIR / "handle_detection" / "cup_handle" / "runs" / "detect" /
                "train4" / "weights" / "best.pt")
if not HANDLE_MODEL.exists():
    sys.exit(f"❌  Handle model not found: {HANDLE_MODEL}")
print("[INFO] Loading handle model…")
yolo_handle = YOLO(str(HANDLE_MODEL))
YOLO_DEVICE = "mps" if sys.platform == "darwin" else "cpu"
YOLO_CONF   = 0.4

# ────────────────────── RealSense ─────────────────────────────────────
print("[INFO] Initializing RealSense …")
pipeline, config = rs.pipeline(), rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

align        = rs.align(rs.stream.color)
depth_scale  = profile.get_device().first_depth_sensor().get_depth_scale()

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
o3d_intr = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

# ────────────────────── YOLO selector initialization ────────────────
selector = YOLOSelector()

# ────────────────────── Helpers ───────────────────────────────────────

def detect_handle(img: np.ndarray, obj_class: str) -> Tuple[bool, np.ndarray | None]:
    """Returns (has_handle, annotated_img). If no handle or not 'cup': annotated_img=None."""
    res = yolo_handle.predict(img, device=YOLO_DEVICE, imgsz=640,
                              conf=YOLO_CONF, verbose=False)[0]
    if res.boxes is None or res.boxes.cls is None:
        return False, None
    handle_ids = [i for i, name in res.names.items() if "handle" in name.lower()]
    has = any(int(c) in handle_ids for c in res.boxes.cls.cpu().numpy())
    # Valid handle only if object is a "cup"
    if has and obj_class.lower() == "cup":
        return True, res.plot()
    return False, None


def clean_cloud(pcd: o3d.geometry.PointCloud) -> Optional[o3d.geometry.PointCloud]:
    pcd = pcd.voxel_down_sample(0.003)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    _, plane = pcd.segment_plane(0.004, 3, 800)
    pcd = pcd.select_by_index(plane, invert=True)
    labels = _np.array(pcd.cluster_dbscan(eps=0.01, min_points=30))
    if labels.size == 0:
        return None
    return pcd.select_by_index(_np.where(labels == _np.bincount(labels[labels >= 0]).argmax())[0])


def get_pointcloud(frames: rs.composite_frame) -> Optional[o3d.geometry.PointCloud]:
    aligned = align.process(frames)
    d, c = aligned.get_depth_frame(), aligned.get_color_frame()
    if not d or not c:
        return None
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(_np.asanyarray(c.get_data())),
        o3d.geometry.Image(_np.asanyarray(d.get_data())),
        depth_scale=1.0 / depth_scale, depth_trunc=3.0, convert_rgb_to_intensity=False)
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr)
    cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return clean_cloud(cloud)


def view_cloud(cloud: o3d.geometry.PointCloud, title: str):
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, 920, 720)
    cloud.paint_uniform_color([1, 0, 0])
    opt = vis.get_render_option(); opt.background_color, opt.point_size = _np.zeros(3), 5
    vis.add_geometry(cloud); vis.run(); vis.destroy_window()


# ────────────────────── Main loop ────────────────────────────
cv2.namedWindow("RealSense - YOLO Selection", cv2.WINDOW_NORMAL)
print("[INFO] c : capture (with or without YOLO selection) | a : change angle | q : quit")
subprocess.run(["python", "torque_ini.py"])
print("torque enabled")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame_bgr = _np.asanyarray(color_frame.get_data())

        # Update YOLO selection + overlay
        selector.process(frame_bgr)
        overlay = selector.draw_overlay(frame_bgr.copy())
        cv2.imshow("RealSense - YOLO Selection", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            selector.cycle_camera_angle()
        elif key == ord('c'):
            # ─── SELECTION INFORMATION (optional) ────────────────────────
            if selector.has_selection():
                obj_class, obj_conf = selector.get_selection_info()
                print(f"[CAPTURE] YOLO object selected: {obj_class} (confidence: {obj_conf:.2f})")
            else:
                obj_class, obj_conf = "Manual", 0.0
                print("[CAPTURE] No YOLO object selected - manual capture")
            
            # ─── SINGLE CAPTURE ───────────────────────────────────────────
            # Capture for handle detection
            cap_frames = pipeline.wait_for_frames()
            color_cap = cap_frames.get_color_frame()
            if not color_cap:
                print("[WARN] Missing color frame.")
                continue
            frame_cap = _np.asanyarray(color_cap.get_data())

            print("[CAPTURE] Handle detection...")
            has_handle, annotated = detect_handle(frame_cap, obj_class)

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save JPEG if handle detected
            if annotated is not None:
                jpg = BASE_DIR / f"handle_{ts}.jpg"
                cv2.imwrite(str(jpg), annotated)
                print(f"🖼  Handle image saved: {jpg.name}")

            # ─── CLASSIFICATION ON A SINGLE CLOUD ─────────────────────────
            print("[CAPTURE] Capturing point cloud...")
            cloud = get_pointcloud(cap_frames)
            
            if cloud is None or len(cloud.points) < MIN_POINTS_THRESHOLD:
                print("[WARN] Failed to capture cloud or too few points.")
                continue

            shape = classify_shape(_np.asarray(cloud.points))
            print(f"[CAPTURE] Detected shape: {shape} ({len(cloud.points)} points)")

            # Save the cloud
            ply = BASE_DIR / f"object_{ts}.ply"
            o3d.io.write_point_cloud(str(ply), cloud)
            print(f"✅  Cloud saved: {ply.name} ({len(cloud.points)} points)")

            print(f"🔎  Object: {obj_class} | Handle: {'Yes' if has_handle else 'No'} | Shape: {shape}")

            # Show the cloud
            title = f"{obj_class} - {shape} – Handle: {'Yes' if has_handle else 'No'}"
            if obj_class == "Manual":
                title = f"{shape} – Handle: {'Yes' if has_handle else 'No'} (Manual capture)"

            #### PLACE TO ADD LOGIC TO CHANGE ######
            view_cloud(cloud, title)

            # ─── CONFIRMATION BEFORE CONTINUING ───────────────────
            resp = input("Close the window. Continue the pipeline? [y/N] : ").strip().lower()
            print(f"Debug: received '{resp}', length={len(resp)}, repr={repr(resp)}")
            if 'n' in resp :
                print("Pipeline interrupted by user.\n")
                continue  # restart main loop

            # ─── MANUAL OVERRIDE OF SHAPE ───────────────────────
            override = input(f"Detected shape = '{shape}'. Press Enter to confirm or type another shape: ").strip()
            if override:
                print(f"→ Shape replaced: '{override}'\n")
                shape = override

            #########

            # ─── ROBOT ROUTING (uses shape) ─────────────────────────
            shape_l = shape.lower()
            robot_dir = BASE_DIR / "robot"
            
            if has_handle:                         
                robot_main = robot_dir / "hook_main.py"
                label = "hook"
                if not robot_main.exists():
                    print(f"[ERR] {robot_main.name} not found ⇒ {label} branch ignored.")
                else:
                    print(f"[PIPE] Shape {label} detected → launching {robot_main.name} …")
                    try:
                        # Pass handle image as argument
                        cmd = [sys.executable, str(robot_main), str(ply)]
                        if annotated is not None:
                            cmd.extend(["--handle_image", str(BASE_DIR / f"handle_{ts}.jpg")])
                        subprocess.run(cmd, cwd=robot_dir, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"[ERR] {robot_main.name} failed: {e}", file=sys.stderr)
                        

            elif shape_l == "cuboid":
                robot_main = robot_dir / "CUBOID2" / "cuboid_main.py"
                label = "cuboid"
                if not robot_main.exists():
                    print(f"[ERR] {robot_main.name} not found ⇒ {label} branch ignored.")
                else:
                    print(f"[PIPE] Shape {label} detected → launching {robot_main.name} …")
                    try:
                        subprocess.run([sys.executable, str(robot_main), str(ply)], cwd=robot_dir, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"[ERR] {robot_main.name} failed: {e}", file=sys.stderr)
                        
            elif shape_l == "cylindrique":
                robot_main = robot_dir / "CYLINDRICAL" / "cylinder_main.py"
                label = "cylinder"
                if not robot_main.exists():
                    print(f"[ERR] {robot_main.name} not found ⇒ {label} branch ignored.")
                else:
                    print(f"[PIPE] Shape {label} detected → launching {robot_main.name} …")
                    try:
                        subprocess.run([sys.executable, str(robot_main), str(ply)], cwd=robot_dir, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"[ERR] {robot_main.name} failed: {e}", file=sys.stderr)
                        
            elif shape_l == "spherique":
                robot_main = robot_dir / "SPHERICAL2" / "sphere_main.py"
                label = "sphere"
                if not robot_main.exists():
                    print(f"[ERR] {robot_main.name} not found ⇒ {label} branch ignored.")
                else:
                    print(f"[PIPE] Shape {label} detected → launching {robot_main.name} …")
                    try:
                        subprocess.run([sys.executable, str(robot_main), str(ply)], cwd=robot_dir, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"[ERR] {robot_main.name} failed: {e}", file=sys.stderr)

            else:
                print(f"[INFO] Shape {shape_l} not recognized, no robot action.")

except KeyboardInterrupt:
    pass

finally:
    print("[INFO] Arrêt RealSense…")
    pipeline.stop()
    cv2.destroyAllWindows()