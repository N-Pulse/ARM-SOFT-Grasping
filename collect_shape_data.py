"""
collect_shape_data.py
=====================
Interactive dataset collector for YOLOv8 shape classification.

Streams live frames from the RealSense D405 via ObjectIsolator (HSV red
detection), shows a cropped view of the detected object, and saves labelled
crops to a folder ready for `yolo classify train`.

Output structure
----------------
    data/
      train/
        cylinder/   ← press C
        cuboid/     ← press B

Controls
--------
    C          save crop as CYLINDER
    B          save crop as CUBOID
    S / Space  skip (don't save this frame)
    Q / ESC    quit

Usage
-----
    python collect_shape_data.py
    python collect_shape_data.py --data-dir ./data --pad 20

Workflow
--------
1. Run this script and place objects in front of the camera one at a time.
2. Collect ~80–100 images per class.
3. Train (YOLO splits train/val automatically with split=0.8):
       yolo classify train \\
           model=yolov8n-cls.pt \\
           data=./data/train \\
           epochs=50 \\
           imgsz=128 \\
           batch=16 \\
           fraction=0.8 \\
           name=shape
4. Best weights end up at runs/classify/shape/weights/best.pt
5. Pass that path to ShapeClassifier in ObjectIsolator.
"""

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capture.object_isolation import ObjectIsolator

# ── Defaults ───────────────────────────────────────────────────────────────────
_DATA_DIR = "./data"
_PAD      = 20      # pixels to pad around the bounding box crop
_MIN_AREA = 30 * 30 # minimum crop area to bother saving (pixels²)


def _ensure_dirs(data_dir: str):
    for cls in ("cylinder", "cuboid"):
        os.makedirs(os.path.join(data_dir, "train", cls), exist_ok=True)


def _counts(data_dir: str) -> dict:
    out = {}
    for cls in ("cylinder", "cuboid"):
        d = os.path.join(data_dir, "train", cls)
        out[cls] = len([f for f in os.listdir(d) if f.endswith(".jpg")]) \
                   if os.path.isdir(d) else 0
    return out


def _save(crop, data_dir: str, cls: str) -> str:
    folder = os.path.join(data_dir, "train", cls)
    path   = os.path.join(folder, f"{int(time.monotonic() * 1000)}.jpg")
    cv2.imwrite(path, crop)
    return path


def run(data_dir: str = _DATA_DIR, pad: int = _PAD):
    _ensure_dirs(data_dir)

    isolator = ObjectIsolator(min_points=50)
    isolator.start()
    print("[collect]  Waiting for camera …")
    isolator.ready.wait()
    print("[collect]  Camera ready.")
    print("Controls:  C = cylinder   B = cuboid   S/Space = skip   Q/ESC = quit\n")

    WIN_FULL = "Full frame"
    WIN_CROP = "Crop to save  [C / B]"
    cv2.namedWindow(WIN_FULL, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_CROP, cv2.WINDOW_NORMAL)

    last_save = 0.0

    try:
        while True:
            frame = isolator.get_full_frame()
            crop  = None

            if frame is not None:
                _, _, preview_bgr = frame
                preview = preview_bgr.copy()

                # Crop from the RAW frame (no overlays, no green mask)
                box     = isolator.last_box   # [x1, y1, x2, y2] or None
                raw_bgr = isolator.last_bgr   # original camera pixels
                if box is not None and raw_bgr is not None:
                    h_img, w_img = raw_bgr.shape[:2]
                    x1 = max(0,     int(box[0]) - pad)
                    y1 = max(0,     int(box[1]) - pad)
                    x2 = min(w_img, int(box[2]) + pad)
                    y2 = min(h_img, int(box[3]) + pad)
                    if (x2 - x1) * (y2 - y1) >= _MIN_AREA:
                        crop = raw_bgr[y1:y2, x1:x2].copy()

                # Overlay count on the preview (annotated view for the operator)
                cnts = _counts(data_dir)
                label = (f"cylinder={cnts['cylinder']}  cuboid={cnts['cuboid']}  "
                         f"| C=cylinder  B=cuboid  Q=quit")
                cv2.putText(preview, label, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow(WIN_FULL, preview)

            if crop is not None:
                cv2.imshow(WIN_CROP, crop)

            key = cv2.waitKey(30) & 0xFF
            now = time.monotonic()

            if key in (ord('q'), 27):
                break
            elif key in (ord('c'), ord('b')):
                if crop is None:
                    print("[collect]  No object detected — nothing to save")
                elif now - last_save < 0.2:
                    pass  # debounce rapid key presses
                else:
                    cls  = "cylinder" if key == ord('c') else "cuboid"
                    path = _save(crop, data_dir, cls)
                    last_save = now
                    cnts = _counts(data_dir)
                    print(f"[collect]  saved {cls}  "
                          f"(cylinder={cnts['cylinder']}  cuboid={cnts['cuboid']})  "
                          f"→ {path}")
            # S or Space = skip silently

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        isolator.stop()
        cnts = _counts(data_dir)
        print(f"\n[collect]  Done.  cylinder={cnts['cylinder']}  cuboid={cnts['cuboid']}")
        print(f"\nNext step — train:\n"
              f"  yolo classify train \\\n"
              f"      model=yolov8n-cls.pt \\\n"
              f"      data={os.path.abspath(os.path.join(data_dir, 'train'))} \\\n"
              f"      epochs=50 imgsz=128 batch=16 fraction=0.8 name=shape")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=_DATA_DIR)
    p.add_argument("--pad", type=int, default=_PAD)
    a = p.parse_args()
    run(data_dir=a.data_dir, pad=a.pad)
