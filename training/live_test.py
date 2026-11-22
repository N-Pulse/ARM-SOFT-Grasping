"""
training/live_test.py
=====================
Live detection demo on the Intel RealSense D405 (or an image / video file).
Each box shows class, confidence and the median depth inside it, so the
30 / 50 / 70 cm working distances can be checked on screen.

    # 1. COCO-pretrained YOLO26n (no ball/can/box classes in COCO except
    #    "sports ball", "bottle", "cup"; useful as a colour-agnostic baseline)
    python training/live_test.py --model yolo26n.pt --classes "sports ball,bottle,cup"
    # 2. our fine-tuned model
    python training/live_test.py --model training/runs/detect/iter1/weights/best.pt
    # without a camera
    python training/live_test.py --source some_photo.jpg

Keys:  s = save snapshot to training/results/live/   q / Esc = quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.paths import DEFAULT_MODEL, RESULTS_DIR  # noqa: E402


def box_depth_m(depth_m: np.ndarray, box, inner: float = 0.5):
    """Median of the valid (> 0) depths in the central `inner` part of `box`."""
    h, w = depth_m.shape
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) * inner / 2, (y2 - y1) * inner / 2
    xa, xb = int(max(0, cx - hw)), int(min(w, cx + hw + 1))
    ya, yb = int(max(0, cy - hh)), int(min(h, cy + hh + 1))
    patch = depth_m[ya:yb, xa:xb]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else None


def class_ids(model_names: dict, wanted):
    if not wanted:
        return None
    by_name = {v: k for k, v in model_names.items()}
    unknown = [n for n in wanted if n not in by_name]
    if unknown:
        raise ValueError(f"unknown classes {unknown}; model has {sorted(by_name)}")
    return [by_name[n] for n in wanted]


def realsense_frames(width=640, height=480, fps=30):
    """Yield (bgr, depth_m) from the D405, depth aligned to colour."""
    import pyrealsense2 as rs
    if len(rs.context().query_devices()) == 0:
        sys.exit("no RealSense device found (USB 3 port? another app holding it?)")
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipe.start(cfg)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    try:
        while True:
            fs = align.process(pipe.wait_for_frames())
            df, cf = fs.get_depth_frame(), fs.get_color_frame()
            if df and cf:
                yield (np.asanyarray(cf.get_data()).copy(),
                       np.asanyarray(df.get_data()).astype(np.float32) * scale)
    finally:
        pipe.stop()


def file_frames(path: str):
    """Yield (bgr, None) from an image or a video file."""
    img = cv2.imread(path)
    if img is not None:
        while True:
            yield img, None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"cannot open {path}")
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame, None


def annotate(result, depth_m) -> np.ndarray:
    img = result.plot()
    if depth_m is not None:
        for box in result.boxes.xyxy.cpu().numpy():
            z = box_depth_m(depth_m, box)
            if z is not None:
                cv2.putText(img, f"{z * 100:.0f} cm", (int(box[0]), int(box[3]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--source", default="realsense", help="'realsense' or an image/video path")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--classes", help="comma-separated class names to keep (default: all)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = run until q / end of file)")
    args = ap.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(args.model)
    keep = class_ids(model.names, [c.strip() for c in args.classes.split(",")]
                     if args.classes else None)
    tag = Path(args.model).stem if args.model.endswith("yolo26n.pt") \
        else Path(args.model).resolve().parent.parent.name
    frames = realsense_frames() if args.source == "realsense" else file_frames(args.source)
    snap_dir = RESULTS_DIR / "live"

    n = 0
    t_prev = time.perf_counter()
    for bgr, depth in frames:
        res = model.predict(bgr, conf=args.conf, imgsz=args.imgsz,
                            device=args.device, classes=keep, verbose=False)[0]
        img = annotate(res, depth)
        now = time.perf_counter()
        cv2.putText(img, f"{tag}  {1 / max(now - t_prev, 1e-6):.1f} FPS  "
                         f"{len(res.boxes)} obj", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        t_prev = now
        cv2.imshow("live_test", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            snap_dir.mkdir(parents=True, exist_ok=True)
            path = snap_dir / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(path), img)
            print(f"saved {path}")
        elif key in (ord("q"), 27):
            break
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
    cv2.destroyAllWindows()
    print(f"{n} frames processed")


if __name__ == "__main__":
    main()
