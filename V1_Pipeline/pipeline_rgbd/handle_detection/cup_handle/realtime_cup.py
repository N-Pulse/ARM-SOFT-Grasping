import sys
import time
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Real-time cup-handle detection")
    parser.add_argument("--model", type=Path,
                        default=Path("runs/detect/train4/weights/best.pt"),
                        help="path to the .pt model")
    parser.add_argument("--cam-id", type=int, default=0,
                        help="camera ID (0, 1, ...)")
    parser.add_argument("--device", type=str, default="mps",
                        choices=["cpu", "mps", "cuda"],
                        help="device for inference")
    parser.add_argument("--conf-thres", type=float, default=0.5,
                        help="confidence threshold [0.0–1.0]")
    parser.add_argument("--img-size", type=int, default=640,
                        help="image resolution (square)")
    parser.add_argument("--show-stats", action="store_true",
                        help="show total number of frames at the end")
    return parser.parse_args()


def main():
    args = parse_args()

    # Model check
    if not args.model.exists():
        sys.exit(f"❌  Model not found: {args.model}")

    print(f"[INFO] Loading model from: {args.model}")
    model = YOLO(str(args.model))

    print(f"[INFO] Opening camera ID {args.cam_id}…")
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        sys.exit(f"❌  Unable to open camera ID {args.cam_id}")

    window_name = "Cup Handle Detector (q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    t_start = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame not read, stopping.")
                break

            # Inference timing
            t0 = time.time()
            results = model.predict(
                source=frame,
                device=args.device,
                imgsz=args.img_size,
                conf=args.conf_thres,
                verbose=False
            )
            inf_time = (time.time() - t0) * 1000  # ms

            # Draw boxes
            annotated = results[0].plot()

            # Overlay text: FPS & inference time
            frame_count += 1
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            text = f"FPS: {fps:.1f} | Inf: {inf_time:.1f} ms | Conf>={args.conf_thres}"
            cv2.putText(annotated, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Get detected classes
            classes = results[0].boxes.cls
            if len(classes):
                cls_names = [model.names[int(c)] for c in classes]
                unique = set(cls_names)
                stats = " | ".join(f"{u}:{cls_names.count(u)}" for u in unique)
                cv2.putText(annotated, stats, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(window_name, annotated)

            # Quit with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        total_time = time.time() - t_start
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"[INFO] Total frames : {frame_count}")
        print(f"[INFO] Elapsed time : {total_time:.1f}s")
        print(f"[INFO] Average FPS  : {avg_fps:.1f}")

        if args.show_stats:
            print("=== DETECTION FINISHED ===")
            print(f"Frames processed : {frame_count}")
            print(f"Total duration   : {total_time:.1f} s")
            print(f"Average FPS      : {avg_fps:.1f}")

if __name__ == "__main__":
    main()
