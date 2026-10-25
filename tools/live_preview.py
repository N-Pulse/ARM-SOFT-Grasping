"""
tools/live_preview.py
=====================
Live 2-D view of the RealSense camera, for positioning it before a test run.

Shows the model's annotated view (red mask, bounding box, label) next to a
colour-mapped depth image, with the depth at the image centre and the latest
grasp summary overlaid. Press `q` or Esc to quit, `s` to save a snapshot.

    python tools/live_preview.py
    python tools/live_preview.py --table-normal 0,-1,0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armsoft.core.classifier import build_classifier       # noqa: E402
from armsoft.core.pipeline import GraspPipeline            # noqa: E402
from armsoft.sources import create_frame_source            # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "models", "shape_classifier.pt")


def depth_view(depth_m: np.ndarray, max_m: float) -> np.ndarray:
    """Depth in metres → colour image; black where there is no reading."""
    d = np.clip(depth_m / max_m, 0, 1)
    img = cv2.applyColorMap((255 * (1 - d)).astype(np.uint8), cv2.COLORMAP_JET)
    img[depth_m <= 0] = 0
    return img


def text(img, s, y, color=(255, 255, 255)):
    cv2.putText(img, s, (8, y), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, (8, y), FONT, 0.5, color, 1, cv2.LINE_AA)


def main() -> int:
    p = argparse.ArgumentParser(description="Live camera preview.")
    p.add_argument("--source", default="camera", choices=["camera", "sim", "replay"])
    p.add_argument("--replay-path", default=None)
    p.add_argument("--table-normal", default="0,-1,0")
    p.add_argument("--classifier", default="auto",
                   choices=["auto", "yolo", "geometric", "fixed"])
    p.add_argument("--depth-max", type=float, default=1.0,
                   help="depth (m) mapped to the far end of the colour scale")
    args = p.parse_args()

    normal = np.array([float(x) for x in args.table_normal.split(",")])
    pipeline = GraspPipeline(
        table_normal=normal,
        classifier=build_classifier(args.classifier, model_path=os.path.abspath(WEIGHTS)))

    source = create_frame_source(args.source, path=args.replay_path)
    source.open()
    win = "ARM-SOFT live preview  (q = quit, s = snapshot)"
    try:
        for frame in source.frames(limit=None):
            r = pipeline.process(frame)
            iso = pipeline.last_isolation
            left = (iso.preview_bgr if iso is not None and iso.preview_bgr is not None
                    else frame.color_bgr).copy()
            right = depth_view(frame.depth_m, args.depth_max)

            # Depth at the image centre, as a quick tape-measure cross-check.
            h, w = frame.depth_m.shape
            patch = frame.depth_m[h // 2 - 3:h // 2 + 4, w // 2 - 3:w // 2 + 4]
            valid = patch[patch > 0]
            centre = f"{np.median(valid):.3f} m" if valid.size else "no reading"
            for img in (left, right):
                cv2.drawMarker(img, (w // 2, h // 2), (255, 255, 255),
                               cv2.MARKER_CROSS, 16, 1)
            text(right, f"centre depth: {centre}", 20)

            if r.ok:
                text(left, f"{r.shape}  d={r.distance_m:.3f}m  "
                           f"w={r.object_width_m * 1e3:.0f}mm  "
                           f"h={r.object_height_m * 1e3:.0f}mm", h - 30, (0, 255, 0))
                text(left, f"stable={r.stable}  {r.timing_ms.get('total', 0):.0f}ms",
                     h - 10, (0, 255, 0) if r.stable else (0, 200, 255))
            else:
                text(left, r.status, h - 10, (60, 60, 255))

            view = np.hstack([left, right])
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                name = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(name, view)
                print(f"[preview] saved {name}")
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
