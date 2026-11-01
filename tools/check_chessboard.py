"""
tools/check_chessboard.py
=========================
Why is `--calibrate` not finding the chessboard?

Grabs camera frames, tries the board size the calibration expects (10 x 7
inner corners) and the other common sizes, checks that the corners have
depth, and saves the image it looked at as `chessboard_check.png`.

    python tools/check_chessboard.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from armsoft.core.table_plane import (BOARD_COLS, BOARD_ROWS,   # noqa: E402
                                     detect_table_plane)
from armsoft.sources import create_frame_source              # noqa: E402

SIZES = [(BOARD_COLS, BOARD_ROWS)] + [
    (c, r) for c in range(15, 3, -1) for r in range(c, 3, -1)
    if (c, r) != (BOARD_COLS, BOARD_ROWS)]
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def main() -> int:
    source = create_frame_source("camera")
    source.open()
    try:
        for _ in range(15):            # let auto-exposure settle
            frame = source.read()
        # Same camera, same moment: does the real calibration routine see it?
        print("[check] running the real --calibrate routine on the live camera:")
        normal, _ = detect_table_plane(source, max_frames=60)
    finally:
        source.close()

    gray = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("chessboard_raw.png", frame.color_bgr)
    out = frame.color_bgr.copy()
    ok_default, _ = cv2.findChessboardCorners(gray, (BOARD_COLS, BOARD_ROWS), None)
    ok_flags, _ = cv2.findChessboardCorners(gray, (BOARD_COLS, BOARD_ROWS), FLAGS)
    print(f"[check] on one frame: default flags (as --calibrate) -> {ok_default}, "
          f"adaptive+normalize flags -> {ok_flags}")
    print(f"[check] calibration expects {BOARD_COLS} x {BOARD_ROWS} inner corners "
          f"(= {BOARD_COLS + 1} x {BOARD_ROWS + 1} squares)")

    found_any = False
    for cols, rows in SIZES:
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), FLAGS)
        if not ok:
            continue
        found_any = True
        uv = corners.reshape(-1, 2).round().astype(int)
        z = frame.depth_m[uv[:, 1].clip(0, gray.shape[0] - 1),
                          uv[:, 0].clip(0, gray.shape[1] - 1)]
        n_depth = int((z > 0).sum())
        tag = "  <- the size --calibrate needs" if (cols, rows) == (BOARD_COLS, BOARD_ROWS) else ""
        print(f"[check] FOUND {cols} x {rows} inner corners "
              f"({cols + 1} x {rows + 1} squares), {n_depth}/{len(uv)} corners with depth{tag}")
        cv2.drawChessboardCorners(out, (cols, rows), corners, ok)
        break

    if not found_any:
        print("[check] no chessboard of any size found in the image")
        print("        -> whole board in view? flat? no glare? not hidden by the cube?")
    elif (cols, rows) != (BOARD_COLS, BOARD_ROWS):
        print(f"[check] your board is {cols} x {rows}, but --calibrate looks for "
              f"{BOARD_COLS} x {BOARD_ROWS}: print a {BOARD_COLS + 1} x {BOARD_ROWS + 1}-square "
              f"board, or tell Claude to make the size configurable")
    elif n_depth < 10:
        print("[check] the board is seen but its corners have no depth -> move closer, "
              "improve lighting, avoid glossy paper")
    else:
        print("[check] the board is fine: --calibrate should work with this view")

    cv2.imwrite("chessboard_check.png", out)
    print("[check] image saved to chessboard_check.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
