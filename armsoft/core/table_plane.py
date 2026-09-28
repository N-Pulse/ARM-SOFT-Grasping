"""
armsoft/table_plane.py
=======================
Chessboard table-plane calibration, driven by a :class:`FrameSource` so that
it works with a live camera, a recording or the simulator alike.

Returns ``(normal, d)`` of the plane ``normal · X + d = 0`` in camera
coordinates, with ``normal`` pointing *towards the camera* (i.e. "up" from the
table).  Falls back to the caller-supplied default if the board is not seen
within ``max_frames``.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..sources.base import FrameSource

BOARD_COLS = 10
BOARD_ROWS = 7


def detect_table_plane(source: FrameSource,
                       board_cols: int = BOARD_COLS,
                       board_rows: int = BOARD_ROWS,
                       max_frames: int = 150,
                       default=(0.0, -1.0, 0.0),
                       verbose: bool = True):
    """Search ``max_frames`` frames for the chessboard and fit a plane by SVD."""
    hint = getattr(source, "table_normal_hint", None)
    if hint is not None:
        n = np.asarray(hint, float)
        n = n / (np.linalg.norm(n) + 1e-12)
        if verbose:
            print(f"[table] source provides a known table normal {np.round(n, 3)}")
        return n, None

    board = (board_cols, board_rows)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for _ in range(max_frames):
        frame = source.read()
        if frame is None:
            break
        gray = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board, None)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        K = frame.intrinsics
        pts = []
        for (u, v) in corners.reshape(-1, 2):
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < K.width and 0 <= vi < K.height):
                continue
            z = float(frame.depth_m[vi, ui])
            if z <= 0:
                continue
            pts.append([(u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z])
        pts = np.asarray(pts)
        if len(pts) < 10:
            continue

        centroid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centroid)
        normal = vh[-1]
        if normal[2] > 0:            # point it back towards the camera
            normal = -normal
        d = -float(normal @ centroid)
        if verbose:
            print(f"[table] plane found — normal {np.round(normal, 3)}, d={d:+.3f}")
        return normal, d

    n = np.asarray(default, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    if verbose:
        print(f"[table] no chessboard seen — using default normal {np.round(n, 3)}")
    return n, None
