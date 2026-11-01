"""
tools/panels.py
===============
Shared drawing code for the visual-check tools.

Builds the four review panels — camera view, depth, fitted shape + grasp, and
the top-down plan view — plus the caption bar underneath. Used by
``save_sim_results.py`` (stills) and ``make_sim_video.py`` (animation).
"""

from __future__ import annotations

import cv2
import numpy as np

from armsoft.core.grasp_geometry import GRIPPER_LINES, wireframe_vertices

FONT = cv2.FONT_HERSHEY_SIMPLEX

GREEN, ORANGE, CYAN, RED, WHITE = ((0, 255, 0), (0, 140, 255), (255, 220, 0),
                                   (60, 60, 255), (255, 255, 255))

def project(pts_3d: np.ndarray, K) -> np.ndarray:
    """Camera-frame points (N, 3) → pixel coordinates (N, 2) int."""
    pts = np.asarray(pts_3d, float).reshape(-1, 3)
    z = np.clip(pts[:, 2], 1e-6, None)
    u = pts[:, 0] * K.fx / z + K.cx
    v = pts[:, 1] * K.fy / z + K.cy
    return np.stack([u, v], axis=1).astype(np.int32)


def label(img, text, org, color=WHITE, scale=0.5, thick=1):
    """
    Text on a darkened strip, so it stays readable over any background.

    Note: drawing a thicker outline pass behind the text does NOT work — in
    OpenCV a larger stroke thickness also widens the glyph advance, so the
    outline drifts right of the fill and its tail shows as a ghost.
    """
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    x0, y0 = max(x - 4, 0), max(y - th - 4, 0)
    x1, y1 = min(x + tw + 4, img.shape[1]), min(y + base + 2, img.shape[0])
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = (roi * 0.35).astype(np.uint8)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def panel_camera(frame, iso, pred, conf, accepted):
    """Colour image + red-object mask, bounding box and classifier verdict."""
    img = frame.color_bgr.copy()
    if iso.mask is not None:
        overlay = np.zeros_like(img)
        overlay[iso.mask] = GREEN
        img = cv2.addWeighted(img, 0.75, overlay, 0.25, 0)
    if iso.box is not None:
        x1, y1, x2, y2 = iso.box
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
        col = GREEN if accepted else (80, 180, 255)
        txt = pred or "?"
        if conf is not None:
            txt += f"  {conf:.2f}"
        if not accepted:
            txt += "  REJECTED"
        label(img, txt, (x1 - 20, max(y1 - 8, 14)), col, 0.6, 2)
    label(img, "1. camera view + classifier", (10, 22), WHITE, 0.55, 1)
    label(img, f"{len(iso.object_points)} object points", (10, 44), WHITE, 0.45, 1)
    return img


def panel_depth(frame):
    """Depth image as a colour map; black where the sensor returned nothing."""
    d = frame.depth_m
    valid = d > 0
    img = np.zeros((*d.shape, 3), np.uint8)
    if valid.any():
        lo, hi = float(d[valid].min()), float(d[valid].max())
        norm = np.zeros_like(d)
        norm[valid] = (d[valid] - lo) / max(hi - lo, 1e-6)
        img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[~valid] = 0
        label(img, f"{lo:.2f} - {hi:.2f} m", (10, 44), WHITE, 0.45, 1)
    label(img, "2. depth", (10, 22), WHITE, 0.55, 1)
    return img


def panel_fit(frame, pipeline, result):
    """Fitted wireframe and gripper skeleton, projected back onto the image."""
    img = (frame.color_bgr * 0.45).astype(np.uint8)
    K = frame.intrinsics

    ls = pipeline.last_shape_ls
    if ls is not None:
        verts = wireframe_vertices(ls)
        px = project(verts, K)
        for a, b in np.asarray(ls.lines):
            cv2.line(img, tuple(px[a]), tuple(px[b]), CYAN, 1, cv2.LINE_AA)

    if result.ok:
        gp = project(np.asarray(result.gripper_points), K)
        for a, b in GRIPPER_LINES:
            cv2.line(img, tuple(gp[a]), tuple(gp[b]), ORANGE, 2, cv2.LINE_AA)
        for i in (4, 5):                       # fingertips
            cv2.circle(img, tuple(gp[i]), 4, ORANGE, -1, cv2.LINE_AA)
        tcp = project(np.asarray([result.position_m]), K)[0]
        cv2.drawMarker(img, tuple(tcp), RED, cv2.MARKER_CROSS, 14, 2)

    label(img, "3. fitted shape + grasp", (10, 22), WHITE, 0.55, 1)
    label(img, "cyan = fit   orange = gripper   red = grasp point",
          (10, 44), WHITE, 0.4, 1)
    return img


def panel_topdown(pipeline, result, table_normal, size=(480, 640)):
    """
    Bird's-eye view: everything projected onto the table plane.

    This is where the grasp is actually readable — in the camera view the
    gripper points almost straight at the lens and collapses to a line.
    """
    h, w = size
    img = np.full((h, w, 3), 18, np.uint8)

    n = np.asarray(table_normal, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    e1 = np.cross(n, ref); e1 /= np.linalg.norm(e1)     # image x
    e2 = np.cross(n, e1);  e2 /= np.linalg.norm(e2)     # image y

    iso = pipeline.last_isolation
    pts = iso.object_points if iso is not None else np.zeros((0, 3))
    ls = pipeline.last_shape_ls
    anchor = (np.asarray(result.position_m) if result.ok
              else (wireframe_vertices(ls).mean(axis=0) if ls is not None
                    else (pts.mean(axis=0) if len(pts) else np.zeros(3))))

    scale = 1800.0          # pixels per metre  (a 20 cm span fills the panel)
    cx, cy = w // 2, h // 2

    def to_px(p):
        d = np.asarray(p, float).reshape(-1, 3) - anchor
        u = cx + (d @ e1) * scale
        v = cy - (d @ e2) * scale
        return np.stack([u, v], axis=1).astype(np.int32)

    # 1 cm grid + a 5 cm scale bar
    step = int(0.01 * scale)
    for gx in range(cx % step, w, step):
        cv2.line(img, (gx, 0), (gx, h), (38, 38, 38), 1)
    for gy in range(cy % step, h, step):
        cv2.line(img, (0, gy), (w, gy), (38, 38, 38), 1)
    cv2.line(img, (20, h - 24), (20 + int(0.05 * scale), h - 24), WHITE, 2)
    label(img, "5 cm", (20, h - 32), WHITE, 0.45, 1)

    if len(pts):
        for q in to_px(pts):
            if 0 <= q[0] < w and 0 <= q[1] < h:
                img[q[1], q[0]] = (235, 170, 90)        # measured surface

    if ls is not None:
        px = to_px(wireframe_vertices(ls))
        for a, b in np.asarray(ls.lines):
            cv2.line(img, tuple(px[a]), tuple(px[b]), CYAN, 1, cv2.LINE_AA)

    if result.ok:
        gp = to_px(np.asarray(result.gripper_points))
        for a, b in GRIPPER_LINES:
            cv2.line(img, tuple(gp[a]), tuple(gp[b]), ORANGE, 2, cv2.LINE_AA)
        for i in (4, 5):
            cv2.circle(img, tuple(gp[i]), 5, ORANGE, -1, cv2.LINE_AA)
        cv2.drawMarker(img, tuple(to_px([result.position_m])[0]), RED,
                       cv2.MARKER_CROSS, 16, 2)
        # approach arrow, from the palm towards the object
        cv2.arrowedLine(img, tuple(gp[1]), tuple(to_px([result.position_m])[0]),
                        (120, 255, 255), 1, cv2.LINE_AA, tipLength=0.15)

    label(img, "4. top-down (table plane)", (10, 22), WHITE, 0.55, 1)
    label(img, "blue dots = measured surface   cyan = fit   orange = gripper",
          (10, 44), (235, 190, 140), 0.4, 1)
    return img


def corruption_note(scene):
    """One phrase describing how badly this frame was corrupted."""
    bits = [f"depth noise {scene.get('noise_m', 0.0008) * 1e3:.1f}mm"]
    if scene.get("worms"):
        bits.append(f"{scene['worms']} worms")
    if scene.get("depth_holes"):
        bits.append(f"{scene['depth_holes']} holes")
    return ",  ".join(bits)


def caption(width, name, scene, result, conf):
    """Two lines of text under the panels: ground truth vs. what the model said."""
    bar = np.full((78, width, 3), 28, np.uint8)
    truth = (f"TRUTH  {scene['shape']}  "
             f"d={scene.get('distance_m', 0.35):.2f}m  "
             f"w={scene.get('diameter_m', 0.06) * 1e3:.0f}mm  "
             f"h={scene.get('height_m', 0.10) * 1e3:.0f}mm  "
             f"yaw={scene.get('yaw_deg', 20):.0f}deg"
             f"     [{corruption_note(scene)}]")
    if result.ok:
        hit = result.shape == scene["shape"]
        got = (f"MODEL  {result.shape} ({'correct' if hit else 'WRONG'}"
               f"{'' if conf is None else f', conf={conf:.2f}'})  "
               f"d={result.distance_m:.3f}m  "
               f"w={result.object_width_m * 1e3:.0f}mm  "
               f"h={result.object_height_m * 1e3:.0f}mm  "
               f"jaw={result.jaw_opening_m * 1e3:.0f}mm  "
               f"stable={result.stable}")
        colour = GREEN if hit else (80, 180, 255)
    else:
        got = f"MODEL  {result.status}"
        if result.status == "no_shape" and conf is not None:
            got += f"  (classifier below the 0.70 threshold, conf={conf:.2f})"
        colour = RED
    label(bar, name, (12, 30), WHITE, 0.6, 2)
    label(bar, truth, (300, 26), (190, 190, 190), 0.5, 1)
    label(bar, got, (300, 56), colour, 0.5, 1)
    return bar


