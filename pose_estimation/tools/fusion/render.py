"""Perspective RGB-D and review overlays from the fixed simulated chest camera."""

from dataclasses import replace

import cv2
import numpy as np

from scene import FINGERS, LONG_FINGERS, apply, point_map, transform, unit


WIDTH, HEIGHT = 848, 480
K = np.array([[450.0, 0, 423.5], [0, 450.0, 239.5], [0, 0, 1.0]])
NEAR_M = 0.015
VIEWS = (
    ("palm_front", (0.016, 0.087, 0.379), -6, 0),
    ("palm_oblique", (0.070, 0.075, 0.346), -31, 36),
    ("thumb_side", (0.065, 0.058, 0.385), -43, 77),
    ("back_oblique", (0.021, 0.079, 0.376), -24, 145),
    ("palm_near", (-0.047, 0.100, 0.365), 13, -28),
    ("reach", (0.050, 0.018, 0.361), -48, 0),
)


def rotation(axis, degrees):
    return cv2.Rodrigues(unit(np.asarray(axis, dtype=float)) * np.deg2rad(degrees))[0]


CAMERA_FROM_BODY = transform(rotation([1, 0, 0], 35), [0, 0, 0])


def placements(reference_points):
    p = point_map(reference_points)
    wrist, palm = p["wrist_axle_center"], p["palm_center"]
    along = unit(np.mean([p[digit + "_base"] for digit in LONG_FINGERS], axis=0) - wrist)
    normal = unit(np.cross(p["index_base"] - p["little_base"], along))
    hand_frame = np.stack([unit(np.cross(along, normal)), along, normal], axis=1)
    for name, wrist_position, tilt, twist in VIEWS:
        wrist_position = np.asarray(wrist_position)
        if name == "reach":
            along = unit(np.array([np.sin(np.deg2rad(-32)), -0.10, np.cos(np.deg2rad(-32))]))
            across = -unit(np.cross([0, 1, 0], along))
            arm = rotation(along, -43) @ np.stack([across, along, np.cross(across, along)], axis=1)
            orientation = CAMERA_FROM_BODY[:3, :3] @ arm
        else:
            screen = np.array([np.sin(np.deg2rad(tilt)), -np.cos(np.deg2rad(tilt)), 0])
            palm_camera = wrist_position.copy()
            for _ in range(10):
                to_camera = unit(-palm_camera)
                along = unit(screen - to_camera * (screen @ to_camera))
                face = rotation(along, twist) @ to_camera
                hand = np.stack([unit(np.cross(along, face)), along, face], axis=1)
                orientation = hand @ hand_frame.T
                palm_camera = wrist_position + orientation @ (palm - wrist)
        camera_from_cad = transform(orientation, wrist_position - orientation @ wrist)
        body_from_cad = np.linalg.inv(CAMERA_FROM_BODY) @ camera_from_cad
        angle = np.degrees(np.arccos(np.clip(
            (orientation @ normal) @ unit(-apply(palm, camera_from_cad)), -1, 1
        )))
        yield name, camera_from_cad, body_from_cad, float(angle)


def project(points):
    return (points @ K.T)[:, :2] / points[:, 2:3]


def rasterize(scene, camera_from_cad):
    triangles = apply(scene.triangles, camera_from_cad).astype(np.float32)
    z = triangles[:, :, 2]
    if np.any((z.min(axis=1) < NEAR_M) & (z.max(axis=1) >= NEAR_M)):
        raise ValueError("A mesh triangle crosses the camera near plane")
    valid = z.min(axis=1) >= NEAR_M
    triangles = triangles[valid]
    inverse_z = 1.0 / z[valid]
    colors, body_ids = scene.colors[valid], scene.body_ids[valid]
    x = np.float32(K[0, 2]) + np.float32(K[0, 0]) * triangles[:, :, 0] * inverse_z
    y = np.float32(K[1, 2]) + np.float32(K[1, 1]) * triangles[:, :, 1] * inverse_z
    left = np.maximum(0, np.ceil(x.min(axis=1))).astype(int)
    right = np.minimum(WIDTH - 1, np.floor(x.max(axis=1))).astype(int)
    top = np.maximum(0, np.ceil(y.min(axis=1))).astype(int)
    bottom = np.minimum(HEIGHT - 1, np.floor(y.max(axis=1))).astype(int)
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal_length = np.linalg.norm(normal, axis=1)
    normal[np.sum(normal * triangles[:, 0], axis=1) > 0] *= -1
    dx12, dx02 = x[:, 2] - x[:, 1], x[:, 0] - x[:, 2]
    dy12, dy02 = y[:, 1] - y[:, 2], y[:, 0] - y[:, 2]
    area = dy12 * dx02 + dx12 * dy02
    draw = np.flatnonzero((left <= right) & (top <= bottom) & (np.abs(area) >= 1e-5) & (normal_length > 0))
    nx, ny, nz = normal[draw].T
    light = (-np.float32(0.35) * nx - np.float32(0.47) * ny - np.float32(0.81) * nz) / normal_length[draw]
    lit = np.float32(0.45) + np.float32(0.55) * np.maximum(0, light)
    shaded = (colors[draw] * lit[:, None]).astype(np.uint8)
    rgb = np.full((HEIGHT, WIDTH, 3), [56, 64, 73], dtype=np.uint8)
    depth = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float32)
    surface = np.full((HEIGHT, WIDTH), -1, dtype=np.int32)
    x_pixels, y_pixels = np.arange(WIDTH, dtype=np.float32), np.arange(HEIGHT, dtype=np.float32)
    for i, color in zip(draw, shaded):
        rows, cols = slice(top[i], bottom[i] + 1), slice(left[i], right[i] + 1)
        px, py = x_pixels[cols] - x[i, 2], y_pixels[rows, None] - y[i, 2]
        u = (dy12[i] * px + dx12[i] * py) / area[i]
        v = (-dy02[i] * px + dx02[i] * py) / area[i]
        w = 1 - u - v
        covered = (u >= -1e-5) & (v >= -1e-5) & (w >= -1e-5)
        z0, z1, z2 = inverse_z[i]
        depths = np.full_like(u, np.inf)
        np.divide(1, u * z0 + v * z1 + w * z2, out=depths, where=covered)
        closer = depths < depth[rows, cols]
        depth[rows, cols][closer] = depths[closer]
        rgb[rows, cols][closer] = color
        surface[rows, cols][closer] = body_ids[i]

    depth[~np.isfinite(depth)] = 0
    millimeters = np.rint(depth * 1000)
    if millimeters.max() > 65535:
        raise ValueError("Depth exceeds 16-bit millimeter range")
    return rgb, millimeters.astype(np.uint16), surface


COLORS = {
    "wrist": (230, 235, 240), "palm": (230, 235, 240), "thumb": (255, 151, 83),
    "index": (69, 211, 251), "middle": (87, 225, 160),
    "ring": (245, 207, 75), "little": (211, 148, 255),
}


def draw_overlay(rgb, pixels, points, edges):
    overlay = rgb.copy()
    for start, stop in edges:
        a, b = tuple(pixels[start]), tuple(pixels[stop])
        cv2.line(overlay, a, b, (32, 35, 41), 4, cv2.LINE_AA)
        cv2.line(overlay, a, b, COLORS[points[stop]["group"]], 2, cv2.LINE_AA)
    for pixel, point in zip(pixels, points):
        cv2.circle(overlay, tuple(pixel), 4, (32, 35, 41), -1, cv2.LINE_AA)
        thickness = -1 if point["type"] == "hinge" else 1
        cv2.circle(overlay, tuple(pixel), 3, COLORS[point["group"]], thickness, cv2.LINE_AA)
    return overlay


def finger_side_views(scene, points):
    """Render each long finger along its hinge axis, with the palm out of the way."""
    by_name = point_map(points)
    across = by_name["index_base"] - by_name["little_base"]
    images = []
    for finger in LONG_FINGERS:
        prefix = FINGERS[finger] + "+"
        chain = [point for point in points if point["group"] == finger]
        cad = np.array([point["world_m"] for point in chain])
        axis = unit(np.asarray(scene.joint("Revolute 5", prefix, first_only=True)["axis"]))
        if axis @ across < 0:
            axis *= -1
        z = -axis
        x = cad[1] - cad[0]
        x = unit(x - z * (z @ x))
        orientation = np.stack((x, np.cross(z, x), z))
        camera = transform(orientation, [0, 0, 0.18] - orientation @ cad.mean(0))
        bodies = np.array([path.startswith(prefix) for path in scene.paths])
        mask = bodies[scene.body_ids]
        finger_scene = replace(scene, triangles=scene.triangles[mask],
                               colors=scene.colors[mask], body_ids=scene.body_ids[mask])
        rgb, _, _ = rasterize(finger_scene, camera)
        pixels = np.rint(project(apply(cad, camera))).astype(np.int32)
        overlay = draw_overlay(rgb, pixels, chain, [(0, 1), (1, 2), (2, 3)])
        # Same crop for every finger and pose, to aid comparison between rows.
        images.append(overlay[132:348, 244:604])
    return images
