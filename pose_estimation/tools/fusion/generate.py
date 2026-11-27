"""Turn solved Fusion snapshots into a reviewed-batch candidate."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from render import (
    CAMERA_FROM_BODY, HEIGHT, K, VIEWS, WIDTH, draw_overlay, finger_side_views, placements, project, rasterize,
)
from scene import LONG_FINGERS, apply, keypoints, load_scene, point_map, read_json, signed_turns


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def write_png(path, image):
    # Encoding separately supports non-ASCII paths on Windows.
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Could not encode {path}")
    path.write_bytes(encoded.tobytes())


def read_png(path, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), flag)


def prepare_frames(group, scene, points, edges, views):
    cad = np.array([p["world_m"] for p in points])
    frames = []
    for view, camera_from_cad, body_from_cad, angle in views:
        xyz = apply(cad, camera_from_cad)
        require(np.all(xyz[:, 2] > 0), f"{view} has keypoints behind the camera")
        uv = project(xyz)
        pixels = np.rint(uv).astype(np.int32)
        require(np.all((pixels >= 0) & (pixels < [WIDTH, HEIGHT])), f"{view} clips a keypoint")
        rgb, depth, surface = rasterize(scene, camera_from_cad)
        require(depth.min() == 0 and depth.max() > 0, f"{view} has an invalid depth image")
        records = []
        for point, xyz_i, uv_i, (u, v) in zip(points, xyz, uv, pixels):
            z = int(depth[v, u])
            body = surface[v, u]
            records.append({
                "id": point["name"], "xyz_camera_m": xyz_i.tolist(), "uv_px": uv_i.tolist(),
                "in_frame": True, "surface_depth_valid": z > 0, "surface_z_mm": z or None,
                "front_surface": scene.paths[body] if body >= 0 else None,
                "observability": "unreviewed",
            })
        annotation = {
            "pose_group": group, "view": view, "palm_to_camera_deg": angle, "K": K.tolist(),
            "T_camera_from_cad_m": camera_from_cad.tolist(),
            "T_body_from_cad_m": body_from_cad.tolist(),
            "T_camera_from_body_m": CAMERA_FROM_BODY.tolist(), "keypoints": records,
        }
        frames.append((annotation, rgb, depth, draw_overlay(rgb, pixels, points, edges)))
    return frames


def validate_files(output):
    manifest = read_json(output / "manifest.json")
    names = [p["name"] for p in manifest["keypoints"]]
    require(len(names) == len(set(names)) == 21, "Expected 21 unique CAD points")
    max_error, missing_depth = 0.0, []
    groups = {group: set() for group in manifest["pose_keypoints_cad_m"]}
    reference_views = {}
    for frame_id in manifest["frames"]:
        annotation = read_json(output / "labels" / f"{frame_id}.json")
        group, view = annotation["pose_group"], annotation["view"]
        require(view not in groups[group], f"Duplicate {group} / {view}")
        groups[group].add(view)
        points = annotation["keypoints"]
        require([p["id"] for p in points] == names, f"Frame {frame_id} changed point order")
        camera_from_cad = np.array(annotation["T_camera_from_cad_m"])
        camera_from_body = np.array(annotation["T_camera_from_body_m"])
        body_from_cad = np.array(annotation["T_body_from_cad_m"])
        rotation = camera_from_cad[:3, :3]
        require(np.allclose(camera_from_body, CAMERA_FROM_BODY)
                and np.allclose(body_from_cad, reference_views.setdefault(view, body_from_cad))
                and np.allclose(camera_from_cad, camera_from_body @ body_from_cad)
                and np.allclose(rotation.T @ rotation, np.eye(3))
                and np.isclose(np.linalg.det(rotation), 1)
                and np.array_equal(annotation["K"], K), f"Frame {frame_id} has invalid camera geometry")
        xyz = apply(np.array(manifest["pose_keypoints_cad_m"][group]), camera_from_cad)
        require(np.all(xyz[:, 2] > 0) and np.allclose(xyz, [p["xyz_camera_m"] for p in points], atol=1e-9),
                f"Frame {frame_id} has invalid camera coordinates")
        uv = project(xyz)
        error = float(np.linalg.norm(uv - np.array([p["uv_px"] for p in points]), axis=1).max())
        max_error = max(max_error, error)
        require(error < 1e-6, f"Frame {frame_id} failed reprojection")
        rgb = read_png(output / annotation["rgb"])
        overlay = read_png(output / annotation["overlay"])
        depth = read_png(output / annotation["depth"], cv2.IMREAD_UNCHANGED)
        require(rgb.shape == overlay.shape == (HEIGHT, WIDTH, 3) and np.any(rgb != overlay),
                f"Frame {frame_id} has invalid RGB or overlay")
        require(depth.shape == (HEIGHT, WIDTH) and depth.dtype == np.uint16
                and depth.min() == 0 and depth.max() > 0, f"Frame {frame_id} has invalid depth")
        for (u, v), point in zip(np.rint(uv).astype(int), points):
            require(0 <= u < WIDTH and 0 <= v < HEIGHT and point["in_frame"],
                    f"Frame {frame_id} has an off-image keypoint")
            z = int(depth[v, u])
            require((z or None) == point["surface_z_mm"] and bool(z) == point["surface_depth_valid"],
                    f"Frame {frame_id} has inconsistent surface depth")
            if not z:
                missing_depth.append([frame_id, point["id"]])
    require(all(views == {view[0] for view in VIEWS} for views in groups.values()),
            "One or more pose groups lack a placement")
    return {"frames": len(manifest["frames"]), "points": 21 * len(manifest["frames"]),
            "max_reprojection_error_px": max_error, "points_without_surface_depth": missing_depth}


def make_gallery(rows, path, image_size):
    columns = len(rows[0])
    width, image_height = image_size
    height = image_height + 27
    sheet = np.full((height * len(rows), width * columns, 3), 30, dtype=np.uint8)
    for i, row in enumerate(rows):
        for j, (title, overlay) in enumerate(row):
            cell = sheet[i * height:(i + 1) * height, j * width:(j + 1) * width]
            cell[:image_height] = cv2.resize(overlay, image_size, interpolation=cv2.INTER_AREA)
            cv2.putText(cell, title, (5, image_height + 17), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (240, 240, 240), 1, cv2.LINE_AA)
    write_png(path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def generate(source, output):
    report = read_json(source / "export_report.json")
    requested = report["poses"]
    reference = load_scene(source / "as_saved")
    reference_points, edges = keypoints(reference)
    views = list(placements(reference_points))
    output.mkdir(parents=True)
    for folder in ("rgb", "depth_mm", "overlay", "labels"):
        (output / folder).mkdir()
    manifest = {
        "source": "Assembly_hand_old_motors", "version": 22, "fusion_build": report["fusion_build"],
        "resolution": [WIDTH, HEIGHT], "K": K.tolist(),
        "camera": "Fixed ideal chest pinhole camera, pitched down 35 degrees; no sensor calibration or noise.",
        "axes": "Camera x right, y down, z forward; coordinates in meters.",
        "depth": "Aligned 16-bit PNG. Optical-axis Z in millimeters, 0 for no surface.",
        "ready_for_training": False, "keypoints": reference_points,
        "edges": [[reference_points[a]["name"], reference_points[b]["name"]] for a, b in edges],
        "edges_pending_hardware_review": [["thumb_distal", "thumb_tip"]],
        "pose_keypoints_cad_m": {}, "frames": [],
    }
    audits, accepted, rejected = {}, {}, {}
    gallery, side_gallery = [], []
    for name, exported in requested.items():
        try:
            if "error" in exported:
                raise ValueError(exported["error"])
            scene = reference if name == "as_saved" else load_scene(source / name)
            require(scene.body_samples.keys() == reference.body_samples.keys(), "Visible CAD bodies differ from reference")
            points = reference_points if name == "as_saved" else keypoints(scene, reference, point_map(reference_points))[0]
            turns = signed_turns(scene, points)
            for finger, angles in turns.items():
                require(all(0 < a < 90 for a in angles.values()), f"{finger} has an unexpected signed turn: {angles}")
            render_started = perf_counter()
            frames = prepare_frames(name, scene, points, edges, views)
            side_views = finger_side_views(scene, points)
            render_seconds = perf_counter() - render_started
        except (ValueError, KeyError, StopIteration, OSError) as error:
            rejected[name] = str(error) or type(error).__name__
            continue
        manifest["pose_keypoints_cad_m"][name] = [p["world_m"] for p in points]
        audits[name] = scene.joints
        row = []
        for annotation, rgb, depth, overlay in frames:
            frame = f"{len(manifest['frames']) + 1:03d}"
            annotation.update(frame=frame, rgb=f"rgb/{frame}.png", depth=f"depth_mm/{frame}.png",
                              overlay=f"overlay/{frame}.png")
            write_png(output / annotation["rgb"], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            write_png(output / annotation["depth"], depth)
            write_png(output / annotation["overlay"], cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            write_json(output / "labels" / f"{frame}.json", annotation)
            manifest["frames"].append(frame)
            row.append((f"{frame} {name} / {annotation['view']}", overlay))
        gallery.append(row)
        side_gallery.append([(f"{name} / {finger}", image) for finger, image in zip(LONG_FINGERS, side_views)])
        accepted[name] = {
            "signed_turns_deg": turns, "driver_deg": exported["measured_deg"],
            "solve_seconds": exported.get("solve_seconds"), "export_seconds": exported.get("export_seconds"),
            "render_seconds": render_seconds,
        }

    write_json(output / "manifest.json", manifest)
    write_json(output / "joint_audit.json", {"units": "Fusion centimeters and radians", "per_pose_joints": audits})
    write_json(output / "geometry_review.json", {"accepted": accepted, "rejected": rejected, "export": report})
    require("as_saved" in accepted, "The saved reference failed. See geometry_review.json.")
    validation = validate_files(output)
    validation.update(accepted_groups=len(accepted), rejected_groups=len(rejected))
    write_json(output / "validation.json", validation)
    make_gallery(gallery, output / "gallery.png", (240, 136))
    make_gallery(side_gallery, output / "finger_side_review.png", (300, 180))
    print(f"{len(accepted)} accepted groups, {len(rejected)} rejected groups, {validation['frames']} frames")
    print(output / "validation.json")
    return bool(rejected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory containing Fusion export_report.json")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    raise SystemExit(generate(args.input, args.out))
