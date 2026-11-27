"""Read N-Pulse Fusion v22 mesh snapshots and extract the 21 CAD points."""

from collections import defaultdict
from dataclasses import dataclass
import json

import numpy as np


FINGERS = {
    "thumb": "NewThumbAssembly:1",
    "index": "animation (3):1",
    "middle": "animation (2):1",
    "ring": "animation (1):1",
    "little": "animation:1",
}
LONG_FINGERS = tuple(name for name in FINGERS if name != "thumb")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def unit(vector):
    scale = np.linalg.norm(vector)
    if not np.isfinite(scale) or scale < 1e-12:
        raise ValueError("A CAD direction is missing or degenerate")
    return vector / scale


def transform(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    return matrix


def apply(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def rigid_fit(local, world, tolerance):
    a, b = local - local.mean(0), world - world.mean(0)
    u, singular_values, vt = np.linalg.svd(a.T @ b)
    if singular_values[1] < 1e-12:
        raise ValueError("CAD points are collinear")
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    matrix = transform(rotation, world.mean(0) - rotation @ local.mean(0))
    if np.linalg.norm(apply(local, matrix) - world, axis=1).max() > tolerance:
        raise ValueError("CAD body points do not follow a rigid transform")
    return matrix


def part_color(path, name):
    name_lower = path.lower()
    if any(s in name_lower for s in ("control_board", "power_board", "battery", "din 912", "liner:")):
        return None
    top = path.split("+")[0]
    if top.startswith("biomimetic") and not (
        (path == top and name in ("top", "bottom", "battery_casing"))
        or any(s in name_lower for s in ("coque", "wrist_protection", "structure assembly"))
    ):
        return None
    if any(s in name_lower for s in ("silicon", "oldthumbtip", "protection", "coque", "liner")):
        return (91, 106, 118)
    if any(s in name_lower for s in ("gear", "faulhaber", "moons", "axe", "axis", "bearing", "ressort")):
        return (140, 153, 160)
    return (187, 195, 199)


@dataclass
class Scene:
    triangles: np.ndarray
    colors: np.ndarray
    body_ids: np.ndarray
    paths: list
    vertices: dict
    body_samples: dict
    joints: list

    def joint(self, name, prefix, first_only=False):
        for joint in self.joints:
            ends = (joint["one"],) if first_only else (joint["one"], joint["two"])
            if joint["name"] == name and any((part or "").startswith(prefix) for part in ends):
                return joint
        raise ValueError(f"Missing joint {name} in {prefix}")


def load_scene(folder):
    raw = read_json(folder / "scene.json")
    if (raw["doc"], raw["version"]) != ("Assembly_hand_old_motors", 22):
        raise ValueError("Expected Assembly_hand_old_motors v22")
    errors = [item for item in raw["errors"] if item[0] != "unresolved occurrence"]
    if errors or raw["skip"]:
        raise ValueError(f"Fusion reported mesh or joint export problems: {errors + raw['skip']}")
    vertices = defaultdict(list)
    triangles, colors, body_ids, paths = [], [], [], []
    samples, meshes = {}, {}
    for body in raw["bodies"]:
        path, name = body["path"], body["name"]
        color = part_color(path, name)
        if not body["visible"] or color is None:
            continue
        local, world = np.asarray(body["local"]), np.asarray(body["world"])
        matrix = rigid_fit(local, world, tolerance=0.001)
        key = body["key"]
        if key not in meshes:
            v = np.fromfile(folder / (key + ".v"), dtype="<f4").reshape(-1, 3)
            f = np.fromfile(folder / (key + ".f"), dtype="<i4").reshape(-1, 3)
            if not f.size or f.min() < 0 or f.max() >= len(v):
                raise ValueError(f"Invalid mesh indices for {path} / {name}")
            meshes[key] = v, f
        v, f = meshes[key]
        posed = apply(v, matrix) * 0.01  # Fusion centimeters to meters
        vertices[path].append(posed)
        samples[(path, name)] = (local, world * 0.01)
        body_ids.append(np.full(len(f), len(paths), dtype=np.int32))
        paths.append(path)
        triangles.append(posed[f])
        colors.append(np.tile(np.asarray(color, dtype=np.uint8), (len(f), 1)))
    return Scene(np.concatenate(triangles), np.concatenate(colors), np.concatenate(body_ids),
                 paths, vertices, samples, raw["joints"])


def mesh_vertices(scene, prefix, fragment):
    matches = [verts for path, group in scene.vertices.items()
               if path.startswith(prefix) and fragment in path for verts in group]
    if not matches:
        raise ValueError(f"Missing mesh {prefix} / {fragment}")
    return np.unique(np.concatenate(matches).round(7), axis=0)


def centered_hinge(joint, vertices):
    # Choose a point on the CAD axis at the middle of the relevant part width.
    axis = unit(np.asarray(joint["axis"], dtype=float))
    origin = joint["origin"] if joint["origin"] is not None else joint["t1"][3:12:4]
    origin = np.asarray(origin) * 0.01
    width = vertices @ axis
    return origin + axis * ((width.min() + width.max()) / 2 - origin @ axis)


def tip_cap(vertices, anchor):
    center = (vertices.min(0) + vertices.max(0)) / 2
    projection = vertices @ unit(center - anchor)
    return vertices[projection >= projection.max() - 0.0015].mean(0)


def keypoints(scene, reference=None, reference_points=None):
    points = []
    edges = [(0, 1)]

    def add(name, group, kind, coords):
        points.append({"name": name, "group": group, "type": kind, "world_m": coords.tolist()})

    wrist_prefix = "assemblage2:1+°Axe hand 2:"
    wrist_joint = scene.joint("Revolute 1", wrist_prefix, first_only=True)
    wrist = centered_hinge(wrist_joint, mesh_vertices(scene, wrist_prefix, "°Axe hand 2:"))
    add("wrist_axle_center", "wrist", "hinge", wrist)
    add("palm_center", "palm", "landmark", wrist)
    bases = []
    for digit, prefix in FINGERS.items():
        if digit == "thumb":
            tip_path = prefix  # visible NewTip body at this occurrence
            raw_hinges = (("base", "Pouce A (holder-PhalangeThumb)"), ("distal", "Pouce C"))
            vertices = np.concatenate(scene.vertices[tip_path])
        else:
            tip_path = next(path for path in scene.vertices if path.startswith(prefix + "+Tip_Silicon_part:"))
            raw_hinges = (("base", "Revolute 1"), ("intermediate", "Revolute 5"), ("distal", "Revolute 11"))
            vertices = mesh_vertices(scene, prefix + "+", "Tip_Silicon_part")
        parent = 1
        for part, raw_name in raw_hinges:
            center = centered_hinge(scene.joint(raw_name, prefix), vertices)
            if part == "base" and digit != "thumb":
                bases.append(center)
            edges.append((parent, len(points)))
            parent = len(points)
            add(digit + "_" + part, digit, "hinge", center)
        tip_name = digit + "_tip"
        if reference is None:
            tip = tip_cap(vertices, center)
        else:
            body = next(key for key in reference.body_samples if key[0] == tip_path)
            original_local, original_world = reference.body_samples[body]
            local, world = scene.body_samples[body]
            if not np.allclose(local, original_local, atol=1e-8, rtol=0):
                raise ValueError(f"{digit} terminal mesh changed")
            tip = apply(reference_points[tip_name], rigid_fit(original_world, world, tolerance=1e-6))
        edges.append((parent, len(points)))
        add(tip_name, digit, "landmark", tip)
    points[1]["world_m"] = ((wrist + np.mean(bases, axis=0)) / 2).tolist()
    return points, edges


def point_map(points):
    return {point["name"]: np.asarray(point["world_m"]) for point in points}


def signed_turns(scene, points):
    p = point_map(points)
    radial = p["index_base"] - p["little_base"]
    turns = {}
    for digit in LONG_FINGERS:
        chain = [p[digit + "_" + part] for part in ("base", "intermediate", "distal", "tip")]
        angles = {}
        for i, (part, name) in enumerate((("middle", "Revolute 5"), ("distal", "Revolute 11"))):
            joint = scene.joint(name, FINGERS[digit] + "+", first_only=True)
            axis = unit(np.asarray(joint["axis"], dtype=float))
            if axis @ radial < 0:
                axis *= -1
            a, b = chain[i + 1] - chain[i], chain[i + 2] - chain[i + 1]
            a, b = unit(a - axis * (axis @ a)), unit(b - axis * (axis @ b))
            angles[part] = float(np.degrees(np.arctan2(axis @ np.cross(a, b), a @ b)))
        turns[digit] = angles
    return turns
