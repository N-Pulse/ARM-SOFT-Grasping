"""Run from Autodesk Fusion's Scripts and Add-Ins dialog."""

import array
import json
import math
from pathlib import Path
import re
from time import perf_counter
import traceback

import adsk.core
import adsk.fusion


DRIVER_OCCURRENCES = ("animation (3):1", "animation (2):1", "animation (1):1", "animation:1")
SOURCE_NAME = "Assembly_hand_old_motors"
SOURCE_VERSION = 22


def optional(read, default=None):
    # Fusion's joint types do not all expose the same properties.
    try:
        return read()
    except (AttributeError, RuntimeError):
        return default


def vector(obj):
    return list(obj.asArray())


def matrix(obj):
    return [obj.getCell(r, c) for r in range(4) for c in range(4)]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def occurrences(collection, errors):
    for index in range(collection.count):
        try:
            item = collection.item(index)
            if not item or not item.component:
                errors.append(["unresolved occurrence", index])
                continue
            name = item.name
        except RuntimeError as error:
            errors.append(["unresolved occurrence", str(error)])
            continue
        if not name.startswith(("Liner:", "structure assembly v13", "control_board:")):
            yield item
            children = optional(lambda: item.childOccurrences)
            if children is not None:
                yield from occurrences(children, errors)


def joint_record(joint, kind):
    motion = joint.jointMotion
    return {
        "name": joint.name,
        "kind": motion.jointType,
        "typ": kind,
        "one": optional(lambda: joint.occurrenceOne.fullPathName),
        "two": optional(lambda: joint.occurrenceTwo.fullPathName),
        "t1": optional(lambda: matrix(joint.geometryOneTransform)),
        "t2": optional(lambda: matrix(joint.geometryTwoTransform)),
        "origin": optional(lambda: vector(joint.geometry.origin)),
        "axis": optional(lambda: vector(motion.rotationAxisVector)),
        "links": optional(lambda: [l.name for l in joint.motionLinks], []),
        "limits": optional(lambda: [
            motion.rotationLimits.isMinimumValueEnabled, motion.rotationLimits.minimumValue,
            motion.rotationLimits.isMaximumValueEnabled, motion.rotationLimits.maximumValue,
        ]),
    }


def export_scene(root, folder):
    folder.mkdir()
    scene = {"doc": SOURCE_NAME, "version": SOURCE_VERSION, "bodies": [], "joints": [],
             "errors": [], "skip": []}
    errors = scene["errors"]
    meshes = {}
    assembly = list(occurrences(root.occurrences, errors))
    for occurrence in assembly:
        path = occurrence.fullPathName
        old_thumb = "OldThumbTip:" in path
        if not optional(lambda: occurrence.isVisible, False) and not old_thumb:
            continue
        try:
            for body in occurrence.bRepBodies:
                if not body.isVisible and not old_thumb:
                    continue
                native = body.nativeObject
                if not native:
                    continue
                token = native.entityToken
                count = native.vertices.count
                if token not in meshes:
                    if native.faces.count > 3500 or count < 4:
                        scene["skip"].append([path, body.name, native.faces.count])
                        continue
                    key = str(len(meshes))
                    calculator = native.meshManager.createMeshCalculator()
                    calculator.setQuality(adsk.fusion.TriangleMeshQualityOptions.LowQualityTriangleMesh)
                    mesh = calculator.calculate()
                    (folder / (key + ".v")).write_bytes(array.array("f", mesh.nodeCoordinatesAsFloat).tobytes())
                    (folder / (key + ".f")).write_bytes(array.array("i", mesh.nodeIndices).tobytes())
                    meshes[token] = key

                # Matching native and assembly-context points preserves the
                # solved transform of each nested body.
                indices = sorted({i * count // 18 for i in range(18)} | {count - 1})
                local, world = [], []
                for index in indices:
                    vertex = native.vertices.item(index)
                    local.append(vector(vertex.geometry))
                    world.append(vector(vertex.createForAssemblyContext(occurrence).geometry))
                scene["bodies"].append({
                    "visible": body.isVisible, "path": path, "name": body.name,
                    "key": meshes[token], "local": local, "world": world,
                })
        except Exception as error:
            errors.append([path, str(error)])

    owners = [(root, None)]
    owners.extend((o.component, o) for o in assembly if optional(lambda: o.isVisible, False))
    for component, occurrence in owners:
        for kind, collection in (("joint", optional(lambda: component.joints, [])),
                                 ("asbuilt", optional(lambda: component.asBuiltJoints, []))):
            for native in collection:
                try:
                    joint = native.createForAssemblyContext(occurrence) if occurrence else native
                    scene["joints"].append(joint_record(joint, kind))
                except Exception as error:
                    errors.append([component.name, native.name, str(error)])
    write_json(folder / "scene.json", scene)
    return {"bodies": len(scene["bodies"]), "joints": len(scene["joints"]),
            "errors": errors, "skipped_bodies": scene["skip"]}


def driver_motions(root):
    by_name = {o.name: o for o in root.occurrences}
    motions = []
    for name in DRIVER_OCCURRENCES:
        occurrence = by_name[name]
        joint = next(j for j in occurrence.component.joints if j.name == "Revolute 1")
        motions.append(joint.createForAssemblyContext(occurrence).jointMotion)
    thumb = next(j for j in root.asBuiltJoints if j.name.startswith("Pouce pilote"))
    return motions + [thumb.jointMotion]


def drive(app, root, targets):
    motions = driver_motions(root)
    start = [math.degrees(m.rotationValue) for m in motions]
    steps = 0
    if targets is not None:
        steps = max(1, math.ceil(max(abs(a - b) for a, b in zip(start, targets)) / 3))
        for step in range(1, steps + 1):
            for motion, a, b in zip(motions, start, targets):
                motion.rotationValue = math.radians(a + (b - a) * step / steps)
            adsk.doEvents()
        app.activeViewport.refresh()
        adsk.doEvents()
    measured = [math.degrees(m.rotationValue) for m in motions]
    if targets is not None and any(not math.isclose(a, b, abs_tol=0.1) for a, b in zip(measured, targets)):
        raise ValueError(f"Fusion joint readback {measured} does not match {targets}")
    return {"requested_deg": targets, "measured_deg": measured, "steps": steps}


def read_poses(config):
    poses = json.loads(config.read_text(encoding="utf-8-sig"))["poses"]
    if "as_saved" not in poses or poses["as_saved"] is not None:
        raise ValueError('Keep "as_saved": null as the reference pose.')
    used_targets = set()
    for name, targets in poses.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name):
            raise ValueError(f"Invalid pose name: {name}")
        if name == "as_saved":
            continue
        if (not isinstance(targets, list) or len(targets) != 5
                or any(type(angle) not in (int, float) or not math.isfinite(angle) or abs(angle) > 180
                       for angle in targets)):
            raise ValueError(f"{name}: supply five joint angles in degrees within [-180, 180].")
        if tuple(targets) in used_targets:
            raise ValueError(f"{name}: duplicate angles.")
        used_targets.add(tuple(targets))
    return poses


def checked_design(doc):
    entry = doc.dataFile if doc else None
    if not entry or entry.name != SOURCE_NAME or entry.versionNumber != SOURCE_VERSION:
        raise ValueError(f"Open {SOURCE_NAME} v{SOURCE_VERSION} in the Fusion desktop app.")
    if doc.isModified:
        raise ValueError("The assembly has unsaved changes. Open a fresh v22 tab before exporting.")
    return entry


def export_batch(app, doc, poses, output):
    source = checked_design(doc)
    output.mkdir()
    report = {"fusion_build": app.version, "poses": {}}
    for name, targets in poses.items():
        if doc is None:
            doc = app.documents.open(source, True)
        checked_design(doc)
        try:
            doc.activate()
            root = adsk.fusion.Design.cast(app.activeProduct).rootComponent
            start = perf_counter()
            result = drive(app, root, targets)
            solved_at = perf_counter()
            result.update(export_scene(root, output / name))
            result.update(solve_seconds=solved_at - start, export_seconds=perf_counter() - solved_at)
        except Exception:
            result = {"requested_deg": targets, "error": traceback.format_exc()}
        finally:
            if not doc.close(False):
                raise RuntimeError("Fusion could not close the export tab. Close it without saving.")
        report["poses"][name] = result
        write_json(output / "export_report.json", report)
        doc = None


def run(_context):
    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        doc = app.activeDocument
        checked_design(doc)
        dialog = ui.createFileDialog()
        dialog.title = "Select N-Pulse pose configuration"
        dialog.filter = "Pose configuration (*.json)"
        if dialog.showOpen() != adsk.core.DialogResults.DialogOK:
            return
        config = Path(dialog.filename).resolve()
        poses = read_poses(config)
        if ui.messageBox(
            f"Export {len(poses)} poses to fusion_export beside this JSON?\n"
            "Fusion will reopen the active v22 assembly between poses and discard the script's changes.",
            "N-Pulse Fusion export",
            adsk.core.MessageBoxButtonTypes.OKCancelButtonType,
        ) != adsk.core.DialogResults.DialogOK:
            return
        output = config.parent / "fusion_export"
        export_batch(app, doc, poses, output)
        ui.messageBox(f"N-Pulse export finished. Review export_report.json in:\n{output}")
    except Exception:
        ui.messageBox(traceback.format_exc())
