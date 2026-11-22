#!/usr/bin/env python3
"""Prepare a clean Hugging Face dataset snapshot and optionally publish it."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


def copy_raw_object_sessions(local_data: Path, staging: Path) -> list[Path]:
    source = local_data / "raw" / "object"
    destination = staging / "raw" / "object"
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    if not source.exists():
        return copied
    for session in sorted(path for path in source.iterdir() if path.is_dir()):
        target = destination / session.name
        shutil.copytree(session, target, dirs_exist_ok=True)
        for unwanted in target.rglob("derived"):
            if unwanted.is_dir():
                shutil.rmtree(unwanted)
        copied.append(target)
    return copied


def copy_approved_annotations(
    local_data: Path, staging: Path,
) -> tuple[list[dict], dict[str, set[str]]]:
    source_root = local_data / "annotations" / "object_segmentation"
    destination_root = staging / "annotations" / "object_segmentation"
    destination_root.mkdir(parents=True, exist_ok=True)
    approved = []
    class_ids: dict[str, set[str]] = defaultdict(set)

    for source in sorted(source_root.glob("*/annotations/*.json")):
        data = json.loads(source.read_text(encoding="utf-8"))
        if data.get("review_status") != "approved":
            continue
        session_name = Path(data["source_session"]).name
        data["source_session"] = f"raw/object/{session_name}"
        relative = source.relative_to(source_root)
        target = destination_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        approved.append(data)
        for obj in data.get("objects", []):
            class_ids[str(obj["class_name"])].add(str(obj["class_id"]))
    return approved, class_ids


def write_metadata(staging: Path, approved: list[dict],
                   class_ids: dict[str, set[str]]) -> None:
    metadata = staging / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)

    (metadata / "naming_rules.md").write_text(
        """# Naming Rules\n
- Sessions use `raw/object/<session-name>/` and contain immutable RGB-D files.
- RGB and depth frame numbers use matching names: `frame_000001.png` and `frame_000001.npy`.
- Session names describe the captured object or pose, without changing after publication.
- Annotation files follow `annotations/object_segmentation/<session>/annotations/frame_<id>.json`.
- Annotations reference raw sessions relatively and never duplicate raw images.
- Only `approved` annotations are included in this snapshot.
- `derived/`, previews, temporary masks, models, exports, and augmentation archives are excluded.
""",
        encoding="utf-8",
    )
    (metadata / "collection_guide.md").write_text(
        """# Collection Guide\n
Collect aligned RGB-D sessions with stable intrinsics and unchanged raw files.
Record distance, angle, background, lighting, partial occlusion, depth validity,
and camera motion in session notes where available.

Keep raw RGB and depth frames unchanged. Generate preprocessing artifacts separately
and never overwrite `raw/`. Review annotations before training, and use only
`approved` objects.

Dataset updates are published as new Hugging Face commits/revisions, never by
silently replacing existing data.
""",
        encoding="utf-8",
    )

    classes = []
    for name in sorted(class_ids):
        ids = sorted(class_ids[name], key=int)
        classes.append({
            "name": name,
            "class_ids": [int(value) for value in ids],
        })
    (metadata / "classes.yaml").write_text(
        "schema: npulse.classes.v1\n"
        f"class_count: {len(classes)}\n"
        "classes:\n"
        + "".join(
            f"  - name: {item['name']}\n"
            f"    class_ids: {item['class_ids']}\n"
            for item in classes
        ),
        encoding="utf-8",
    )
    (metadata / "keypoints.yaml").write_text(
        "schema: npulse.keypoints.v1\n"
        "keypoint_count: 0\n"
        "keypoints: []\n"
        "status: arm-pose-keypoints-not-yet-collected\n",
        encoding="utf-8",
    )


def write_readme(staging: Path, raw_sessions: list[Path],
                 approved_count: int) -> None:
    (staging / "README.md").write_text(
        f"""# N-Pulse RGB-D Dataset\n
Aligned real RGB-D data for object segmentation and robotic/prosthesis grasp planning.

## Structure

- `raw/object/`: original RGB-D capture sessions.
- `raw/arm_pose/`: reserved for future arm-pose sessions.
- `annotations/object_segmentation/`: approved frame annotations.
- `annotations/arm_pose/`: reserved for future arm-pose annotations.
- `metadata/`: classes, naming rules, collection guidance, and keypoint schema.

## Current Snapshot

- Object sessions: {len(raw_sessions)}
- Approved annotation frames: {approved_count}
- Augmented images are not stored in this snapshot.
- Augmentation settings should be stored separately and reproduced on demand.

## Annotation Contract

Each approved annotation may contain `bbox_xyxy` and `polygon_xy`. Each annotation
references a raw session relative to this repository. RGB and matching depth frames
retain matching frame numbers.

## Updates

New data is added through a new Hugging Face commit/revision. Existing revisions
remain available and are not silently overwritten.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-data", type=Path,
        default=Path(r"C:\npulse_cv\ARM-SOFT-data"),
    )
    parser.add_argument(
        "--staging", type=Path,
        default=Path(r"C:\npulse_cv\ARM-SOFT-data\catalog\huggingface_clean_v1"),
    )
    parser.add_argument(
        "--repo", default="npulseDataCV/npulse_cv",
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--commit-message",
        default="Clean dataset structure and publish approved snapshot",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Delete every remote file before uploading instead of adding on top",
    )
    args = parser.parse_args()

    if args.staging.exists():
        shutil.rmtree(args.staging)
    args.staging.mkdir(parents=True)

    raw_sessions = copy_raw_object_sessions(args.local_data, args.staging)
    (args.staging / "raw" / "arm_pose").mkdir(parents=True)
    (args.staging / "raw" / "arm_pose" / "README.md").write_text(
        "# Arm Pose Raw Sessions\n\nReserved for future RGB-D arm-pose sessions.\n",
        encoding="utf-8",
    )
    approved, class_ids = copy_approved_annotations(
        args.local_data, args.staging,
    )
    (args.staging / "annotations" / "arm_pose").mkdir(parents=True)
    (args.staging / "annotations" / "arm_pose" / "README.md").write_text(
        "# Arm Pose Annotations\n\nReserved for future approved arm-pose annotations.\n",
        encoding="utf-8",
    )
    write_metadata(args.staging, approved, class_ids)
    write_readme(args.staging, raw_sessions, len(approved))
    (args.staging / ".gitattributes").write_text(
        "*.png filter=lfs diff=lfs merge=lfs -text\n"
        "*.npy filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "local_data": str(args.local_data.resolve()),
        "staging": str(args.staging.resolve()),
        "raw_object_sessions": len(raw_sessions),
        "approved_annotation_frames": len(approved),
        "classes": {name: sorted(ids, key=int) for name, ids in class_ids.items()},
    }
    (args.staging / "metadata" / "snapshot.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))

    if not args.upload:
        print("Dry run complete. Re-run with --upload to publish.")
        return 0

    api = HfApi()
    upload_arguments = dict(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=args.staging,
        path_in_repo="",
        commit_message=args.commit_message,
        commit_description=(
            "Clean snapshot following the N-Pulse raw/annotations/metadata structure."
        ),
    )
    if args.replace_existing:
        upload_arguments["delete_patterns"] = "**/*"
    result = api.upload_folder(**upload_arguments)
    print(f"Published revision: {result.commit_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
