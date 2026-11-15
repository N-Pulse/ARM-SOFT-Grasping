# Local datasets

This directory is the local data workspace. Large captures, annotations,
exports and weights are intentionally ignored by Git.

- `captures/`: immutable RGB-D sessions from the shared collector.
- `object_segmentation/`: object masks, splits and YOLO-Seg exports.
- `mechanical_hand_pose/`: keypoints, splits and YOLO-Pose exports.

Keep the session identifier in every downstream annotation record. Dataset
Studio stores real captures in the sibling `../ARM-SOFT-data` directory by
default. See the [Dataset Studio guide](../docs/guides/dataset-studio.md) for
the current workflow and storage rules.
