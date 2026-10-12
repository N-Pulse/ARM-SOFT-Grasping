# Mechanical-hand pose architecture

## Branches

`feature/mechanical-hand-pose` is the integration branch. It now includes the
shared RGB-D collector and depth-preprocessing stack. Short-lived work branches
can still be created from it and merged back through review:

1. `feat/keypoint-schema`
2. `feat/orange-tip-tracking`
3. `feat/annotation-tool` — completed in the integration branch
4. `feat/video-label-propagation`
5. `feat/yolo-pose-export` — completed in the integration branch
6. `feat/yolo-pose-training` — training entry point completed
7. `feat/pseudo-labeling`
8. `feat/pose-evaluation`
9. `feat/d405-depth-fusion`

Create a branch only when its work starts, so it is based on the latest
integration commit. Keep human MediaPipe experiments on
`feature/mediapipe-actions`; do not merge them into this project.

## Data flow

```text
recordings -> frame selection -> assisted annotation -> canonical annotations
  -> YOLO export -> training -> evaluation -> pseudo-label/review loop
  -> RGB-D inference
```

The canonical 17-keypoint order lives in `armsoft/mechanical_hand/keypoints.py`.
Every annotation, exporter, model and evaluator must import or validate against
that definition.

## Pull requests

Each pull request should contain one capability, tests, a sample command and a
small visual example. Dataset-format changes require a version bump and
migration notes. Never commit raw recordings, full datasets or generated
weights to normal Git.
