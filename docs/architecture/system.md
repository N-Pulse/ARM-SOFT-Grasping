# System architecture

## Purpose

ARM-SOFT uses one aligned RGB-D stream for data collection, model inference, metric 3-D measurements, and grasp-related geometry.

```text
RealSense D405 / simulator / replay
                 │
                 ▼
      aligned RGB + depth + intrinsics
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
 model adapter       depth preprocessing
 detection /         range / ROI / table
 segmentation / pose       │
       │                   │
       └─────────┬─────────┘
                 ▼
       centre/keypoint depth fusion
                 ▼
      XYZ in D405 camera coordinates
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
 Web/live output      optional point cloud
 failure cases        PLY + JSON metadata
```

## Code boundaries

```text
armsoft/       reusable camera, data, preprocessing, models, geometry, and output code
tools/         executable command-line entry points
web/           Dataset Studio static frontend
training/      detector data preparation, training, evaluation, and Colab workflow
configs/       versioned dataset and model definitions
models/        reviewed team model catalog
docs/          stable guides, architecture, and future proposals
porting/       target-device compatibility experiments
```

`tools/dataset_web.py` owns the camera and serves Dataset Studio. Model workers consume the same latest RGB-D frame; they must not open a second RealSense connection.

## Model policy

The stable Web path loads one primary model at a time to keep latency predictable. The registry supports detection, segmentation, and pose checkpoints. Each registered model records its task, classes/keypoints, dataset version, semantic version, and weights checksum.

Running object detection and arm pose simultaneously is future work and is documented under [`../future_work/`](../future_work/README.md).

## Common live output

Each live result may include:

- model ID, task, timestamp, latency, and FPS;
- class, confidence, and frame-local instance information;
- segmentation polygon, detection box, or pose keypoints;
- centre pixel, valid depth, and XYZ in the D405 camera frame;
- an explicit invalid-depth state instead of a fabricated zero distance.

Coordinate convention:

- `+X`: camera right;
- `+Y`: camera down;
- `+Z`: forward;
- distances: metres.

## Data flow and storage

Raw Dataset Studio data lives in `../ARM-SOFT-data` by default. It is shared across worktrees and excluded from Git. Training downloads/builds live under `training/datasets/` and `training/sources/`, also excluded from Git.

Rules:

- raw RGB-D is immutable and saved once;
- annotations and derived products reference raw frames;
- point clouds are generated on demand rather than for every frame;
- model runs reference their dataset/model versions;
- invalid or missing depth remains explicit;
- large local checkpoints belong in the runtime model registry, not normal Git history.

## Current capabilities

- D405, simulation, and recorded RGB-D sources;
- session capture and duplicate-frame suppression;
- depth range, ROI, and optional table filtering;
- synchronized RGB-D augmentation;
- assisted segmentation annotation and manual review;
- YOLO dataset export, local training, and live testing;
- object detection and mechanical-arm pose workflows;
- 2-D to 3-D depth fusion;
- point-cloud filtering and PLY/JSON export;
- versioned bundled/runtime model registry;
- failure-case capture and reintegration into the dataset workflow.

## Safety and acceptance rules

1. One process owns the D405.
2. Switching models stops the previous inference worker.
3. File operations stay inside the configured data root.
4. Raw captures are not overwritten by derived operations.
5. Every output records enough configuration to reproduce it.
6. Camera-free simulation/replay remains available for testing.
7. Changes to stable behavior require tests and updated guides.
