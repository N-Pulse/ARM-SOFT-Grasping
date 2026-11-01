# Unified RGB-D vision pipeline

## Branch and scope

The integration branch is `integration/unified-vision-pipeline`. It starts from
the newest `feature/object-segmentation` branch and adds the point-cloud and
mechanical-hand pose/RGB-D features without bringing back older copies of the
Dataset Studio.

The target is one real-time RGB-D runtime with selectable models, not several
scripts that each open the camera independently.

```text
RealSense D405 / replay
          |
          v
 one aligned RGB-D frame + intrinsics
          |
          +--> selected model adapter
          |      segmentation | detection | pose
          |              |
          |              v
          |      common instances/keypoints
          |              |
          +------> valid-depth sampling
          |              |
          |              +--> centre pixel, XYZ, camera distance
          |
          +------> optional point-cloud pipeline
                         |
                         +--> scene/instance PLY + JSON metadata
```

## Model policy

The first Web implementation should load **one primary model at a time**. This
keeps latency predictable and allows trained segmentation, detection, and pose
weights to be selected from one registry. The registry records task type,
weights, class/keypoint names, image size, confidence threshold, and device.

Multiple simultaneous models are a later opt-in mode. They must share the same
captured frame and run through a scheduler; no model is allowed to create a
second RealSense connection.

## Common output contract

Every live frame will expose:

- model name, task type, frame timestamp, inference latency, and effective FPS;
- each detected instance's class, confidence, centre pixel, centre XYZ in the
  D405 camera coordinate system, and centre depth in metres;
- segmentation polygon/mask, detection box, or pose keypoints when applicable;
- depth validity/confidence so an invalid distance is never shown as zero;
- optional point-cloud artifact paths and the exact filtering configuration.

For segmentation, centre depth is the median of valid depth inside the mask and
the centre pixel is the mask centroid. For detection, depth comes from the
central box region. For pose, the chosen semantic keypoint or robust median of
valid keypoints is used. All XYZ values are produced with the aligned depth and
the same frame's camera intrinsics.

## Storage rules

- Raw RGB-D capture sessions remain immutable and are never copied into model
  run folders.
- Live overlays are transient unless the user saves a failure case.
- Point clouds are generated only on request or at a configured low rate; they
  are not saved for every live frame by default.
- A saved PLY has a neighbouring JSON record containing source session/frame,
  selected model, filter settings, centre distance, and generation time.
- Model weights stay under the shared `ARM-SOFT-data/models` or `runs` tree and
  are referenced by the registry instead of copied into Git branches.

## Current implementation status

- Dataset Studio segmentation/detection training and Live Test: integrated.
- Detection/segmentation centre depth and XYZ: integrated.
- Mechanical-hand annotation, YOLO Pose training tools, and RGB-D 3-D keypoint
  fusion: integrated and tested.
- RGB-D point-cloud generation, filtering, PLY/JSON export, and CLI viewing:
  integrated and tested.
- One Web Live Test selector for segmentation/detection/pose models: next.
- On-demand scene/selected-instance point-cloud export from Web Live Test: next.

## Acceptance checks for the Web integration

1. Only one process owns the D405 and every consumer receives the same frame.
2. Switching models stops the previous inference worker before loading another.
3. The page shows task type, weights, FPS, latency, centre pixel, depth, and XYZ.
4. Missing/invalid depth is displayed explicitly and never fabricated.
5. Point-cloud export cannot write outside the shared data root and does not
   overwrite an existing artifact.
6. CPU-only, Apple MPS, and CUDA systems degrade cleanly according to available
   devices; the camera-free replay path remains testable.
