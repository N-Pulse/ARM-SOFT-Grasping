# RGB-D interaction tracking

Status: **proposed future work**

## Goal

Combine the two main vision models over time:

1. the object detection model identifies objects;
2. the mechanical-arm pose model identifies the arm and hand;
3. D405 depth converts both results into 3-D positions;
4. tracking connects detections across frames;
5. interaction logic identifies simple actions such as approaching, grasping, moving and releasing.

The models do not need to be retrained for the first version. This work extends the live inference pipeline after model prediction.

## Proposed pipeline

```text
                     D405 RGB + Depth
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      Object Detection Model      Mechanical-Arm Pose Model
        class + box/mask              arm keypoints
              │                           │
              ▼                           ▼
       Object Depth Fusion          Pose Depth Fusion
          object XYZ                 hand/arm XYZ
              │                           │
              └─────────────┬─────────────┘
                            ▼
                         Tracking
               persistent object and arm IDs
                            │
                            ▼
                    Interaction Logic
       detected → approaching → grasped → moved → released
                            │
                            ▼
                   Web display and event log
```

## Why this is useful

The current models answer what is visible in one frame. The additional modules would answer what is happening across several frames:

- Is this the same object as before?
- Is the hand approaching the object?
- Was the object successfully grasped?
- Is the object moving together with the hand?
- Was the object released?

This is inspired by the tubelet/state-graph idea from *Tracking and Understanding Object Transformations*, but the proposed implementation is deliberately lightweight and RGB-D based. It does not require the full TubeletGraph stack or an online language model.

## Connection to the current project

Existing components that can be reused:

- YOLO object detection/segmentation;
- YOLO mechanical-arm pose estimation;
- D405-aligned RGB and depth;
- object centre depth and XYZ calculation;
- mechanical-arm keypoint depth fusion;
- temporal smoothing and stable fingertip IDs;
- Dataset Studio Live Test and failure-case capture.

Missing components:

- persistent IDs for detected objects;
- synchronized execution of both models on the same RGB-D frame;
- object-to-hand association;
- interaction state machine;
- event history in Dataset Studio.

## Suggested implementation stages

### Stage 1 — Dual-model inference

Run the object and arm-pose models on the same RGB-D frame. Display both predictions and their 3-D positions in Live Test.

Expected result:

```json
{
  "objects": [{"class": "box", "xyz_m": [0.05, 0.12, 0.48]}],
  "hand": {"xyz_m": [0.07, 0.11, 0.45]}
}
```

### Stage 2 — RGB-D tracking

Give each object a persistent `track_id`. Associate consecutive detections using class, 2-D overlap and 3-D distance. Keep a track alive briefly during occlusion instead of deleting it immediately.

### Stage 3 — Interaction state machine

Start with deterministic states:

```text
detected → approaching → contact_candidate → grasped → moved → released
```

State transitions should use 3-D hand-object distance, relative motion, height above the table and multi-frame stability.

### Stage 4 — Web visualization and logging

Add to Live Test:

- persistent object IDs;
- current interaction state;
- hand-object distance;
- a timestamped event timeline;
- an option to save incorrect tracking or state transitions as failure cases.

### Stage 5 — Annotation assistance

Use tracks to propagate annotations through similar consecutive frames. Send only uncertain frames, ID switches, new tracks and occlusion recoveries to manual review.

## Proposed modules

```text
armsoft/
├── tracking/
│   └── rgbd_tracker.py
├── interaction/
│   ├── state_machine.py
│   └── events.py
└── fusion/
    └── object_hand_fusion.py
```

Names and boundaries can change after the first technical design review.

## Validation

Record short RGB-D sequences covering:

- approach without grasping;
- successful grasp and movement;
- failed grasp;
- temporary hand/object occlusion;
- release onto the table;
- multiple visible objects.

Measure:

- object ID switches;
- lost and recovered tracks;
- 3-D hand-object distance error;
- grasp/release event precision and recall;
- latency and achieved FPS with both models enabled.

## Risks and open questions

- Running two models may reduce live FPS.
- The pose model must provide a reliable hand reference point for distance measurement.
- Grasp detection thresholds depend on camera placement and hand geometry.
- Similar nearby objects may cause ID switches.
- Missing or noisy depth must produce an explicit uncertain state, never a false zero distance.
- The project must decide whether dual-model inference runs on a computer, an edge GPU or another target platform.

## Recommended first milestone

Implement dual-model replay on recorded RGB-D data before modifying the live camera path. A successful first milestone should display object XYZ, hand XYZ and their distance for every replayed frame without yet attempting grasp-state classification.

