# Fine-tuning on D405 images

Separate from `benchmark_val.md`. Boxes come from approved, visually checked frames.

## Test by class (objects never seen in train or val)

| run | class | boxes | found | wrong class | recall | AP50 |
|---|---|---|---|---|---|---|
| b2_yolo26s_d405 | can | 11 | 11 | 0 | 1.00 | 0.995 |
| b2_yolo26s_d405 | box | 19 | 19 | 0 | 1.00 | 0.995 |
| b2_yolo26s_mix | can | 11 | 11 | 0 | 1.00 | 0.995 |
| b2_yolo26s_mix | box | 19 | 19 | 0 | 1.00 | 0.995 |

## Val by class (used to choose the epoch, so optimistic)

| run | class | boxes | found | wrong class | recall | AP50 |
|---|---|---|---|---|---|---|
| b2_yolo26s_d405 | can | 8 | 8 | 0 | 1.00 | 0.995 |
| b2_yolo26s_d405 | box | 20 | 20 | 0 | 1.00 | 0.995 |
| b2_yolo26s_mix | can | 8 | 8 | 0 | 1.00 | 0.995 |
| b2_yolo26s_mix | box | 20 | 20 | 0 | 1.00 | 0.995 |

## By session

| run | session (split) | class | frames | boxes | found | wrong class | recall | AP50 | extra detections |
|---|---|---|---|---|---|---|---|---|---|
| b2_yolo26s_d405 | can_7 (test) | can | 11 | 11 | 11 | 0 | 1.00 | 0.995 | 0 |
| b2_yolo26s_d405 | test-box-2 (test) | box | 19 | 19 | 19 | 0 | 1.00 | 0.995 | 4 |
| b2_yolo26s_d405 | can_5 (val) | can | 8 | 8 | 8 | 0 | 1.00 | 0.995 | 0 |
| b2_yolo26s_d405 | box-beige-1 (val) | box | 20 | 20 | 20 | 0 | 1.00 | 0.995 | 0 |
| b2_yolo26s_mix | can_7 (test) | can | 11 | 11 | 11 | 0 | 1.00 | 0.995 | 3 |
| b2_yolo26s_mix | test-box-2 (test) | box | 19 | 19 | 19 | 0 | 1.00 | 0.995 | 3 |
| b2_yolo26s_mix | can_5 (val) | can | 8 | 8 | 8 | 0 | 1.00 | 0.995 | 0 |
| b2_yolo26s_mix | box-beige-1 (val) | box | 20 | 20 | 20 | 0 | 1.00 | 0.995 | 1 |

## False detections on hands, mug and egg

None of these objects is one of the six classes, so every detection counts as false. All frames of each session are used.

| run | session | frames | frames with a detection | detections | by class |
|---|---|---|---|---|---|
| b2_yolo26s_d405 | right-hand-3fingers-1 | 20 | 1 | 1 | pen_pencil 1 |
| b2_yolo26s_d405 | right-hand-thumbs-up-1 | 20 | 1 | 1 | box 1 |
| b2_yolo26s_d405 | right-hand-thumbs-up-2 | 20 | 15 | 15 | can 15 |
| b2_yolo26s_d405 | mug_1 | 20 | 20 | 34 | box 14, can 20 |
| b2_yolo26s_d405 | ellipsoid | 20 | 20 | 20 | can 20 |
| b2_yolo26s_d405 | ellipsoid_2 | 20 | 19 | 20 | can 20 |
| b2_yolo26s_d405 | ellipsoid_3 | 20 | 14 | 14 | can 14 |
| b2_yolo26s_mix | right-hand-3fingers-1 | 20 | 0 | 0 | – |
| b2_yolo26s_mix | right-hand-thumbs-up-1 | 20 | 0 | 0 | – |
| b2_yolo26s_mix | right-hand-thumbs-up-2 | 20 | 0 | 0 | – |
| b2_yolo26s_mix | mug_1 | 20 | 0 | 0 | – |
| b2_yolo26s_mix | ellipsoid | 20 | 20 | 20 | ball 20 |
| b2_yolo26s_mix | ellipsoid_2 | 20 | 20 | 20 | ball 20 |
| b2_yolo26s_mix | ellipsoid_3 | 20 | 20 | 20 | ball 20 |

## Open Images val: were classes forgotten?

| run | set | mAP50 | ball | can | box | bottle | pen_pencil | door_handle |
|---|---|---|---|---|---|---|---|---|
| b2_yolo26s_d405 | near | 0.028 | 0.029 | 0.036 | 0.020 | 0.015 | 0.066 | 0.000 |
| b2_yolo26s_d405 | far | 0.048 | 0.003 | 0.235 | 0.025 | 0.000 | 0.026 | 0.000 |
| b2_yolo26s_mix | near | 0.593 | 0.702 | 0.680 | 0.492 | 0.541 | 0.621 | 0.519 |
| b2_yolo26s_mix | far | 0.318 | 0.176 | 0.444 | 0.306 | 0.414 | 0.239 | 0.331 |

## Limits

- Very small test: one object per class (tall can, red cube), no pen_pencil, no ball, bottle or
  door_handle. Indicative only, not a verdict on CV-REQ-002.
- Captures are close: about 20-35 cm on the object, measured from depth on 3 frames per session,
  so outside the required 50 cm-1 m range.
- Almost the same white table in every session. Frames of a session are strongly correlated.
- Counts use conf 0.25 and IoU 0.5. AP50 comes from Ultralytics validation.
