# Object detection status

## Current model

The project detects six fixed classes in this order:

```text
0 ball
1 can
2 box
3 bottle
4 pen_pencil
5 door_handle
```

The bundled experimental checkpoint is:

```text
models/catalog/yolo26s-mix-openimages-d405-v1.0.0/best.pt
```

It was fine-tuned from YOLO26s on Open Images plus the first D405 dataset. Dataset Studio discovers it automatically in Live Test.

## Training data

The versioned definition is `configs/datasets/openimages_d405_v1.yaml`. It pins Open Images sampling, the Hugging Face revision, class order, D405 session splits, excluded frames, and repeat factor.

Verified v1 counts:

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| D405 only | 122 | 28 | 30 |
| Open Images + D405 | 2686 | 473 | 445 |

Reproduction instructions: [`../guides/training-data.md`](../guides/training-data.md).

## Measured status

- The published model supports all six classes but remains experimental.
- The first D405 set is small, visually narrow, and mainly covers `can`, `box`, and `pen_pencil`.
- D405 captures were mostly around 20–35 cm, while later validation should include the intended 0.5–1 m workspace.
- Mixing Open Images with D405 prevented the severe forgetting observed with D405-only fine-tuning.
- Small/far objects and visually similar negatives remain important failure modes.

Detailed immutable evidence remains in `training/results/` and decisions in `training/decisions.md`.

## Runtime integration

Dataset Studio Live Test provides:

- class, box/mask, and confidence;
- median valid depth from the instance region;
- centre XYZ in D405 camera coordinates;
- latency and FPS;
- labelled failure-case capture.

The detector output is not yet a complete robot-frame grasp pose. Camera-to-robot calibration, per-instance geometry, and validated grasp interaction logic are separate stages.

## Next validation priorities

1. Collect all six classes at 0.5–1 m with varied backgrounds and negative frames.
2. Freeze a larger D405 test split with unseen physical objects.
3. Continue mixed training from the published checkpoint at a conservative learning rate.
4. Compare validation metrics, D405 failure cases, latency, and memory on the target device.
5. Promote a new immutable model version only after those checks pass.

Training commands: [`../../training/README.md`](../../training/README.md). Colab instructions: [`../guides/colab-training.md`](../guides/colab-training.md).
