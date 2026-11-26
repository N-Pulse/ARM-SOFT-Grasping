# Documentation

Use this page as the documentation entry point. Stable user instructions, system design, future proposals, and experiment evidence are kept separate.

## Guides

| Guide | Purpose |
| --- | --- |
| [Dataset Studio](guides/dataset-studio.md) | Capture, preprocess, annotate, review, export, train, and test RGB-D data |
| [Training data](guides/training-data.md) | Reproduce the Open Images + Hugging Face D405 dataset |
| [Colab training](guides/colab-training.md) | Continue YOLO training on a Colab GPU |
| [Point clouds](guides/point-cloud.md) | Generate, inspect, and export RGB-D point clouds |
| [Runtime models](guides/runtime-models.md) | Register and share model checkpoints |
| [Arduino UNO Q deployment](guides/uno-q-deployment.md) | Connect the D405, deploy YOLO, and view live detection |

## Architecture

| Document | Purpose |
| --- | --- |
| [System architecture](architecture/system.md) | Current components, data flow, storage, and boundaries |
| [Object detection](architecture/object-detection.md) | Detector status, classes, model, and known limitations |
| [Design solution](architecture/design-solution.md) | Formal computer-vision subsystem design definition |

## Future work

Proposals that are not part of the stable pipeline live in [`future_work/`](future_work/README.md). Move an idea into the stable guides only after it is implemented and tested.

## Experiment evidence

Training decisions and measured results remain close to the code under:

```text
training/decisions.md
training/results/
```

Deployment experiments remain under `porting/`. Generated simulation reports remain under `temp-sim-res/`.
