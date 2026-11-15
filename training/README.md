# Object-detection training

This directory contains the reproducible six-class detector workflow for `ball`, `can`, `box`, `bottle`, `pen_pencil`, and `door_handle`.

Start with:

- [training-data reproduction](../docs/guides/training-data.md);
- [Colab GPU training](../docs/guides/colab-training.md);
- [current detector status](../docs/architecture/object-detection.md).

## Environment

Use the project `.venv` for building, training, evaluation, and live tests. Keep FiftyOne in the separate `training/.venv-download` environment because its OpenCV dependencies can conflict with the camera application.

```bash
source .venv/bin/activate
python -m pip install huggingface_hub

python3 -m venv training/.venv-download
training/.venv-download/bin/python -m pip install fiftyone pyyaml
hf auth login
```

Windows uses `.venv\Scripts\python.exe` and `training\.venv-download\Scripts\python.exe`.

## Build the dataset

```bash
python training/data/prepare_dataset.py check
python training/data/prepare_dataset.py all --per-class 50  # quick isolated build
python training/data/prepare_dataset.py all                 # full pinned v1
python training/data/prepare_dataset.py validate
```

The trainable YAML is:

```text
training/data/openimages6_d405_v1.yaml
```

Generated sources, datasets, and YAML files are ignored by Git.

## Smoke training

```bash
python training/train.py \
  --name smoke_yolo26s_mix_continue \
  --model models/catalog/yolo26s-mix-openimages-d405-v1.0.0/best.pt \
  --data training/data/openimages6_d405_v1.yaml \
  --epochs 1 --fraction 0.05 --imgsz 416 --batch 4 \
  --device cpu --optimizer AdamW --lr0 0.0002 --patience 1
```

Results are saved under:

```text
training/runs/detect/<run-name>/
└── weights/
    ├── best.pt
    └── last.pt
```

Use a unique run name. Do not use `--exist-ok` unless intentional replacement is safe.

## Colab training

Package the current code and dataset:

```bash
bash training/colab/make_code_zip.sh
bash training/colab/make_data_zip.sh openimages6_d405_v1
```

Then follow [`docs/guides/colab-training.md`](../docs/guides/colab-training.md). Colab writes each epoch directly to `MyDrive/npulse/runs/<RUN_NAME>/`.

## Evaluate

```bash
python training/eval.py \
  --weights training/runs/detect/<run-name>/weights/best.pt \
  --split val \
  --notes "experiment description"

python training/eval.py \
  --weights training/runs/detect/<run-name>/weights/best.pt \
  --split test \
  --notes "final reporting"
```

Choose models using validation results. Reserve test results for final reporting. D405-specific evaluation is available through `training/eval_d405.py`.

## Live test

The recommended interface is Dataset Studio:

```bash
python tools/dataset_web.py
```

The standalone detector window remains available:

```bash
python training/live_test.py \
  --model training/runs/detect/<run-name>/weights/best.pt
```

## Export

Example ONNX export:

```bash
python -c "from ultralytics import YOLO; YOLO('training/runs/detect/<run-name>/weights/best.pt').export(format='onnx', imgsz=416)"
```

Install ONNX dependencies explicitly in a compatible environment. Target-device validation belongs under `porting/`.

## Rules

- Keep class names and order identical across datasets and models.
- Split D405 data by complete session, not adjacent frames.
- Record dataset version, starting checkpoint, optimizer, learning rate, image size, seed, and source commit.
- Never commit generated datasets, ordinary run folders, tokens, or unreviewed checkpoints.
- Promote a checkpoint to `models/catalog/` only with metadata, checksum, metrics, and known limitations.
