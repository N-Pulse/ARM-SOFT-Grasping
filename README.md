# ARM-SOFT Vision Platform

Computer-vision tools for the N-Pulse project:

- Intel RealSense D405 RGB-D capture and dataset management;
- depth preprocessing, assisted annotation, review and YOLO export;
- object detection/segmentation and mechanical-hand pose workflows;
- live model testing, distance measurement and point-cloud export;
- a reproducible Open Images + Hugging Face D405 dataset pipeline.

`main` is the supported team integration branch. Create new feature branches from an up-to-date `main`.

## 1. Environment setup

Python 3.9–3.12 is recommended. Create the virtual environment locally; do not copy `.venv` between computers.

### macOS / Linux

```bash
git clone https://github.com/N-Pulse/ARM-SOFT-Grasping.git
cd ARM-SOFT-Grasping
bash setup_env.sh
source .venv/bin/activate
```

### Windows PowerShell

```powershell
git clone https://github.com/N-Pulse/ARM-SOFT-Grasping.git
cd ARM-SOFT-Grasping
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install pyrealsense2
```

For an NVIDIA GPU, install the matching CUDA build of PyTorch instead of the CPU command. Simulator and replay modes work without a camera. Live D405 use requires `pyrealsense2` on Windows/Linux; macOS requires locally built librealsense Python bindings.

Verify the installation:

```bash
python -m armsoft.tests.selftest
python -m pytest -q armsoft/tests training/tests
```

## 2. Start Dataset Studio

Dataset Studio is the main local web app for capture, preprocessing, annotation, review, dataset export, training and live testing.

```bash
python tools/dataset_web.py
```

Open <http://127.0.0.1:8765> and stop the server with `Ctrl+C`.

Run without a camera:

```bash
python tools/dataset_web.py --source sim
```

Check camera access:

```bash
python -c "from armsoft import camera_available; print(camera_available())"
```

The default shared data directory is `../ARM-SOFT-data`. Raw captures, annotations, exported datasets, failure cases and model weights stay outside normal Git history. Override the location with `ARM_SOFT_DATA_ROOT` or `--data-root`.

Camera notes:

- Use a USB 3 port and close RealSense Viewer or any process using the camera.
- On Windows, run directly in PowerShell; administrator mode is not required.
- On macOS, follow the librealsense setup in the [Dataset Studio guide](docs/dataset_studio_user_guide.md).

## 3. Build the training dataset

The pipeline downloads six Open Images classes and the team D405 dataset from Hugging Face, combines them and validates the YOLO detection dataset. Pinned inputs are defined in `configs/datasets/openimages_d405_v1.yaml`.

One-time setup:

```bash
python -m pip install huggingface_hub
python3 -m venv training/.venv-download
training/.venv-download/bin/python -m pip install fiftyone pyyaml
hf auth login
```

Request access to the gated `DouaaKDR/Npulse` dataset before logging in. Then run:

```bash
python training/data/prepare_dataset.py check
python training/data/prepare_dataset.py all --per-class 50  # quick test
python training/data/prepare_dataset.py all                 # full build
python training/data/prepare_dataset.py validate
```

Outputs:

- `training/datasets/openimages6_d405_v1/` — mixed training dataset;
- `training/datasets/d405_v1/` — D405-only evaluation dataset;
- `training/data/openimages6_d405_v1.yaml` — YOLO training configuration.

Generated data is ignored by Git. See the [dataset pipeline handover](docs/dataset_pipeline_handover.md) for source refresh and new D405 revisions.

## 4. Train a detector

The experimental team checkpoint is stored at:

```text
models/catalog/yolo26s-mix-openimages-d405-v1.0.0/best.pt
```

One-epoch smoke test:

```bash
python training/train.py \
  --name smoke_yolo26s_mix_continue \
  --model models/catalog/yolo26s-mix-openimages-d405-v1.0.0/best.pt \
  --data training/data/openimages6_d405_v1.yaml \
  --epochs 1 --fraction 0.05 --imgsz 416 --batch 4 \
  --device cpu --optimizer AdamW --lr0 0.0002 --patience 1
```

Results go to `training/runs/detect/<run-name>/`; the best checkpoint is `weights/best.pt`.

For faster cloud training, open `training/colab/colab_train.ipynb` in Google Colab. It mounts Drive, validates the dataset/model, continues from the team checkpoint and saves runs to `MyDrive/npulse/runs`.

Full options are in [training/README.md](training/README.md).

## 5. Important locations

| Path | Purpose |
| --- | --- |
| `tools/dataset_web.py` | Starts Dataset Studio |
| `armsoft/` | Camera, preprocessing and inference code |
| `configs/datasets/` | Versioned dataset definitions |
| `training/` | Dataset preparation, training, tests and Colab notebook |
| `models/catalog/` | Versioned team models |
| `docs/dataset_studio_user_guide.md` | Web workflow and troubleshooting |
| `docs/dataset_pipeline_handover.md` | Open Images + Hugging Face pipeline |
| `docs/object_detection_handover.md` | Detector status and training handover |
| `docs/point_cloud_pipeline.md` | RGB-D point-cloud workflow |
| `docs/team_git_and_features.md` | Git workflow and feature map |

## 6. Team Git workflow

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/<short-name>
```

Commit source code, configuration and documentation. Do not commit `.venv`, downloaded datasets, generated runs, raw captures, tokens or temporary files. Add large shared models through the documented model catalog/runtime-model workflow.
