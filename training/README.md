# training/ — multi-class object detector (ball, can, box, bottle)

YOLO26n detection, fine-tuned from COCO. It will replace (in a later step) the
red-HSV segmentation + cylinder/cuboid classifier in `armsoft/`, which is not
modified here. Works on any object colour, several objects per image.

Design choices and their reasons: [`decisions.md`](decisions.md).

## 1. Environments (native Windows; not WSL: it cannot see the D405)

Below, `python` means the interpreter of the environment named in each section.
Always call it as `python -m pip` / `python -m pytest` (not pip.exe / pytest.exe).

**Training / evaluation / live test** — the team venv
`...\N-pulse\Fresh-Branch\.venv-win\Scripts\python.exe` (Python 3.13.7, torch 2.14 CPU,
ultralytics 8.4.163, pyrealsense2 2.58.4, opencv-python 5.0). Only pytest was missing:

    python -m pip install pytest
    python -m pytest training/tests        # 14 unit tests, no network/camera

A fresh venv also works: install CPU torch first
(`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`), then
`pip install "ultralytics>=8.4" pyrealsense2 opencv-python pyyaml pytest`.

**Open Images download only** — a separate venv, because fiftyone pulls
`opencv-python-headless`, which would replace the training venv's `cv2` and break `cv2.imshow`:

    py -3.13 -m venv training/.venv-download
    training/.venv-download/Scripts/python -m pip install fiftyone pyyaml

Set `YOLO_AUTOINSTALL=false` (PowerShell: `$env:YOLO_AUTOINSTALL="false"`) so Ultralytics
never pip-installs packages on its own; ONNX export then needs `onnx` installed explicitly.

## 2. Dataset (Open Images V7 → YOLO)

    training/.venv-download/Scripts/python training/data/download_openimages.py   # ~500 img/class
    python training/data/preview_dataset.py                                       # visual check

Classes: Ball→ball, Tin can→can, Box→box, Bottle→bottle (`training/data/yolo_format.py`).
Detections flagged IsGroupOf or IsDepiction are skipped; the image is kept.
Split 70/15/15, fixed by seed 0 (the test split is frozen).
Output: `training/datasets/openimages4/` (git-ignored) + `stats.json`; preview in
`training/results/dataset_preview.jpg`. First run downloads ~4.8 GB of Open Images metadata
into `training/datasets/_fiftyone_cache/` (~12 min in total on 2026-10-10).

## 3. Train (local CPU)

    python training/train.py --name iter1          # CPU defaults: imgsz 416, 30 epochs, batch 16

Options: `--epochs --imgsz --batch --device --fraction --model --data --patience`.
Weights: `training/runs/detect/<name>/weights/best.pt` (git-ignored).
Estimated from a 1-epoch smoke test on an Intel Core Ultra 5 225H: ~2 min/epoch, so ~1 h for
30 epochs at imgsz 416 (±30 %). Training is done on Colab instead (next section).

## 4. Training on Colab (free T4 GPU)

1. **Make the zips** (Git Bash, repo root):

       bash training/colab/make_code_zip.sh                # -> training/colab/training_code.zip
       bash training/colab/make_data_zip.sh openimages4    # -> training/colab/openimages4.zip

2. **Upload to Google Drive**: `training_code.zip` → `MyDrive/npulse/` (again after each code
   change); `openimages4.zip` → `MyDrive/npulse/data/` (once; future datasets, e.g. the in-house
   D405 one, get their own zip there).
3. **Run the notebook** `training/colab/colab_train.ipynb` in Colab:
   *Runtime > Change runtime type > T4 GPU*, set `RUN_NAME` / `DATA_ZIPS` / `EPOCHS` / `IMGSZ` /
   `BATCH` / `PATIENCE` / `EXTRA_ARGS` in the first cell, then *Run all*. Checkpoints are written to
   `MyDrive/npulse/runs/<RUN_NAME>/` after every epoch (a resume cell is included).
4. **Download** `MyDrive/npulse/runs/<RUN_NAME>/` into `training/runs/detect/<RUN_NAME>/` locally.
5. **Evaluate locally**, so all numbers come from the same environment:

       python training/eval.py --weights training/runs/detect/<RUN_NAME>/weights/best.pt --split val  --notes "trained on Colab T4"
       python training/eval.py --weights training/runs/detect/<RUN_NAME>/weights/best.pt --split test --notes "trained on Colab T4; info only"

## 5. Evaluate

    python training/eval.py --weights training/runs/detect/iter1/weights/best.pt --split val --notes "baseline"

Writes `training/results/iter1/val/` (metrics.json, metrics.md, confusion matrices) and appends a
row to `training/results/evaluations.csv`. **Every reported number comes from `evaluations.csv`.**
Every decision uses `val`. Each iteration is also evaluated with `--split test`, for information
only; the final test numbers are those of the best-on-val iteration.
`training/results/log.md` holds the analysis (errors, next step).

## 6. Live test (Intel RealSense D405)

    python training/live_test.py --model yolo26n.pt --classes "sports ball,bottle,cup"   # COCO baseline
    python training/live_test.py --model training/runs/detect/iter1/weights/best.pt
    python training/live_test.py --source some_photo_or_video.mp4 --max-frames 30         # no camera

`s` saves a snapshot, `q` quits. Protocol (4 scenes × 30/50/70 cm) and observations:
`training/results/live_tests.md`.

## 7. Switch to the in-house dataset

Put it in YOLO format (`images/{train,val,test}`, `labels/{train,val,test}`) under
`training/datasets/<name>/`, copy `training/data/data.yaml` to e.g. `training/data/inhouse.yaml`,
change `path:` (relative to the yaml), keep `names` identical, then pass
`--data training/data/inhouse.yaml` to train.py and eval.py (on Colab: add `--data ...` to
`EXTRA_ARGS` and the dataset zip to `DATA_ZIPS`). To start from the Open Images model:
`--model training/runs/detect/<best>/weights/best.pt`.

## 8. Export (UNO Q)

    python -c "from ultralytics import YOLO; YOLO('training/runs/detect/<best>/weights/best.pt').export(format='onnx', imgsz=416)"

Needs `onnx` (not installed in the team venv yet). TFLite: see `porting/export_tflite.py`.
