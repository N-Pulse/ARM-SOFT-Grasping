# training/ — multi-class object detector (ball, can, box, bottle, pen_pencil, door_handle)

YOLO26n detection, fine-tuned from COCO. It will replace (in a later step) the
red-HSV segmentation + cylinder/cuboid classifier in `armsoft/`, which is not
modified here. Works on any object colour, several objects per image.

Design choices and their reasons: [`decisions.md`](decisions.md).
Current status, results and next steps: [`docs/object_detection_handover.md`](../docs/object_detection_handover.md).
Team tutorials: [reproduce the training dataset](../docs/reproduce_training_dataset.md) and
[continue training on Google Colab](../docs/colab_training_tutorial.md).

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

### One-command Open Images + Hugging Face D405 pipeline

`training/data/prepare_dataset.py` downloads both sources, converts the Dataset Studio
annotations, builds the D405-only and mixed YOLO detection datasets in a staging directory,
validates them, and publishes them only if every check passes. All pinned inputs (classes,
Open Images count and seed, Hugging Face revision, session splits, excluded frames, class
mapping, D405 repeat factor) live in `configs/datasets/openimages_d405_v1.yaml`.

**One-time setup** (macOS/Linux; on Windows use `Scripts\python`):

    python -m pip install huggingface_hub                 # in the normal training .venv
    python3 -m venv training/.venv-download               # FiftyOne must stay out of .venv
    training/.venv-download/bin/python -m pip install fiftyone pyyaml
    hf auth login                                         # gated dataset access required

**Run:**

    python training/data/prepare_dataset.py check                  # environment + outputs
    python training/data/prepare_dataset.py all --per-class 50     # quick run (_pc50 names)
    python training/data/prepare_dataset.py all                    # pinned full dataset
    python training/data/prepare_dataset.py validate               # re-check, no download

Steps can also run one at a time: `download-openimages`, `download-hf`, `build`.

| Behaviour | Detail |
|---|---|
| Quick run | `--per-class N` different from the config adds `_pcN` to every output (source, datasets, yaml), so it never touches the full dataset |
| Sources | reused when their `DOWNLOAD.json` matches the config; otherwise the run stops. `--refresh-sources` downloads again |
| Existing datasets | never overwritten. `--force` moves them to `training/datasets/.previous/<name>-<time>/` and publishes both new datasets together (rolled back if a move fails) |
| Failed validation | nothing is published; the partial build and both `validation_report.json` are kept in `training/datasets/.failed/<time>/` |
| D405 repeats | hard links by default (one copy on disk). Use `--link-mode copy` before zipping for Colab |
| Checks | every image has a label and vice versa, class ids 0-5, coordinates in [0, 1], non-empty boxes, every image decodes (OpenCV or Pillow), no identical image in two splits; D405 sessions are split whole |

Outputs: `training/datasets/d405_v1/`, `training/datasets/openimages6_d405_v1/` (each with
`dataset_manifest.json`, `validation_report.json`, `stats.json`), and the trainable
`training/data/openimages6_d405_v1.yaml`. The manifest records the git revision and the
settings actually used to download each source. The command ends by printing a ready-to-run
`training/train.py` command.

To change any pinned input, copy the config to a new version (`..._v2.yaml`) with new
`name` and output names instead of editing v1, so earlier results stay reproducible.

### Open Images only

    training/.venv-download/Scripts/python training/data/download_openimages.py   # ~500 img/class
    python training/data/preview_dataset.py                                       # visual check

Classes: Ball→ball, Tin can→can, Box→box, Bottle→bottle, Pen→pen_pencil, Door handle→door_handle (`training/data/yolo_format.py`).
Detections flagged IsGroupOf or IsDepiction are skipped; the image is kept.
Split 70/15/15, fixed by seed 0 (the test split is frozen).
Output: `training/datasets/openimages6/` (git-ignored) + `stats.json`; preview in
`training/results/dataset_preview.jpg`. First run downloads ~4.8 GB of Open Images metadata
into `training/datasets/_fiftyone_cache/` (~12 min in total on 2026-10-10).

## 3. Train (local CPU)

    python training/train.py --name iter1          # CPU defaults: imgsz 416, 30 epochs, batch 16

Options: `--epochs --imgsz --batch --device --fraction --model --data --patience`.
Weights: `training/runs/detect/<name>/weights/best.pt` (git-ignored).
Estimated from a 1-epoch smoke test on an Intel Core Ultra 5 225H: ~2 min/epoch, so ~1 h for
30 epochs at imgsz 416 (±30 %). Training is done on Colab instead (next section).

## 4. Continue training on Colab GPU

1. **Make the zips** (Git Bash, repo root):

       bash training/colab/make_code_zip.sh                # -> training/colab/training_code.zip
       bash training/colab/make_data_zip.sh openimages6_d405_v1

2. **Upload to Google Drive** using this layout:

       MyDrive/npulse/training_code.zip
       MyDrive/npulse/data/openimages6_d405_v1.zip
       MyDrive/npulse/models/yolo26s-mix-openimages-d405-v1.0.0/best.pt

   `MyDrive/npulse/data/openimages6_d405_v1.yaml` is optional: the notebook generates the fixed
   six-class YAML when it is absent.
3. **Run the notebook** `training/colab/colab_train.ipynb` in Colab:
   *Runtime > Change runtime type > GPU*, edit only the first parameter cell, then *Run all*.
   Defaults continue from the published mixed `best.pt` with AdamW, `lr0=0.0002`,
   `openimages6_d405_v1`, 416 px and batch 32. The notebook refuses to start without CUDA, checks
   every uploaded file, verifies image/label counts and the six-class order, and prevents accidental
   run-name reuse. Checkpoints are written to `MyDrive/npulse/runs/<RUN_NAME>/` after every epoch;
   an optional resume cell is included.
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
`training/datasets/<name>/`, copy `training/data/openimages6.yaml` to e.g. `training/data/inhouse.yaml`,
change `path:` (relative to the yaml), keep `names` identical, then pass
`--data training/data/inhouse.yaml` to train.py and eval.py (on Colab: add `--data ...` to
`EXTRA_ARGS` and the dataset zip to `DATA_ZIPS`). To start from the Open Images model:
`--model training/runs/detect/<best>/weights/best.pt`.

## 8. Export (UNO Q)

    python -c "from ultralytics import YOLO; YOLO('training/runs/detect/<best>/weights/best.pt').export(format='onnx', imgsz=416)"

Needs `onnx` (not installed in the team venv yet). TFLite: see `porting/export_tflite.py`.

## 9. Benchmark: models and datasets

Every run is judged on the same frozen images, twice: **near** (`openimages6`, Open Images sizes)
and **far** (`openimages6_far`, same images in grids, apparent size at 50 cm-1 m). Web photos only:
this ranks the options; the D405 verdict comes later.

Train datasets: D1 `openimages6`, D2 `openimages6_large` (more train images), D3 `openimages6_far`.

**Colab settings** (Section 4; `EPOCHS 30`, `BATCH 32`, `PATIENCE 10`, `IMGSZ 416` unless stated):

| Step | `RUN_NAME`                    | `DATA_ZIPS`             | `EXTRA_ARGS`                                                       |
| ---- | ----------------------------- | ----------------------- | ------------------------------------------------------------------ |
| 1    | `b1_yolo26n_d1`               | `openimages6.zip`       | `""`                                                               |
| 1    | `b1_yolo11n_d1`               | `openimages6.zip`       | `"--model yolo11n.pt"`                                             |
| 1    | `b1_yolov8n_d1`               | `openimages6.zip`       | `"--model yolov8n.pt"`                                             |
| 1    | `b1_yolo26s_d1`               | `openimages6.zip`       | `"--model yolo26s.pt"`                                             |
| 2    | `b2_<model>_d2` (best 2)      | `openimages6_large.zip` | `"--model <model>.pt --data training/data/openimages6_large.yaml"` |
| 2    | `b2_<model>_d3` (best 2)      | `openimages6_far.zip`   | `"--model <model>.pt --data training/data/openimages6_far.yaml"`   |
| 3    | `b3_<best>_640` (`IMGSZ 640`) | best dataset            | as its best run                                                    |
| 4    | `b4_<best>_seed1` (best 2)    | as the run              | as the run, plus `--seed 1`                                        |

**Evaluate each run on the PC**, near then far (the size follows the run):

    python training/eval.py --weights training/runs/detect/<run>/weights/best.pt --split val --notes "trained on Colab T4"
    python training/eval.py --weights training/runs/detect/<run>/weights/best.pt --split val --data training/data/openimages6_far.yaml --tag <run>__far --notes "far; trained on Colab T4"

**Table:** `python training/compare_runs.py` writes `training/results/benchmark_val.md`.

**Decision rule, fixed in advance:** keep the lightest model, unless another one gains more recall
on the far set than the gap between the two seeds of step 4, at an acceptable CPU time. Use val for
every decision; evaluate test only for the chosen run (`--split test`, then
`compare_runs.py --split test`).
