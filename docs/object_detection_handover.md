# Object detection: handover

Author: Melvyn Huynh (Computer Vision team), 2026-11-08. Covers the object detector (v1 front-end)
The author is leaving the project; this page is the entry point
for whoever continues it. Design choices and their reasons are in
[`training/decisions.md`](../training/decisions.md); commands are in
[`training/README.md`](../training/README.md).

## 0. Background (read first if you are new to object detection)

**Object detection.** Given an image, produce for each object its class and a
bounding box `(x, y, w, h)` around it. Not "there is a cat in this image"
(classification), but "cat at this rectangle, confidence 92 %".

**YOLO.** A family of CNN-based object detectors. One forward pass of the
network produces all boxes and all classes at once, which is why it runs in real
time. Here we use **YOLO26n** / **YOLO26s** (Ultralytics 8.4): the "n" (nano)
and "s" (small) variants have roughly 3 M and 11 M parameters.

**CNN (Convolutional Neural Network).** A stack of layers; each layer is a set
of small filters (tiny matrices, typically 3x3) that slide over the previous
layer's output and respond when they see a specific pattern. Early layers
respond to simple patterns (edges, colour gradients); deeper layers combine
those into textures, object parts, and eventually whole objects. The last layer
outputs, for each cell of a grid, box coordinates + class scores.

**Weights.** The values inside those filters. For YOLO26n, about 3 million
numbers in total. They start random. During **training**, we show the network
labelled images, compute an error ("loss") on each batch, and the
backpropagation algorithm nudges every weight slightly in the direction that
reduces the error. After many iterations, the weights have converged to values
that, together, detect the target classes. The file `best.pt` is just those
numbers, saved.

**Fine-tuning** (what every run in this project does). Start from weights
pre-trained on another dataset (COCO, 80 classes) and continue training on our
data. Most layers (backbone, neck) barely change: they already know edges,
textures, shapes. The final classification layer is **replaced** (new random
weights for 6 outputs instead of 80) and learns our classes almost from scratch.
Cheaper and more data-efficient than training from zero.

**Train / val / test split.** The dataset is split in three (70/15/15, frozen
by seed 0):

- **train** (~70 %): used to update the weights.
- **val** (~15 %): used at the end of each epoch to pick `best.pt` (snapshot of
  the epoch with the highest val mAP50).
- **test** (~15 %): held out; evaluated once per run **for information only**.
  Choosing by test would leak it into model selection and inflate the reported
  number. All decisions in this project go through val. Details:
  [`training/decisions.md`](../training/decisions.md).

**Where this maps in the code.** [`training/train.py`](../training/train.py) is
a thin CLI wrapper around Ultralytics' `model.train(...)` that saves `best.pt`.
[`training/colab/colab_train.ipynb`](../training/colab/colab_train.ipynb) runs
the same script on a Colab T4 GPU (zipping code + data, mounting Drive,
symlinking the runs folder to Drive so checkpoints survive disconnections).
[`training/eval.py`](../training/eval.py) loads a `best.pt` locally (CPU, for
comparability), calls `model.val(...)`, and appends one row to
[`training/results/evaluations.csv`](../training/results/evaluations.csv): the
single source of truth for every number reported anywhere.

## 1. Status in one page

| Topic | State |
|---|---|
| Model | YOLO detector, 6 classes: `ball`, `can`, `box`, `bottle`, `pen_pencil`, `door_handle` (IDs 0-5, fixed) |
| Best checkpoint so far | `yolo26s-mix-openimages-d405-v1.0.0` in [`models/catalog/`](../models/catalog/), **experimental** |
| Training data | Open Images V7 (all 6 classes) + first D405 dataset (only `can`, `box`, `pen_pencil`) |
| Measured on D405 | Tiny held-out test (30 boxes, 2 objects) found 30/30; false detections measured on hands, mug and egg (section 5) |
| Live camera | Works on a laptop CPU at about 15-20 FPS with the D405. Not run on the UNO Q |
| Not done | Geometry from YOLO boxes (point cloud per object + shape fit), sphere fit, ONNX export, UNO Q tests, tests at 0.5-1 m with ground truth |

The detector is not yet connected to the shape fitter: `armsoft/core/pipeline.py` still takes its
object from the red HSV isolation (v0). Live Test shows class, confidence and the median depth in
the central part of each box, which is a visible-surface distance, not a fitted centre.

## 2. Where the code is

| File | Role |
|---|---|
| [`training/train.py`](../training/train.py) | Train or fine-tune a YOLO model (`--model`, `--data`, `--epochs`, `--imgsz`, ...) |
| [`training/eval.py`](../training/eval.py) | Evaluate weights on a split; writes `training/results/<run>/<split>/` and appends to `evaluations.csv` |
| [`training/compare_runs.py`](../training/compare_runs.py) | Benchmark table across Open Images runs (near and far); ignores D405 fine-tuning runs |
| [`training/eval_d405.py`](../training/eval_d405.py) | D405 report: per class, per session, false detections on hands/mug/egg, Open Images retention |
| [`training/live_test.py`](../training/live_test.py) | Live D405 window: boxes, class, confidence, box depth, FPS (`s` saves a frame) |
| [`training/data/download_openimages.py`](../training/data/download_openimages.py) | Build `openimages6` from Open Images; `--extend-from` grows train only |
| [`training/data/derive_far_dataset.py`](../training/data/derive_far_dataset.py) | Build `openimages6_far` (same images in grids, object size of 50 cm-1 m) |
| [`training/data/build_d405_datasets.py`](../training/data/build_d405_datasets.py) | Build `d405` and `openimages6_d405` from a Dataset Studio data root |
| [`training/data/studio_format.py`](../training/data/studio_format.py) | Studio class-name mapping, session splits and **excluded frames** |
| [`training/data/*.yaml`](../training/data/) | Dataset configs: `openimages6`, `_far`, `_large`, `d405`, `openimages6_d405` (`data.yaml` = old 4-class) |
| [`training/colab/`](../training/colab/) | Colab notebook and zip scripts |
| [`training/tests/`](../training/tests/) | 50 unit tests: `python -m pytest training/tests` |

## 3. Assets outside Git

Datasets, weights and logs are not in Git (`.gitignore`). Before this handover they lived here:

| Asset                                              | Location                                                                       | Action needed                                                                |
| -------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| Colab runs (`best.pt`, `results.csv`, `args.yaml`) | Melvyn's personal Google Drive, `EPFL/Association_Npulse_CV/runs/`             | **Transfer to a team Drive**, otherwise they are lost                        |
| Colab data and code zips                           | same Drive, `Association_Npulse_CV/data/` and root                             | Transfer, or rebuild with the zip scripts                                    |
| Colab notebooks with full training logs            | Melvyn's PC, `N-pulse/train_results_colab/`                                    | Copy to the team Drive: they are the only source of the training-log numbers |
| D405 dataset                                       | Hugging Face `DouaaKDR/Npulse`, revision `ac4f808`                             | Owned by Douaa                                                               |
| CV slides for the SDR (4 slides)                   | Melvyn's PC, `docs/superpowers/N-Pulse_SDR_Computer_Vision.pptx` (git-ignored) | Share with the team                                                          |
| Published weights                                  | `models/catalog/yolo26s-mix-openimages-d405-v1.0.0/`                           | In Git, nothing to do                                                        |

Weights of the other runs (`b1_*`, `b2_yolo26s_d405`) exist only in the Drive `runs/` folder.

## 4. What was trained

All runs: Ultralytics 8.4.163, imgsz 416, batch 32, seed 0, Colab T4, evaluation on the dev PC.
Numbers marked *log* come from the Colab end-of-training validation; the others from `eval.py` or
`eval_d405.py`.

| Run | Start | Train data | Result |
|---|---|---|---|
| `iter1` | `yolo26n.pt` (COCO) | `openimages4` (4 classes, old split) | val mAP50 0.574, test 0.522 ([`evaluations.csv`](../training/results/evaluations.csv)) |
| `b1_yolo26n_d1` | `yolo26n.pt` | `openimages6` | val mAP50 0.604 *log* |
| `b1_yolo11n_d1` | `yolo11n.pt` | `openimages6` | val mAP50 0.583 *log* |
| `b1_yolo26s_d1` | `yolo26s.pt` | `openimages6` | val mAP50 **0.669** *log*, about 3.9x the GFLOPs of YOLO26n |
| `b2_yolo26s_d405` | `b1_yolo26s_d1` | `d405` only, 50 epochs | Forgets the other classes (section 5) |
| `b2_yolo26s_mix` | `b1_yolo26s_d1` | `openimages6_d405` (D405 train repeated 5 times) | Stopped at epoch 11, **best epoch 1**; published as the catalog model |

Comparison of the three `b1` runs: [`training/results/benchmark_preliminary_colab.md`](../training/results/benchmark_preliminary_colab.md).
`yolov8n` was not run. `openimages6_far` and `openimages6_large` were built but never trained.

## 5. D405 results

From [`training/results/d405_finetune/report.md`](../training/results/d405_finetune/report.md)
(conf 0.25, IoU 0.5). `b1_yolo26s_d1` was not included: its weights were not on the PC.

| | `b2_yolo26s_d405` | `b2_yolo26s_mix` |
|---|---|---|
| Test (tall can, red cube; 30 boxes) | 30/30 found, 4 extra boxes | 30/30 found, 6 extra boxes |
| Hands (60 frames) | 17 false detections | 0 |
| Mug (20 frames) | 34 false detections | 0 |
| Egg (60 frames) | 54 detected as `can` | 60 detected as `ball` |
| Open Images val mAP50, near / far | 0.028 / 0.048 | 0.593 / 0.318 |

Reading:

- Fine-tuning on D405 alone destroys the Open Images classes and produces many false detections.
  Keeping Open Images in training (mix) avoids both: the first measured argument for mixing.
- The mix model loses accuracy on Open Images compared with its start point (for example
  `door_handle` AP50 0.698 to 0.519 *log*), probably because of the learning rate (section 7).
- The D405 test is far too small for a verdict: one object per class, captures at about 20-35 cm.
  The 100 % is indicative only.
- Far mAP50 (0.318) is about half the near value: small objects at 0.5-1 m are the hard case.

Live tests (laptop CPU, D405 on USB 3): the D405-only model detected a can it had never seen
(Red Bull) but also a face as `can` (0.80). The mix model detected `can`, `pen_pencil`, `box` and
`door_handle` in one frame at about 45-60 cm, with low confidences (0.40-0.42) on two of them. These
were not recorded in [`live_tests.md`](../training/results/live_tests.md).

## 6. D405 dataset notes

- Source: Dataset Studio sessions exported to Hugging Face. Only approved frames are used.
- Studio class IDs differ between sessions (3 is both `can` and `human-hand`); classes are matched
  **by name** in `studio_format.py` (`can`, `box`, `ball-pen` to `pen_pencil`).
- Split by whole session; test holds objects absent from train and val (`can_7`, `test-box-2`).
- 10 approved frames are excluded after visual control (`EXCLUDED_FRAMES` in `studio_format.py`).
- Sessions with hands, mug and egg are used only to count false detections.
- Limits: no `ball`, `bottle` or `door_handle`; one or two objects per class; mostly one white
  table; captured at about 20-35 cm (measured from depth), outside the required 0.5-1 m.
- `session.json` names no camera model; intrinsics are fx about 393 px at 640x480.

## 7. Pitfalls already met

| Pitfall | Effect | What to do |
|---|---|---|
| Fine-tuning a trained model with `optimizer=auto` | AdamW lr 0.001, results degrade after epoch 1 | Use a lower lr (about 0.0002) and no early stop. `train.py` has no `--lr0`/`--optimizer` yet: add them |
| `EXTRA_ARGS` set twice in the Colab cell | The last one wins, so the wrong model trains | Keep one line; check `YOLO<x> summary` in the log |
| Same `RUN_NAME` twice | Ultralytics writes `<name>2` | Use a new name |
| Old zip still on the Colab disk | Section 4 reuses `/content/<zip>` | Start a new session, or use new zip names |
| Drive path | The notebook defaults to `MyDrive/npulse` | Edit `DRIVE` in section 2 |
| Drive download | Unzips as `run/run/weights/...` | Flatten to `training/runs/detect/<run>/weights/best.pt` |
| `eval.py` image size | Must match training | It now reads the run's `imgsz`; pass `--imgsz` only on purpose |
| Open Images `Door handle` | Mostly knobs, not lever handles; no `Pencil` class | Lever handles and pencils must come from D405 data |
| Python environment | The team venv lived in `N-pulse/Fresh-Branch/.venv-win` | Recreate one: `training/README.md`, section 1 |
| Camera error `stop() cannot be called before start()` | The real cause is earlier in the log | Replug the D405 on USB 3, close other apps using it |

## 8. Next steps, by priority

1. **Capture D405 data at 0.5-1 m** with all six classes, several objects per class, other
   backgrounds, and frames with no target object (hands, people, mugs). This is the main lever.
2. **Freeze a D405 test set** of whole sessions with unseen objects, then rerun `eval_d405.py`
   (add `b1_yolo26s_d1` as the no-fine-tuning reference).
3. **Retrain the mix model** with a lower learning rate (`b2_yolo26s_mix2`), then compare.
4. **Connect YOLO boxes to the geometry**: per-box depth points, table removal, `fit_once`, then
   centre and dimensions; add a sphere fitter for `ball`.
5. **Measure on the UNO Q**: export to ONNX, check RAM and latency. YOLO26s may be too slow; keep
   YOLO26n or a lower resolution as the fallback.
6. Optional: finish the Open Images benchmark (`training/README.md`, section 9: far, large and
   640 runs, second seed).

`training/README.md` still describes YOLO26n and `MyDrive/npulse`; update it when the model choice
is final.
