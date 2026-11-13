# Dataset pipeline (Open Images + Hugging Face D405): handover

Scope: only the training-data download/build feature. For the detector itself see
`docs/object_detection_handover.md`.

## Status

- Merged into `main` through PR #22 (`db3ef61`).
- Main implementation commits: `11408ab`, `8f7fbd0`, `3078602`, `71372cd`.
- Teammate instructions: [`reproduce_training_dataset.md`](reproduce_training_dataset.md)
  and [`colab_training_tutorial.md`](colab_training_tutorial.md).
- Verified on a Mac (2026-11-12): full build valid, 0 corrupt images.
  Mixed `openimages6_d405_v1`: train 2686 / val 473 / test 445 images.
  `d405_v1`: train 122 / val 28 / test 30 frames (190 approved - 10 excluded).

## Before this branch

- `training/data/download_openimages.py`: FiftyOne download of 6 classes, YOLO conversion,
  seeded train/val/test split, drops IsGroupOf/IsDepiction boxes.
- `training/data/build_d405_datasets.py`: read a **manually downloaded** HF Dataset Studio
  folder (`../ARM-SOFT-data-douaa`), approved frames only, class mapping by name, whole-session
  splits, 10 excluded frames, D405 train copied 5x into the mixed set.
- Gaps: manual HF download, two separate commands, splits/revision hard-coded, no dataset
  validation, outputs deleted and rebuilt in place, 5 real copies, no provenance record.

## What this branch adds

| File | Role |
|---|---|
| `configs/datasets/openimages_d405_v1.yaml` | pins classes, Open Images count/seed, HF repo + revision `ac4f808`, session splits, excluded frames, class map, repeat factor |
| `training/data/prepare_dataset.py` | `check`, `download-openimages`, `download-hf`, `build`, `validate`, `all` |
| `training/data/validate_yolo_dataset.py` | labels/images match, class ids, coordinates, zero-area boxes, image decoding, cross-split duplicate hashes, empty splits |
| `training/data/build_d405_datasets.py`, `studio_format.py` | now callable functions; splits/exclusions/class map can come from the config; hard-link option |
| `training/tests/test_prepare_dataset.py`, `test_validate_yolo_dataset.py`, `_png.py` | end-to-end build on fake data, rollback, quick-run naming, mocked downloads |

Behaviour to know:

- Build happens in `training/datasets/.staging/`; published only if validation passes.
  Failures keep reports in `training/datasets/.failed/<time>/`.
- No overwrite without `--force`; replaced versions go to `training/datasets/.previous/`.
  Both datasets are published together or rolled back.
- `--per-class N` (different from the config) is a quick run: every output gets `_pcN`.
- Sources are reused only if their `DOWNLOAD.json` matches the config (`--refresh-sources`
  to force). `dataset_manifest.json` records git revision and the settings actually used.
- D405 repeats are hard links; use `--link-mode copy` before zipping for Colab.
- `training/colab/colab_train.ipynb` is preconfigured for the versioned mixed dataset and the
  published catalog checkpoint. It validates GPU, Drive inputs, split counts, class order and the
  packaged validation report before starting, then saves each epoch under `MyDrive/npulse/runs/`.

## Data locations

| What | Where |
|---|---|
| Open Images (converted) / FiftyOne cache | `training/datasets/openimages6/`, `training/datasets/_fiftyone_cache/` |
| HF snapshot | `training/sources/npulse-ac4f808/` (one folder per revision; never reuse a folder for another revision) |
| Datasets | `training/datasets/d405_v1/`, `training/datasets/openimages6_d405_v1/` |
| Train yaml (generated) | `training/data/openimages6_d405_v1.yaml` |
| New captures before upload | `../ARM-SOFT-data/raw/object/<session>/`, `../ARM-SOFT-data/annotations/object_segmentation/<session>/` |

All of these are git-ignored.

## Access

`DouaaKDR/Npulse` is gated: request access on the dataset page, wait for approval, then
`hf auth login`. A fine-grained token needs read access to gated repos.

## Adding new D405 data (v2)

Teammates upload new sessions to HF -> new revision. Copy the config to
`openimages_d405_v2.yaml`, change `name`, output names, `huggingface.revision`,
`huggingface.output`, add the new sessions to `d405.session_splits` (and `class_map` if new
Studio class names), then `prepare_dataset.py all --config configs/datasets/openimages_d405_v2.yaml`.

## Known gaps

- No script to upload `ARM-SOFT-data` sessions to HF; done by hand.
- Sessions present in the HF snapshot but missing from `session_splits` are silently ignored.
- D405 data covers only can, box, pen_pencil (pen_pencil only in train); ball, bottle,
  door_handle come from Open Images only.
