# Team Git and feature guide

This document is the shared entry point for contributors. The `main` branch is
the runnable semester baseline; feature branches are short-lived work areas.

## First setup

```bash
git clone git@github.com:N-Pulse/ARM-SOFT-Grasping.git
cd ARM-SOFT-Grasping
git switch main
git pull --ff-only
```

Create a fresh environment on every computer. Never copy `.venv` between
macOS, Windows, and Linux.

```bash
# macOS/Linux
bash setup_env.sh
source .venv/bin/activate
python tools/dataset_web.py
```

On Windows PowerShell, create and activate `.venv`, install
`requirements.txt` and `pyrealsense2`, then run
`python tools\dataset_web.py`. Open `http://127.0.0.1:8765`.

## Feature map

| Feature | Web page / command | Main implementation |
| --- | --- | --- |
| RGB-D capture and deduplication | Capture | `tools/dataset_web.py`, `armsoft/data/` |
| Session inspection/deletion | Sessions | shared `ARM-SOFT-data/raw/` |
| Depth, ROI, optional table filtering | Preprocess | `armsoft/preprocessing/` |
| MobileSAM proposals | Annotate | `armsoft/object_segmentation/` |
| Rename, add, remove, approve masks | Review | annotation JSON under shared data root |
| YOLO dataset validation/export | Dataset | `armsoft/object_segmentation/export.py` |
| Local YOLO training | Train | `tools/train_object_segmentation.py` |
| Object/pose inference + centre distance/XYZ | Live Test | Dataset Studio live-test engine |
| Failure-case collection | Live Test | normal deduplicated RGB-D sessions |
| Coloured PLY + metadata | Live Test / CLI | `armsoft/pointcloud/`, `tools/export_pointcloud.py` |
| Hand pose and 2-D→3-D fusion | CLI | `tools/run_mechanical_hand_rgbd.py` |
| Model comparison and D405 evaluation | CLI | `training/compare_runs.py`, `training/eval_d405.py` |
| Bundled model versions | Live Test | `models/catalog/` |
| Local/release model installation | Live Test | `tools/register_runtime_model.py` |

Raw data, derived data, annotations, exports, runs, and locally installed
models belong in the sibling `ARM-SOFT-data` directory. They must not be
committed. Dataset Studio stores references/hard links where possible and its
delete checks protect dependent data.

## Daily Git workflow

Start every task from current `main`:

```bash
git switch main
git pull --ff-only
git switch -c feature/short-description
```

Commit code, tests, and documentation together, then push and open a pull
request:

```bash
git add <only-the-files-you-changed>
git commit -m "feat: describe the outcome"
git push -u origin feature/short-description
```

The pull request must target `main`. Before merge, update from `main`, run the
relevant tests, and resolve conflicts on the feature branch. After merge,
delete the feature branch. Use `experiment/...` for uncertain research and
`fix/...` for focused fixes. Never force-push `main`, commit `.venv`, datasets,
camera recordings, `runs/`, secrets, or generated caches.

## Model versioning

Bundled release models use an immutable folder:

```text
models/catalog/<descriptive-id>-vX.Y.Z/
├── best.pt
├── model.json
└── README.md
```

Never overwrite an existing weight. A retrain is a new version, even when the
architecture is unchanged. `model.json` records task, classes/keypoints,
dataset version, semantic model version, status, and SHA-256. The README must
record metrics, test conditions, limitations, and known failure cases.

The repository currently bundles one ~20 MB experimental model so the whole
team can reproduce Live Test immediately. For future or larger weights, upload
the checkpoint as a GitHub Release asset or use Git LFS, then install it into
`ARM-SOFT-data/models/runtime/` with `tools/register_runtime_model.py`. This
keeps ordinary Git clones small while preserving every model version.

## Before merging to `main`

- Run `python -m pytest -q armsoft/tests training/tests`.
- Confirm Dataset Studio starts and its affected page works.
- Add or update tests for behavior changes.
- Update this guide or the feature-specific guide when commands change.
- Verify `git status` contains no datasets, weights without metadata, secrets,
  duplicate Finder files, or machine-specific paths.
- State the model/dataset version and known limitations in the pull request.

Older branches are references, not alternate products. If a feature exists in
`main`, teammates should use it there and create their next branch from `main`.
