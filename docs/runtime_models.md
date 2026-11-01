# Runtime model registry

Dataset Studio discovers deployable models from the shared data root rather
than storing large weights in Git:

```text
ARM-SOFT-data/models/runtime/<model-id>/
├── best.pt
└── model.json
```

Register a completed Ultralytics segmentation, detection, or pose checkpoint:

```bash
python tools/register_runtime_model.py \
  --weights /path/to/best.pt \
  --name mechanical-hand-pose-v1 \
  --display-name "Mechanical Hand Pose v1" \
  --task pose \
  --dataset mechanical-hand-dataset-v1 \
  --version 1.0.0 \
  --author "team member"
```

The tool loads the checkpoint, verifies its task, reads classes and pose shape,
computes SHA-256, refuses to overwrite an existing id, then copies the weights
and writes `model.json` atomically. Restart Dataset Studio and the model appears
in **Live Test** together with completed local training runs.

## Sharing models

Publish `best.pt` as a GitHub Release asset (preferred) or another team-managed
artifact. Commit only evaluation results, dataset/model version, checksum and
download instructions. Do not add changing `.pt` files to ordinary Git history.

Before accepting a team model, record at least:

- task and class/keypoint definitions;
- dataset version and session-level split;
- validation/test metrics and representative failure cases;
- image size and recommended confidence;
- checkpoint SHA-256 and author;
- tested devices and observed latency.

Deleting a runtime model means removing its dedicated folder from the shared
data root. It never deletes raw captures, annotations, datasets, or training
runs.
