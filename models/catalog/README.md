# Bundled model catalog

Each subfolder is one immutable, reviewable model version. Dataset Studio
automatically lists valid entries on the **Live Test** page.

```text
models/catalog/<model-id>/
├── best.pt
├── model.json
└── README.md
```

Never replace `best.pt` inside an existing version. Create a new folder and
increment `model_version` instead. Record the task, classes/keypoints, dataset
version, metrics, known failure cases, author, and SHA-256. Keep experimental
models clearly marked. For future large checkpoints, prefer Git LFS or a
GitHub Release plus the runtime registry described in
`docs/runtime_models.md`; ordinary Git history grows permanently when a
weight is committed.
