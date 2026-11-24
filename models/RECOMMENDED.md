# Which model should I use?

## ★ Best model so far: `yolo26s-mix-openimages-d405-v2.0.0` (candidate)

```text
models/catalog/yolo26s-mix-openimages-d405-v2.0.0/best.pt
```

In Dataset Studio → **Live Test**, select **★ YOLO26s Open Images + D405 v2 (candidate)**.

| Item | Value |
|---|---|
| Classes | ball, can, box, bottle, pen_pencil, door_handle |
| Trained on | `openimages6_d405_v2` (Open Images + HF `npulseDataCV/npulse_cv@5a6e9a3`) |
| Started from | v1.0.0 |
| Val (v2 split) | mAP50 0.808 · mAP50-95 0.617 · P 0.817 · R 0.723 |
| Recommended settings | imgsz 416, conf 0.25 |
| Status | **candidate**: not yet compared with v1.0.0 on the same split, nor tested live at 0.5–1 m |

Details and known limits are in the [model README](catalog/yolo26s-mix-openimages-d405-v2.0.0/README.md).

Standalone live window:

```bash
python training/live_test.py --model models/catalog/yolo26s-mix-openimages-d405-v2.0.0/best.pt --conf 0.25
```

On macOS the camera needs root, so start Dataset Studio with
`sudo -E "$(which python)" -u tools/dataset_web.py`.

## All kept versions

| Version | Folder | Data | Val mAP50 | Status |
|---|---|---|---:|---|
| **v2.0.0** ★ | `catalog/yolo26s-mix-openimages-d405-v2.0.0/` | openimages6_d405_v2 | 0.808 (v2 val) | candidate |
| v1.0.0 | `catalog/yolo26s-mix-openimages-d405-v1.0.0/` | Open Images + D405 v1 | 0.606 (v1 val) | experimental; the run stopped early, best epoch 1 |

The two val numbers come from different val sets, so they cannot be compared directly.

Full training records (curves, `results.csv`, confusion matrices) live in
`training/runs/detect/<run>/`. They are git-ignored:

- `b3_yolo26s_mix_v3_lr2e4`: the source of v2.0.0, a complete run of 50 epochs
- `b3_yolo26s_mix_v2_lr2e4`: the same experiment, interrupted at epoch 36; kept for reference only

Older Colab experiments (iter1, b1 model comparison, b2 D405-only) are no longer
kept as models. Their metrics remain in `training/results/` and `training/decisions.md`.

## Promoting a new best model

1. Evaluate the new model and the current best on the **same** split with `training/eval.py`.
2. Add a new folder under `catalog/` with `best.pt`, `model.json` (including sha256) and `README.md`.
   Never replace a `best.pt` in an existing version.
3. Move the ★ to the new model in its `display_name` and remove it from the previous one.
4. Update this page.
