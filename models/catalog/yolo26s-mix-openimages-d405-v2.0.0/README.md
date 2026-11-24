# YOLO26s Open Images + D405 v2

Status: **candidate**. Detects `ball`, `can`, `box`, `bottle`, `pen_pencil` and
`door_handle`.

This checkpoint continues the team checkpoint `yolo26s-mix-openimages-d405-v1.0.0`
on the v2 mixed dataset. Dataset Studio lists this folder automatically in **Live Test**.

## Training

| Item | Value |
|---|---|
| Source run | `b3_yolo26s_mix_v3_lr2e4` (Google Colab, T4 GPU) |
| Start weights | `models/catalog/yolo26s-mix-openimages-d405-v1.0.0/best.pt` |
| Dataset | `configs/datasets/openimages_d405_v2.yaml` → `openimages6_d405_v2` |
| Hugging Face source | `npulseDataCV/npulse_cv` @ `5a6e9a3` (top level and `yanis/`) |
| Train / val / test images | 3306 / 658 / 445 (mixed); D405 v2: 615 / 213 / 184 |
| Settings | 50 epochs, imgsz 416, batch 32, AdamW, lr0 0.0002, patience 50, seed 0 |
| Kept checkpoint | best epoch **28** of 50 |

New in the v2 data: the first D405 `door_handle` and `ball` (marble) frames, more
`can`, `box` and pen sessions, and hands, a prosthetic hand and eggs as negative frames
(empty labels). `marble` counts as `ball` and `yellow_marker` counts as `pen_pencil`.

## Validation (v2 val split, end of training)

| mAP50 | mAP50-95 | Precision | Recall |
|---:|---:|---:|---:|
| 0.808 | 0.617 | 0.817 | 0.723 |

Normalised confusion matrix (share of true boxes found with the right class):
ball 0.92, can 0.95, box **0.78**, bottle 0.92, pen_pencil 0.96, door_handle 0.94.
Most misses are on `box` (17 % missed, 5 % predicted as `door_handle`). Most
background false positives are predicted as `bottle`, `ball` and `box`.

**Not comparable with v1.0.0.** The v1.0.0 value (0.606) comes from the smaller v1
val set. The v2 val set contains more D405 frames, and these frames are easier.
Evaluate both checkpoints on the same split before you promote this model:

```bash
python training/eval.py --weights models/catalog/yolo26s-mix-openimages-d405-v1.0.0/best.pt \
  --data training/data/openimages6_d405_v2.yaml --split val --tag v1_on_v2val
python training/eval.py --weights models/catalog/yolo26s-mix-openimages-d405-v2.0.0/best.pt \
  --data training/data/openimages6_d405_v2.yaml --split val --tag v2_on_v2val
```

## Known limits

- The D405 data still contains no `bottle`. That class is learned from Open Images only.
- Validation mAP stopped improving after epoch 28 while the training loss kept
  falling. This suggests mild overfitting. About 30 epochs are enough for a rerun.
- We do not know which door-handle sessions show the same physical handle. Test
  numbers for `door_handle` may therefore be optimistic.
- Most D405 captures were taken at 20–35 cm. The 0.5–1 m range is not yet validated
  with the camera.
- Training ran only once, with a single seed.

SHA-256: `c9a3701d6dc9bb87d224a8f794cc1633604638f4dd738895485543b89ed7261d`
