# YOLO26s Open Images + D405 v1

This is the team's first mixed Open Images/D405 object-detection checkpoint
from run `b2_yolo26s_mix`. It detects `ball`, `can`, `box`, `bottle`,
`pen_pencil`, and `door_handle`.

Status: **experimental**. The training run stopped early with its best result
at epoch 1, so use Live Test to evaluate it on representative D405 scenes and
save failure cases before treating it as a deployment model. The preliminary
comparison in `training/results/benchmark_preliminary_colab.md` describes the
model-comparison work, but is not a final validation report for this exact
checkpoint.

Dataset Studio discovers this folder automatically. Open **Live Test**, select
**YOLO26s Open Images + D405 v1**, connect the camera, and start inference.

SHA-256: `2bcfd146ee2e34c13b65f94e54a548be46569f5aa9adc83c3380df9c04790d4e`
