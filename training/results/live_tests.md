# Live tests — Intel RealSense D405

Protocol: camera fixed and looking at the table; objects on the table at
30 / 50 / 70 cm (read the on-screen depth). For each row press `s` to save a
snapshot (training/results/live/, git-ignored; copy the good ones to the slides).
"OK" = every object boxed with the right class, and no extra box.

The 4 scenes (always the same physical objects, so the models are comparable):
1. **red object**: a single red can (regression check vs the old HSV pipeline)
2. **non-red object**: a single non-red bottle (or ball)
3. **all four together**: ball + can + box + bottle, separated
4. **overlapping**: two or three objects touching / partly hiding each other

Commands:

    python training/live_test.py --model yolo26n.pt --classes "sports ball,bottle,cup"   # COCO baseline
    python training/live_test.py --model training/runs/detect/<best>/weights/best.pt     # fine-tuned

## Status

- **TODO: needs camera** — COCO baseline (yolo26n, 4 scenes × 30/50/70 cm). The D405 was not available on 2026-10-10.
- **TODO: needs camera** — fine-tuned best model (same 4 scenes × 30/50/70 cm).
- Done without camera (2026-10-10): `live_test.py --source temp-sim-res/approach_cuboid.gif --max-frames 30` runs end to end (30 frames, no error); depth helper checked by hand.

## Results

| date | model | scene | objects (colour) | 30 cm | 50 cm | 70 cm | FPS | notes / errors |
|---|---|---|---|---|---|---|---|---|
