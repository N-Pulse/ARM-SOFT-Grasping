# Point Cloud Pipeline

This branch tests a reusable RGB-D to point-cloud stage without changing the
raw capture or the object-segmentation workflow.

```text
RealSense / simulator / captured session
                 │
                 ▼
       aligned RGB + depth + intrinsics
                 │
                 ▼
       Depth range → optional ROI → optional table removal
                 │
                 ▼
       pinhole deprojection (XYZ in metres) + RGB colour
                 │
                 ▼
       voxel downsampling → statistical outlier removal → point cap
                 │
                 ├── optional Open3D live viewer
                 └── PLY + JSON summary + manifest
```

## Coordinate system

Every point is stored in the RealSense camera frame:

- `+X`: camera right;
- `+Y`: camera down;
- `+Z`: forward from the camera;
- unit: metres;
- colour: RGB in the PLY file.

This is still camera-relative geometry. Camera-to-robot calibration will be a
separate transform after this stage is stable.

## Quick tests

Run one synthetic frame without a camera:

```bash
python tools/pointcloud_pipeline.py --source sim --frames 1 \
  --min-depth 0.10 --max-depth 1.50
```

Show a live Open3D window from the D405 and save every tenth frame:

```bash
python tools/pointcloud_pipeline.py --source camera --frames 300 --viz \
  --save-every 10 --min-depth 0.10 --max-depth 1.50 \
  --roi 0.05,0.05,0.95,0.95 --voxel-size 0.005
```

Process an existing captured session reproducibly:

```bash
python tools/pointcloud_pipeline.py \
  --session ../ARM-SOFT-data/raw/object/object_test_003 \
  --frames 100 --out ../ARM-SOFT-data/derived/pointcloud/object_test_003_v1 \
  --min-depth 0.10 --max-depth 1.50 --auto-table \
  --min-table-height 0.015 --voxel-size 0.005
```

Use `--keep-outliers` to inspect raw sensor noise, `--voxel-size 0` to disable
downsampling, or omit `--roi`/`--auto-table` to keep those filters disabled.

Windows PowerShell uses the same Python arguments, with backticks for line
continuation if desired:

```powershell
python tools\pointcloud_pipeline.py --source camera --frames 100 --viz `
  --min-depth 0.10 --max-depth 1.50 --voxel-size 0.005
```

## Output and storage

The default output is under the shared data root:

```text
ARM-SOFT-data/derived/pointcloud/<source>_<timestamp>/
├── config.json
├── manifest.jsonl
├── frame_000000.ply
└── frame_000000.json
```

PLY files are derived artifacts: they can be deleted and regenerated from the
original RGB-D session. `--save-every N`, voxel downsampling, and `--max-points`
limit storage. Raw RGB and depth are referenced, never copied.

## Parameters to tune first

1. Depth range: reject points outside the physical workspace.
2. ROI: limit pixels to the useful camera region.
3. Voxel size: `0.003–0.008 m` is a practical D405 starting range.
4. Table removal: enable only when a stable table plane is visible.
5. Outlier filter: increase `--outlier-std-ratio` if real object edges disappear.

The first evaluation should compare point count, object completeness, noise,
processing time, and PLY size across several camera angles and distances.
