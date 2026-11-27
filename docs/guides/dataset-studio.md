# Dataset Studio guide

Dataset Studio is the local RGB-D workflow for object segmentation and model testing:

```text
Capture → Sessions → Preprocess → Annotate → Review → Dataset → Train → Live Test
                                                                     │
                                                                     └─ failure cases return to Sessions
```

## Start the application

Activate the project environment from the repository root:

```bash
# macOS / Linux
source .venv/bin/activate
python tools/dataset_web.py
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python tools\dataset_web.py
```

Open <http://127.0.0.1:8765>. Stop the server with `Ctrl+C`.

Use the simulator when no camera is available:

```bash
python tools/dataset_web.py --source sim
```

For a D405, use a USB 3 port and close RealSense Viewer or any other process using the camera. On macOS, if the locally built librealsense binding can access the camera only with elevated permissions, preserve the active environment:

```bash
sudo -E "$(which python)" -u tools/dataset_web.py
```

## Capture

1. Enter a unique session name such as `box_normal_light_001`.
2. Set the target frame count and save interval.
3. Set the valid depth range.
4. Move the object slowly through different orientations, distances, and image positions.
5. Start capture and wait for completion.

Every saved frame contains the original RGB image, aligned float32 depth, camera intrinsics, and metadata. Near-duplicate consecutive frames are skipped.

## Sessions

- Open a session to inspect every RGB frame.
- Delete only sessions that are no longer needed.
- A protected session cannot be deleted while downstream data still references it.

Raw RGB-D is the only non-reproducible source. Derived data can be recreated.

## Preprocess

1. Select a raw session and give the recipe a version name.
2. Set near/far depth limits.
3. Enable workspace ROI and table removal only when needed.
4. Run the recipe and inspect Raw, Filtered, and Mask for every frame.

Depth filtering is always active. ROI and table removal are independent. Table estimation falls back from a session plane to a per-frame/previous plane and finally to depth + ROI only; failed plane fitting never deletes a raw frame.

Delete a preprocessing version to remove only reproducible derived images, masks, and configuration.

## Annotate

1. Select a session and preprocessing version, or choose **No preprocessing · Raw RGB-D**.
2. Select the proposal method.
3. Set class ID/name, minimum area, and maximum instances.
4. Generate annotations.

Annotations reference the raw RGB instead of copying it. Regenerating pending proposals preserves approved and rejected decisions. Raw mode can use SAM proposals or blank annotations for manual drawing.

**SAM 2.1** (default) and **MobileSAM** draw the object outline inside each depth proposal, or propose masks on the raw RGB. SAM 2.1 tiny was more accurate on thin objects (door handles, pens) in a test on approved D405 sessions: 18/18 hand-drawn objects at IoU ≥ 0.8, against 15/18 for MobileSAM, at about 0.9 s instead of 0.5 s per object on a laptop CPU. Both read their weights from `<data root>/models/annotation/`: `sam2.1_t.pt` (78 MB) is downloaded there automatically on first use; `mobile_sam.pt` must already be there.

## Review

For every frame:

- select a mask and correct its class;
- delete false objects;
- draw missing polygons;
- approve a correct frame or reject an unusable one.

Only approved annotations are exported.

## Dataset export

1. Select approved sessions.
2. assign complete sessions to train, validation, or test;
3. validate and fix every error;
4. choose a unique export version;
5. export the YOLO segmentation dataset.

Keep adjacent frames from one session in the same split to prevent data leakage. Same-disk exports use hard links when possible.

## Train

1. Select a validated export.
2. Choose a unique run name and segmentation checkpoint.
3. Set epochs, image size, batch size, and compute device.
4. Start training and monitor the logs.

A completed run contains its configuration, metrics, plots, and `weights/best.pt`.

## Live Test

1. Connect the D405.
2. Select a registered model or completed run.
3. Choose device, image size, FPS limit, and confidence.
4. Start the test.

The page displays predictions, confidence, latency, centre depth, and camera-frame XYZ. Only the newest camera frame is processed, so old frames do not build a latency queue.

When a prediction is wrong, save a labelled failure case. The system stores RGB, depth, intrinsics, prediction summary, error type, and note in a deduplicated session that can re-enter the normal workflow.

## Storage and deletion

The default shared data root is the sibling directory `../ARM-SOFT-data`:

```text
ARM-SOFT-data/
├── raw/object/
├── derived/object/
├── annotations/object_segmentation/
├── exports/object_segmentation/
├── runs/object_segmentation/
├── models/runtime/
└── catalog/
```

Override it with `ARM_SOFT_DATA_ROOT` or `--data-root`. Never use the filesystem root or home directory as the data root.

Recommended deletion order:

```text
training run → dataset export → annotation → preprocessing version → raw session
```

Dataset Studio listens on `127.0.0.1` and has no authentication. Do not use `--allow-remote` on an untrusted network.

## Troubleshooting

| Problem | Action |
| --- | --- |
| Browser shows connection refused | Confirm the terminal server is still running and open `http://127.0.0.1:8765` in the browser |
| No RealSense device | Use USB 3, close other camera programs, reconnect the D405, and verify `pyrealsense2`/librealsense |
| Old page or HTML returned instead of JSON | Stop the old server, restart it, then hard-refresh the browser |
| Train has no dataset | Complete and validate a Dataset export first |
| Live Test has no model | Finish a training run or register a runtime model |
| Permission denied | Repair ownership of data created by an older sudo session; current sudo writes restore the invoking user when possible |
