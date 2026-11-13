# Continue YOLO training with Google Colab

This tutorial continues training from the team's published YOLO26s checkpoint using the validated `openimages6_d405_v1` dataset. The provided notebook checks the GPU, uploaded files, class order, image/label counts and dataset validation report before training starts.

## What you need

- the latest `main` branch;
- a completed full dataset build from [the dataset reproduction guide](reproduce_training_dataset.md);
- access to Google Drive and Google Colab;
- the team checkpoint:
  `models/catalog/yolo26s-mix-openimages-d405-v1.0.0/best.pt`.

The default notebook configuration uses:

| Setting | Default |
| --- | --- |
| Starting model | YOLO26s Open Images + D405 v1 |
| Dataset | `openimages6_d405_v1` |
| Epochs | 50 |
| Image size | 416 |
| Batch size | 32 |
| Optimizer | AdamW |
| Initial learning rate | 0.0002 |
| Output | `MyDrive/npulse/runs/<RUN_NAME>/` |

## 1. Prepare the upload files locally

Run from the repository root:

```bash
source .venv/bin/activate   # macOS/Linux
bash training/colab/make_code_zip.sh
bash training/colab/make_data_zip.sh openimages6_d405_v1
```

On Windows, run the same scripts in Git Bash. The outputs are:

```text
training/colab/training_code.zip
training/colab/openimages6_d405_v1.zip
```

The code archive excludes datasets, runs and model weights. The dataset archive excludes Ultralytics cache files containing local absolute paths.

## 2. Upload files to Google Drive

Create this exact layout:

```text
MyDrive/
└── npulse/
    ├── training_code.zip
    ├── data/
    │   └── openimages6_d405_v1.zip
    ├── models/
    │   └── yolo26s-mix-openimages-d405-v1.0.0/
    │       └── best.pt
    └── runs/
```

Upload:

- `training/colab/training_code.zip` to `MyDrive/npulse/`;
- `training/colab/openimages6_d405_v1.zip` to `MyDrive/npulse/data/`;
- `models/catalog/yolo26s-mix-openimages-d405-v1.0.0/best.pt` to the model path above.

Uploading `training/data/openimages6_d405_v1.yaml` is optional. The notebook recreates the fixed six-class YAML when it is absent.

## 3. Open the notebook

1. Upload `training/colab/colab_train.ipynb` to Colab, or open it from the cloned repository.
2. In Colab choose **Runtime → Change runtime type → GPU**.
3. Use an NVIDIA GPU runtime; the notebook deliberately stops if CUDA is unavailable.

## 4. Edit the parameter cell

For a normal run, edit only the first code cell.

Always choose a new descriptive `RUN_NAME`, for example:

```python
RUN_NAME = "b3_yolo26s_mix_continue_lr2e4_teamname"
```

Keep these values for the standard continuation experiment:

```python
DATASET_NAME = "openimages6_d405_v1"
DATA_ZIPS = [f"{DATASET_NAME}.zip"]
START_MODEL_DRIVE = "models/yolo26s-mix-openimages-d405-v1.0.0/best.pt"
EPOCHS = 50
IMAGE_SIZE = 416
BATCH_SIZE = 32
PATIENCE = 50
OPTIMIZER = "AdamW"
LEARNING_RATE = 0.0002
```

If a T4 runs out of memory, reduce `BATCH_SIZE` to `16` or `8`. Do not change the dataset classes or their order.

`REFRESH_DATA = False` reuses an already extracted Drive dataset. Set it to `True` only after replacing the dataset ZIP.

## 5. Run the notebook

Choose **Runtime → Run all**. The notebook will:

1. install the training dependencies;
2. mount Google Drive;
3. require an available CUDA GPU;
4. verify all Drive inputs;
5. extract a clean copy of the code;
6. extract or reuse the dataset;
7. verify class names, split counts and validation metadata;
8. connect the training output directory directly to Drive;
9. launch training with explicit arguments;
10. display the saved configuration, metrics and plots.

Do not close the Drive authorization prompt. The first setup and extraction cells may take several minutes without displaying continuous output.

## 6. Understand where results are saved

Every epoch is saved directly to:

```text
MyDrive/npulse/runs/<RUN_NAME>/
```

Important files:

```text
weights/best.pt   best validation checkpoint
weights/last.pt   latest checkpoint, used to resume
args.yaml         actual training parameters
results.csv       per-epoch metrics
results.png       training curves
```

Because outputs are linked to Drive before training starts, completed epochs survive a Colab disconnect.

The notebook refuses to overwrite an existing `RUN_NAME`. This prevents accidental loss or mixing of experiments.

## 7. Resume an interrupted run

1. Reconnect a GPU runtime.
2. Mount the same Drive.
3. Confirm `MyDrive/npulse/runs/<RUN_NAME>/weights/last.pt` exists.
4. Run the notebook's optional **Resume after a disconnect** cell.

Resume only the same experiment. Do not change the dataset, model architecture or class order while resuming.

## 8. Bring the result back to the repository

Download the complete Drive run folder and place it locally at:

```text
training/runs/detect/<RUN_NAME>/
```

Evaluate the best checkpoint in the team's fixed local environment:

```bash
python training/eval.py \
  --weights training/runs/detect/<RUN_NAME>/weights/best.pt \
  --split val \
  --notes "trained on Colab GPU"

python training/eval.py \
  --weights training/runs/detect/<RUN_NAME>/weights/best.pt \
  --split test \
  --notes "trained on Colab GPU; test for final reporting"
```

Use validation results to choose the best experiment. Keep the test split for final reporting rather than repeated tuning.

## 9. Share a successful model

Do not commit an arbitrary run directory. Share these items with the team:

- `best.pt`;
- `args.yaml`;
- validation and test metrics;
- the exact dataset version;
- the source commit and `RUN_NAME`;
- a short note describing what changed from the previous model.

Register a promoted checkpoint through the project's model catalog workflow so Dataset Studio can discover it consistently.

## Common problems

| Problem | Action |
| --- | --- |
| `No CUDA GPU is available` | Change the Colab runtime type to GPU and reconnect |
| Missing upload error | Reproduce the exact Drive folder layout and filenames above |
| Drive quota or extraction failure | Remove obsolete ZIPs/runs or use another Drive with enough space |
| CUDA out of memory | Lower `BATCH_SIZE` to 16 or 8 |
| Run already exists | Choose a new `RUN_NAME`; do not overwrite an earlier experiment |
| Colab disconnected | Reconnect and resume from the saved `last.pt` |
| Counts or class order mismatch | Rebuild/re-upload the pinned dataset; do not bypass the notebook check |

