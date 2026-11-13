# Reproduce the Open Images + Hugging Face D405 dataset

This guide rebuilds the exact YOLO detection dataset used by the team. The pipeline downloads both sources, converts their annotations, combines them, validates the result and records its provenance.

The pinned v1 definition is [`configs/datasets/openimages_d405_v1.yaml`](../configs/datasets/openimages_d405_v1.yaml). It fixes:

- class order: `ball`, `can`, `box`, `bottle`, `pen_pencil`, `door_handle`;
- 500 Open Images images per class and seed `0`;
- Hugging Face repository and commit revision;
- D405 train/validation/test sessions and excluded frames;
- D405 repeat factor used in the mixed training split.

Do not edit the v1 config to create a different dataset. Copy it to a new version instead.

## 1. Get the code

```bash
git clone https://github.com/N-Pulse/ARM-SOFT-Grasping.git
cd ARM-SOFT-Grasping
git switch main
git pull --ff-only origin main
```

Activate the project's normal environment:

```bash
# macOS / Linux
bash setup_env.sh
source .venv/bin/activate
```

```powershell
# Windows PowerShell, if the environment does not exist yet
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## 2. Get Hugging Face access

1. Open the `DouaaKDR/Npulse` dataset page on Hugging Face.
2. Request access and wait until it is approved.
3. Create a fine-grained Hugging Face token with permission to read gated repositories.
4. Install the client and log in:

```bash
python -m pip install huggingface_hub
hf auth login
```

If an older Hugging Face installation does not provide `hf`, use `huggingface-cli login`.
Never commit or share the token.

## 3. Create the separate Open Images environment

FiftyOne is intentionally kept outside the main environment because its OpenCV dependencies can conflict with the camera/web application.

```bash
# macOS / Linux
python3 -m venv training/.venv-download
training/.venv-download/bin/python -m pip install --upgrade pip
training/.venv-download/bin/python -m pip install fiftyone pyyaml
```

```powershell
# Windows PowerShell
py -3.11 -m venv training/.venv-download
.\training\.venv-download\Scripts\python.exe -m pip install --upgrade pip
.\training\.venv-download\Scripts\python.exe -m pip install fiftyone pyyaml
```

## 4. Check the environment

Run this from the repository root with the main `.venv` activated:

```bash
python training/data/prepare_dataset.py check
```

Resolve every reported missing dependency or access problem before continuing.

## 5. Run a small reproducibility test

```bash
python training/data/prepare_dataset.py all --per-class 50
```

This creates separate `_pc50` sources, datasets and YAML files. It does not overwrite or mix with the full v1 dataset.

Validate it again without downloading:

```bash
python training/data/prepare_dataset.py validate --per-class 50
```

## 6. Build the full pinned dataset

```bash
python training/data/prepare_dataset.py all
```

To build a dataset specifically for copying or zipping to another machine, use real copies instead of hard links:

```bash
python training/data/prepare_dataset.py all --link-mode copy
```

The command performs these stages:

1. download and convert Open Images;
2. download the pinned Hugging Face snapshot;
3. build the D405-only dataset;
4. build the mixed Open Images + D405 dataset;
5. validate images, labels, class IDs, coordinates, split separation and decodability;
6. publish both datasets only after validation succeeds.

## 7. Verify the result

```bash
python training/data/prepare_dataset.py validate
```

The verified v1 build should contain:

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| `d405_v1` | 122 | 28 | 30 |
| `openimages6_d405_v1` | 2686 | 473 | 445 |

Main outputs:

```text
training/datasets/d405_v1/
training/datasets/openimages6_d405_v1/
training/data/openimages6_d405_v1.yaml
training/sources/npulse-ac4f808/
```

Each published dataset includes `dataset_manifest.json`, `validation_report.json` and `stats.json`. These record what was downloaded, how it was built and whether validation passed.

All downloaded and generated data is ignored by Git.

## 8. Safe reruns and recovery

- Normal reruns reuse matching downloads.
- `--refresh-sources` downloads the sources again.
- Existing published datasets are never overwritten automatically.
- `--force` archives the current outputs under `training/datasets/.previous/` before replacement.
- A failed build is not published; its reports remain under `training/datasets/.failed/`.

Examples:

```bash
# Re-download both sources and rebuild, archiving the existing output
python training/data/prepare_dataset.py all --refresh-sources --force

# Execute stages separately for troubleshooting
python training/data/prepare_dataset.py download-openimages
python training/data/prepare_dataset.py download-hf
python training/data/prepare_dataset.py build
python training/data/prepare_dataset.py validate
```

## 9. Create a new dataset version

When the Hugging Face dataset gains new approved D405 sessions:

1. copy `configs/datasets/openimages_d405_v1.yaml` to a v2 filename;
2. change `name`, both output names, generated YAML name, HF revision and HF output folder;
3. add every new session to exactly one entry in `d405.session_splits`;
4. extend `class_map` if Dataset Studio contains new class names;
5. build with the new config:

```bash
python training/data/prepare_dataset.py all \
  --config configs/datasets/openimages_d405_v2.yaml
```

Never point two config revisions at the same Hugging Face output directory.

## Common problems

| Problem | Action |
| --- | --- |
| Hugging Face returns 401/403 | Confirm access approval, run `hf auth login` again and check token permissions |
| FiftyOne is missing | Install it inside `training/.venv-download`, not the main `.venv` |
| Existing source does not match the config | Use `--refresh-sources` |
| Existing dataset blocks publication | Inspect it first; use `--force` only when replacement is intended |
| Validation fails | Read both `validation_report.json` files in the printed `.failed/<time>/` directory |
| Disk use is high | Keep the default hard-link mode locally; use copy mode only when portability requires it |

