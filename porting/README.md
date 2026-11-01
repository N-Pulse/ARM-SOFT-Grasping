# Pre-UNO Q environment and software porting

Target: **Arduino UNO Q** Linux side — Qualcomm QRB2210, 4× Cortex-A53
(ARMv8.0-A, ~2 GHz), 2/4 GB RAM, Debian 13 (trixie, Python 3.13).
Everything here was done without the board, on an x86-64 host, with an
emulated arm64 Debian that uses the same OS release, Python version and CPU model.

## 0. Reproduce everything, step by step

On a Linux x86-64 machine, from a fresh clone. You don't need root, Docker, a camera or a board.

*Verified 2026-09-29:* `setup_env.sh` + `reproduce.sh` run end to end (exit 0) on a
copy of the tree containing only what git would push (Ubuntu 22.04 host). Step 8
reported 60/60 identical frames between x86 PyTorch+Open3D and the arm64 minimal set.

**Prerequisites (once):**

```bash
sudo apt install python3-venv curl qemu-user-static   # qemu-aarch64-static emulates the ARM CPU
sudo apt install libegl1 libgl1 libglib2.0-0          # Open3D on the host (main README §1 / §13 without root)
bash setup_env.sh                                     # host .venv: torch, ultralytics, open3d, ...
```

**All steps in one go** (about 45 min, mostly emulated `pip install`):

```bash
bash porting/reproduce.sh
TFENV=/path/to/tfenv bash porting/reproduce.sh        # also redo TFLite (TF venv: see export_tflite.py header)
```

**The same steps by hand**, in order (this is exactly what `reproduce.sh` runs):

| # | where | command | produces |
|---|---|---|---|
| 1 | host | `.venv/bin/pip install --target porting/cache/hostpkgs onnx onnxruntime onnxslim ai-edge-litert`, then delete numpy/sympy/mpmath/packaging/typing_extensions from that folder (so they cannot shadow the `.venv` copies) | export tools, `.venv` untouched |
| 2 | host | `PYTHONPATH=porting/cache/hostpkgs .venv/bin/python porting/export_classifier.py --int8` | `models/shape_classifier.onnx`, `_int8.onnx` |
| 2b | TF venv | `tfenv/bin/python porting/export_tflite.py --int8` (optional; the `.tflite` files are committed) | `models/*.tflite` |
| 3 | host | `PYTHON=.venv/bin/python bash porting/record_sim.sh` | `porting/testdata/rec/*` (60 frames, git-ignored) |
| 4 | host | `PYTHONPATH=porting/cache/hostpkgs .venv/bin/python porting/test_classifier.py` and `porting/run_replay.py --o3d real --classifier yolo` / `--o3d lite --classifier onnx` | x86 reference, `results/*x86_64*` |
| 5 | host→arm64 | `bash porting/unoq_env.sh create`, `install full`, `install minimal` | `porting/rootfs-arm64/` (≈4 GB, git-ignored) |
| 6 | arm64 | `bash porting/unoq_env.sh run python porting/check_env.py` | dependency table (§2) |
| 7 | arm64 | `… run python porting/test_classifier.py`; `… run python porting/run_replay.py --o3d real --classifier yolo`; `UNOQ_VENV=minimal … run python porting/run_replay.py --o3d lite --classifier onnx` | `results/*aarch64*` |
| 8 | host | `.venv/bin/python porting/compare_results.py results/replay_x86_64_real_yolo.jsonl results/replay_aarch64_minimal.jsonl` | the headline result: identical grasps |

To explore interactively: `bash porting/unoq_env.sh shell` (add `UNOQ_VENV=minimal` for the target set).
The repo is mounted at `/work` inside the environment.

## 1. The compatibility environment

```bash
bash porting/unoq_env.sh create          # debian:trixie arm64 rootfs + apt packages (~10 min)
bash porting/unoq_env.sh install full    # venv: numpy scipy opencv-headless open3d onnxruntime torch ultralytics
bash porting/unoq_env.sh install minimal # venv: numpy scipy opencv-headless onnxruntime  (UNO Q target set)
bash porting/unoq_env.sh shell           # arm64 shell, repo mounted at /work (UNOQ_VENV=full|minimal)
bash porting/unoq_env.sh run python porting/check_env.py
```

How it works: `debian:trixie` arm64 from Docker Hub, unpacked into
`porting/rootfs-arm64/`, run by **proot 5.4.1** (rootless chroot, fetched and
checksum-verified by the script) with **qemu-aarch64-static** emulating a
**Cortex-A53** (`QEMU_CPU=cortex-a53`). No Docker, no root needed.

Emulation pitfalls found and fixed in the script (these do not affect the real board):

| symptom | cause | fix |
|---|---|---|
| `apt`: *Could not switch saved set-user-ID* | proot cannot setuid | `APT::Sandbox::User "root"` |
| `dpkg`: *required read/write access to /var/lib/dpkg* | Ubuntu's proot 5.1 does not emulate `faccessat2` (trixie glibc uses it) | pinned proot 5.4.1 |
| `import onnxruntime` segfaults; `import torch` aborts (*Can't open MIDR_EL1 sysfs entry*) | the host's x86 `/proc/cpuinfo` and `/sys/devices/system/cpu` leak into the guest | bind a fake 4× A53 `cpuinfo` and CPU sysfs tree (`cpuinfo-cortex-a53`, `fake-sys-cpu/`) |

**Timings under emulation are meaningless** (qemu is 5–20× slower than a real
A53, with a different ratio per library). Use the environment to check compatibility, and
measure speed on the board.

## 2. Dependencies and porting checklist

What the pipeline imports, where, and whether it works on arm64 / Python 3.13:

| library | role | where in the code | aarch64 cp313 wheel | status in emulated UNO Q | host size | verdict |
|---|---|---|---|---|---|---|
| numpy | everything | all of `core/` | yes (2.5.3) | OK | 60 MB | keep |
| scipy | `least_squares` (cylinder fit) — **one call** | `core/shape_fitter.py:_best_fit_cylinder` | yes (1.18.1) | OK | 130 MB | keep (also backs `open3d_lite`); a 3-parameter Gauss–Newton loop in numpy could drop it |
| opencv | HSV mask, morphology, contours, `minAreaRect`, `convexHull`, chessboard | `core/isolation.py`, `core/shape_fitter.py`, `core/table_plane.py`, `sources/artifacts.py` | yes (5.0.0, abi3) | OK | 190 MB | keep; use **opencv-python-headless** (no GUI calls anywhere) |
| open3d | voxel grid, DBSCAN, KNN normals, `LineSet`/cylinder mesh, 3-D viewer | `core/shape_fitter.py` (module-level import), `core/classifier.py` (geometric), `sinks/viewer.py` | yes (0.20.0) | OK after `apt install libidn2-0 libgfortran5 libsm6 libice6` | **903 MB** + dash/flask/plotly/jupyter deps | **replace** — `open3d_lite.py` matches it frame for frame |
| torch + torchvision | YOLO runtime | `core/classifier.py:YoloShapeClassifier` | yes (2.14 / 0.29) | runs, but **wrong outputs with oneDNN** (§4) | **720 MB** | **drop on target**; replaced by ONNX/TFLite |
| ultralytics | YOLO wrapper | `core/classifier.py` | pure Python | imports (pulls polars, matplotlib, GUI opencv) | 7 MB + deps | **drop on target** |
| onnxruntime | new classifier runtime | `core/classifier.py:OnnxShapeClassifier` | yes (1.30.0) | OK | 50 MB | **add** (or use OpenCV DNN: 0 MB extra) |
| ai-edge-litert | alternative TFLite runtime | `porting/test_classifier.py` | yes (2.2.0) | see §3 | 57 MB | optional alternative |
| onnx | export only | `porting/export_classifier.py` | **no cp313 aarch64 wheel** | — | — | host only, never needed on the board |
| pyrealsense2 | live D405 camera | `sources/realsense.py`, `sources/__init__.py` | **no cp311/cp313 aarch64 wheel** (only cp39/310/312) | missing | — | **blocker for a USB camera on the board**: build librealsense from source, or use Python 3.12, or feed frames from elsewhere |
| pyserial | `--serial` sink | `sinks/results.py` | pure Python | — | small | fine (UNO Q MCU link is its own topic, see §5) |
| rclpy | `--ros2` sink | `sinks/results.py` | from ROS 2, not pip | — | — | no ROS 2 binaries for Debian 13 arm64 → do not plan on it |

System packages needed on the board: `python3-venv libgomp1`, plus `libgl1 libegl1 libglib2.0-0t64 libidn2-0 libgfortran5 libsm6 libice6`
only while Open3D is still in use (found with `ldd` on its aarch64 wheel; `libusb-1.0-0` for a camera).

Install footprint: current host venv **2.5 GB**. The target set (numpy, scipy,
opencv-headless, onnxruntime) is about **430 MB**, or about 380 MB using OpenCV DNN instead of onnxruntime.

## 3. Classifier: ONNX and TFLite

```bash
# host (needs torch + ultralytics + onnx; onnx/onnxruntime go in a side dir so the .venv stays untouched)
pip install --target porting/cache/hostpkgs onnx onnxruntime onnxslim
PYTHONPATH=porting/cache/hostpkgs python porting/export_classifier.py --int8
tfenv/bin/python porting/export_tflite.py --int8          # separate TensorFlow venv, see file header
# anywhere
python porting/test_classifier.py --json porting/results/classifier_<arch>.json
python -m armsoft --classifier onnx [--onnx-backend onnxruntime|opencv]
```

Produced in `models/`: `shape_classifier.onnx` (fp32, 5.8 MB),
`shape_classifier_int8.onnx` (QDQ, 1.6 MB), `shape_classifier_float32.tflite`,
`_float16.tflite` (2.9 MB), `_full_integer_quant.tflite` (1.6 MB). Input 1×3×128×128
(TFLite: NHWC), RGB in [0, 1], output = softmax over `{0: cuboid, 1: cylinder}`.

`build_classifier("auto")` now falls back to the ONNX model when ultralytics is
missing, so on the board the default CLI uses ONNX without extra flags.

Parity vs the original PyTorch model, 174 real crops + 24 simulated crops, 4 threads (x86-64 host):

| backend | top-1 agree | accept (≥0.70) agree | max \|Δp\| | acc real | acc sim | ms/crop |
|---|---|---|---|---|---|---|
| torch (reference) | 1.000 | 1.000 | 0 | 1.000 | 0.875 | 4.6 |
| **onnxruntime fp32** | 1.000 | 1.000 | 0.051 | 1.000 | 0.875 | 2.1 |
| OpenCV DNN fp32 | 1.000 | 1.000 | 0.051 | 1.000 | 0.875 | 3.4 |
| onnxruntime int8 | 0.985 | 0.990 | 0.332 | 1.000 | 0.833 | 1.5 |
| **TFLite fp32** | 1.000 | 1.000 | 0.051 | 1.000 | 0.875 | 1.3 |
| TFLite fp16 | 1.000 | 1.000 | 0.051 | 1.000 | 0.875 | 1.4 |
| TFLite int8 | 0.970 | 0.960 | 0.568 | 1.000 | 0.625 | 1.4 |
| ONNX fed torchvision preprocessing | 1.000 | 1.000 | **0.0000** | 1.000 | 0.875 | — |

- The export is exact. All of the remaining 0.05 comes from the numpy/cv2 preprocessing
  (`preprocess_crop`: `INTER_AREA` vs PIL's antialiased bilinear). It never
  changed a decision on these 198 crops, but a crop sitting right at the 0.70 threshold could flip.
- **Use fp32 (or fp16 TFLite).** int8 saves nothing that matters, since fp32 is already about 2 ms, and it
  flips decisions, mostly on the simulated crops.
- OpenCV DNN (5.0) cannot load the QDQ int8 graph.
- onnxruntime's default one-spinning-thread-per-core pool slowed the numpy/OpenCV
  stages running between inferences by about 3×. `OnnxShapeClassifier` now uses 4 threads with
  spinning off.

arm64: see §4. The exported models run unchanged under onnxruntime, OpenCV DNN and LiteRT on the emulated A53; torch there is the outlier.

## 4. Pipeline on recorded RGB-D data

```bash
bash porting/record_sim.sh            # 3 × 20 frames: cylinder, cuboid (yaw 30°), cylinder + 12 worms
python porting/run_replay.py --o3d real --classifier yolo --jsonl a.jsonl
python porting/run_replay.py --o3d lite --classifier onnx --jsonl b.jsonl   # no Open3D, no torch
python porting/compare_results.py a.jsonl b.jsonl
```

Copy real `--record` directories into `porting/testdata/rec/<name>/` and they are
picked up automatically. **There are no real D405 recordings anywhere yet**, so the
sequences above are simulated.

x86-64 host, identical for all four combinations (real/lite Open3D × YOLO/ONNX):

| sequence | ok | stable | shape (cyl/cub) | width | height | jaw |
|---|---|---|---|---|---|---|
| cylinder (60×100 mm) | 20/20 | 13 | 20/0 | 61.2 mm | 99.2 mm | 77.2 mm |
| cuboid | 12/20 | 1 | 0/12 | 128.3 mm | 93.9 mm | 144.3 mm |
| cylinder + worms | 19/20 | 1 | 0/19 | 97.9 mm | 89.6 mm | 113.9 mm |

Frame-by-frame, PyTorch+Open3D vs ONNX+open3d_lite: 60/60 same status, shape
and stability flag. Max difference is 0.000 mm in position, jaw, width and height, and 3e-9 rad
in the joint targets. The shim is a drop-in replacement.

**arm64 (emulated Cortex-A53, Debian 13, Python 3.13), ONNX + open3d_lite:** the
same table exactly. Frame by frame against the x86 PyTorch + Open3D reference it gives 60/60
same status, shape and stability flag, with max difference 0.000 mm and 1.8e-9 rad
(`results/diff_x86yolo_vs_arm_lite_onnx.json`). **This is the target configuration, and it works.**

**arm64 with PyTorch is wrong:** the cylinder is rejected in 20/20 frames, and cuboid ok
drops to 4/20 (`results/replay_aarch64_real_yolo.json`). This comes from torch's aarch64 oneDNN kernels,
not from the model or the data:

| same 20 real crops, same input tensor | max \|Δp\| vs onnxruntime |
|---|---|
| x86 torch | 2.5e-10 |
| arm64 onnxruntime vs x86 onnxruntime | 2.4e-10 |
| arm64 torch, default (oneDNN) | **0.087** |
| arm64 torch, `torch.backends.mkldnn.enabled = False` | 7.4e-10 |

It could be a qemu artifact (oneDNN JIT-generates A53 code) or a real torch bug. That can
only be settled on the board, but it is one more reason not to ship torch there. In
`results/classifier_aarch64.json` every backend's "disagreement" with torch is
therefore torch's own error. The ONNX/TFLite backends agree with each other on arm64 just as on x86.
(The simulated crops in that test are regenerated on each machine and differ slightly
across architectures; the recorded replays above are the controlled comparison.)

Emulated per-frame time is about 1 s (isolation about 540 ms, ONNX classify about 135 ms, torch about 460 ms).
Under qemu these numbers only give relative costs; they are not A53 timings.

## 5. Lightweight replacements investigated

| heavy dependency | replacement | status |
|---|---|---|
| Open3D voxel grid | `np.unique` on integer voxel keys + `np.add.at` | done, `open3d_lite.voxel_down_sample` |
| Open3D DBSCAN | `scipy.spatial.cKDTree.query_ball_point` + BFS, same core-point rule | done, `open3d_lite.dbscan` |
| Open3D KNN normals | `cKDTree.query(k=6)` + batched 3×3 `eigh` | done, `open3d_lite.estimate_normals` |
| Open3D `LineSet` / `create_cylinder` | plain vertex/edge arrays, same 102-vertex layout | done (grasp code only reads the vertices) |
| Open3D viewer (`--viz`) | `--save-preview` / `tools/panels.py` images | nothing to port — there is no display on the target |
| torch + ultralytics | ONNX via onnxruntime or OpenCV DNN; TFLite via ai-edge-litert | done, `OnnxShapeClassifier` |
| scipy `least_squares` | 3-parameter Levenberg–Marquardt in numpy (~30 lines) | not done; only worth it if scipy's 130 MB matters |
| pyrealsense2 | none on cp313 aarch64 — build librealsense, or Python 3.12 | open |

## 6. Still required for real UNO Q deployment

1. **Make Open3D optional in the code, not just shimmed.** Move `open3d_lite`'s
   functions into `armsoft/core` (e.g. `pointcloud.py`), have `shape_fitter` return
   plain `(verts, edges)` instead of `o3d.geometry.LineSet`, and import Open3D only
   inside `sinks/viewer.py`. Right now `import armsoft` fails without Open3D.
2. **Split requirements**: `requirements-board.txt` (numpy, scipy,
   opencv-python-headless, onnxruntime) vs `requirements-dev.txt` (torch,
   ultralytics, open3d, onnx). `setup_env.sh` always installs torch.
3. **Camera on the board**: confirm the QRB2210's USB port can run a D405 (USB 3,
   power), then get `pyrealsense2` for Python 3.13 aarch64 (build librealsense with
   `-DBUILD_PYTHON_BINDINGS=ON`). Otherwise stream frames from another host through a `FrameSource`.
4. **Re-check torch on the real board** only if it is kept there for any reason: verify the oneDNN discrepancy (§4) or disable mkldnn.
5. **Measure on the real board**: per-stage latency (isolation dominated at about 45 ms/frame on
   x86, and it scales with resolution; the A53 will be several times slower), peak RSS against
   2 GB, and thermals. Candidate knobs: 424×240 streams, and a larger voxel.
6. **Record real D405 sequences** (`--record`) and add them to `testdata/rec/`. Every
   check here used simulated frames, and the real-camera width bias (+7 to +29 mm) is still unexplained.
7. **Held-out classifier data**: `data/val` points back to `data/train`, so the classifier
   has never been validated on held-out crops. Collect a real val set before trusting any
   int8 model, or comparing any future models.
8. **Preprocessing drift**: either accept the ≤0.05 probability drift or make
   `preprocess_crop` reproduce PIL's antialiased resize exactly. A threshold-margin check is cheap.
9. **Robot link**: the grasp dict has to reach the arm through the UNO Q's STM32U585 MCU. Decide
   between the Arduino App Lab RPC bridge and `--serial`. There is no ROS 2 for Debian 13 arm64.
10. **float32 numerics** (from the main README) are still untested; the shim is float64.

## Files

| file | purpose |
|---|---|
| `unoq_env.sh` | build / enter the emulated UNO Q environment |
| `check_env.py` | import **and exercise** every dependency, OK / MISSING / BROKEN |
| `export_classifier.py` | `.pt` → ONNX fp32 (+ int8 QDQ calibrated on real crops) |
| `export_tflite.py` | ONNX → TFLite fp32 / fp16 / int8 (separate TF venv) |
| `test_classifier.py` | parity + latency of every classifier backend vs PyTorch |
| `open3d_lite.py` | numpy/scipy stand-in for the Open3D subset armsoft uses |
| `run_replay.py` | run the pipeline on recorded `.npz` sequences, real or lite Open3D |
| `compare_results.py` | frame-by-frame diff of two replay runs |
| `record_sim.sh` | regenerate `testdata/rec/` |
| `testdata/real_crops/` | 174 real classifier crops (copy of `../data/train`) |
| `results/` | JSON outputs of the runs above |
