# ARM-SOFT Grasping — standalone model

Vision-based grasp planning for the **n-pulse** soft robotic hand.

One aligned RGB-D frame goes in; the shape of the object, a grasp pose and the
six joint targets for the arm come out. This tree is **self-contained**: it does
not import anything from outside this folder, and it needs no robot, no
middleware, no GPU and includes a built-in camera simulator so no camera is needed to kickstart.

```
RGB-D frame  ──►  isolate the object  ──►  classify the shape  ──►  fit a
                  (HSV + depth)            YOLOv8-cls, on CPU        primitive
                                           (cylinder / cuboid)          │
              grasp pose + joint targets  ◄──  derive the grasp  ◄──────┘
```

The one neural network is the shape classifier: a small **YOLOv8n-cls** model
(`models/shape_classifier.pt`, trained on cylinder/cuboid crops) that runs on
the CPU and answers a single two-class question. Everything else — isolation,
primitive fitting, grasp geometry — is classical computer vision. A
geometry-only fallback can replace the network entirely; see
`--classifier` below.

---

## 1. Requirements

* Python **3.9 – 3.12**.
* About **3 GB** of disk for the virtual environment (most of it PyTorch).
* No GPU, no robot. A camera is optional.

### Which platform?

| | model + simulator + replay | real RealSense camera |
| --- | --- | --- |
| **Linux** (x86-64 / ARM64) | yes | yes |
| **Windows** | yes | yes |
| **WSL2** | yes | awkward — see below |
| **macOS** | yes | needs librealsense built from source |

Everything the model needs ships as a wheel for all three desktop platforms, so
the simulator and recorded-frame workflows run anywhere. Only the camera is
picky: `pyrealsense2` publishes Linux and Windows wheels but none for macOS.

**WSL2** runs the model fine, but a USB camera has to be forwarded from Windows
with [usbipd-win](https://github.com/dorssel/usbipd-win), which is fiddly and
loses frames. Develop in WSL against the simulator or recorded frames, and use
native Windows or native Linux when the camera is plugged in.

The environment shipped in `.venv/` is **Linux x86-64 only** — it holds
absolute paths and Linux wheels. On any other platform run `setup_env.sh`
(macOS, Linux, WSL, Git Bash) or the manual steps below (Windows PowerShell).

On a bare Linux server, install the OpenGL system libraries once, because
Open3D links against them even when you never open a window:

```bash
sudo apt install libegl1 libgl1 libglib2.0-0     # Debian / Ubuntu
```

If you cannot install system packages, see *Troubleshooting* at the end.

## 2. Set up the environment

A ready-made environment is already present in `.venv/`. Activate it:

```bash
cd Fresh-Branch
source .venv/bin/activate            # Windows: .venv\Scripts\activate
```

To build it from scratch instead — on another machine, or after moving this
folder — run:

```bash
bash setup_env.sh                    # creates .venv and installs everything
source .venv/bin/activate
```

The script installs the CPU builds of PyTorch and torchvision and then
`requirements.txt`, and verifies that every import works.

> Install `torch` and `torchvision` from the **same** index. A mixed pair
> (one CPU wheel, one from PyPI) fails at runtime with
> `operator torchvision::nms does not exist`, and the model quietly falls back
> to the weaker geometric classifier.

<details>
<summary>Doing it by hand (and on Windows)</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Windows PowerShell — the same steps, different activation:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
</details>

Versions this tree was tested with (Ubuntu 22.04, Python 3.10, CPU only):
`numpy 2.2.6`, `scipy 1.15.3`, `opencv 5.0.0`, `open3d 0.20.0`,
`torch 2.14.0+cpu`, `ultralytics 8.4.158`.

## 3. Check that it works

```bash
python -m armsoft.tests.selftest
```

It renders a synthetic scene, isolates the object, fits both a cylinder and a
cuboid, and validates the output format. The last line must read
`All checks passed.`

## 4. Run an inference

```bash
python -m armsoft --frames 1
```

With no camera attached this prints:

```
[frame_source] no RealSense camera — falling back to SIMULATION mode.
[run] table normal = [ 0. -1.  0.]
[run] shape hint from: yolo
[0000] cylinder pos=(+0.000,+0.031,+0.351)m d=0.352m jaw=77mm roll=+1.57rad elev=+0.0° bear=+0.0° stable=False (49ms)
[run] 1 grasp(s) computed in 0.42s
```

More ways to run it:

```bash
# full JSON result for every frame
python -m armsoft --source sim --frames 10 --json

# a cuboid instead, 40 cm away, 5 cm wide, 12 cm tall, rotated 30°
python -m armsoft --source sim --sim-shape cuboid \
    --sim-distance 0.40 --sim-diameter 0.05 --sim-height 0.12 --sim-yaw 30

# swap YOLO for the geometry-only fallback (this path alone needs no torch)
python -m armsoft --source sim --classifier geometric --frames 5

# log one JSON object per frame
python -m armsoft --source sim --frames 50 --jsonl grasps.jsonl

# save the annotated camera image of the last frame
python -m armsoft --source sim --frames 1 --save-preview preview.png

# with a real depth camera: live, unlimited, 3-D window, publishing to ROS 2
python -m armsoft --source camera --frames 0 --viz --ros2

# record real frames now; replay them later on a machine with no camera
python -m armsoft --source camera --frames 60 --record rec/
python -m armsoft --source replay --replay-path rec/ --frames 0
```

`python -m armsoft` and `python -m armsoft.cli` are the same thing;
`--help` lists every flag.

### The flags you will actually use

| Flag | Default | Meaning |
| --- | --- | --- |
| `--source auto\|camera\|sim\|replay` | `auto` | where frames come from; `auto` = real camera if one is reachable, otherwise the simulator |
| `--frames N` | `1` | how many frames to process; `0` = run until the source stops or you press Ctrl+C |
| `--classifier auto\|yolo\|geometric\|fixed` | `auto` | how the object's shape is decided; `auto` uses the trained YOLO model when it and its weights are available, else the geometric fallback |
| `--table-normal x,y,z` | `0,-1,0` | the work-surface normal in camera coordinates |
| `--calibrate` | off | find the table plane from a chessboard instead |
| `--json` / `--jsonl FILE` | off | full result to the terminal / one JSON object per line to a file |
| `--viz` | off | live 3-D window (needs a display) |
| `--ros2` / `--serial PORT` | off | publish the grasp to ROS 2 / to a serial device |

## 5. Using it from your own code

The package has three layers — `armsoft.sources` (where frames come from),
`armsoft.core` (the model) and `armsoft.sinks` (where results go). Everything
common is re-exported at the top level, so you can import from either place:

```python
from armsoft import GraspPipeline, create_frame_source, build_classifier

with create_frame_source("auto") as source:          # camera, or simulator
    pipeline = GraspPipeline(
        table_normal=source.table_normal_hint,       # or your own calibration
        classifier=build_classifier("auto", model_path="models/shape_classifier.pt"),
    )

    for frame in source.frames(limit=100):
        result = pipeline.process(frame)
        if result.ok and result.stable:
            send_to_robot(result.to_dict())          # plain JSON-safe dict
            break
```

Two worked examples are in `examples/`:

```bash
python examples/minimal_inference.py     # ~15 lines, one frame, one grasp
python examples/custom_source.py         # plug in your own camera / data
```

## 6. Input and output formats

Both are deliberately narrow and versioned, so that new hardware only has to
re-implement the *edges* — never the model.

### Input: `RGBDFrame`

| Field | Type | Meaning |
| --- | --- | --- |
| `color_bgr` | `(H, W, 3) uint8` | colour image, OpenCV **BGR** order |
| `depth_m` | `(H, W) float32` | depth in **metres**, aligned to the colour image; `0.0` means "no reading" |
| `intrinsics` | `CameraIntrinsics` | `width, height, fx, fy, cx, cy` of the aligned pair |
| `index`, `timestamp`, `source` | `int`, `float`, `str` | bookkeeping only |

Coordinates are **+X right, +Y down, +Z forward**, in metres.

Anything that can produce those two arrays is a valid source: subclass
`FrameSource` and implement `read()` — the rest of the model is unchanged. See
`examples/custom_source.py`.

### Output: `GraspResult.to_dict()`

Plain JSON — scalars and flat lists of floats, no numpy and no Open3D objects.
The `schema` field is `"arm-soft-grasp/1.0"`; bump it if you change the fields.

| Key | Type | Meaning |
| --- | --- | --- |
| `status` | str | `ok`, `no_object`, `no_shape` or `no_grasp` |
| `shape` | str \| null | `"cylinder"` or `"cuboid"` |
| `shape_source` | str | which classifier produced the hint |
| `position_m` | `[x, y, z]` | tool centre point, between the fingertips |
| `approach`, `closing`, `binormal` | `[3]` | the grasp axes (columns of `rotation`) |
| `rotation` | `[3][3]` | row-major rotation matrix |
| `jaw_opening_m` | float | full jaw opening = object width + 2 × 8 mm clearance |
| `object_width_m`, `object_height_m`, `distance_m` | float | fitted object descriptors |
| `joint_names`, `joint_positions` | `[6]` | targets for the arm |
| `base_roll_rad` | float | `π/2 − alpha`; turns the jaw line vertical relative to the table |
| `elevation_deg`, `bearing_deg` | float | approach direction |
| `hand_pose` | int | grasp trigger: `1` once the result is stable |
| `stable`, `pos_std_m`, `jaw_std_m` | bool, float | stability gate over the last 8 frames (4 mm / 3 mm) |
| `gripper_points` | `[6][3]` | gripper skeleton, for drawing |
| `timing_ms` | dict | per-stage timing |

A grasp is only worth acting on when `status == "ok"` **and** `stable` is true:
that means the tool centre point and the jaw opening have both held still for
eight consecutive frames.

## 7. Running without a camera

`SimulatedSource` renders a red cylinder or cuboid standing on a lightly
patterned table: it samples the primitive's surface densely, shades it, and
z-buffers the points through the same pinhole model a real camera uses. That
reproduces the two properties the model actually cares about — only the
camera-facing surface is visible, and the depth is noisy.

It is accurate enough to serve as a regression test: on a simulated 60 mm ×
100 mm cylinder the fitter recovers `r = 30.5 mm, h = 98.7 mm`.

What it does **not** reproduce: real sensor artefacts (multipath, dropouts at
grazing angles, colour noise), cluttered backgrounds, motion blur, and any
in-camera post-processing. Numbers measured in simulation are a sanity check,
not a validation of accuracy — for that, record real frames with `--record` and
replay them.

Use `--source sim` to force simulation even when a camera is plugged in, and
`--source camera` to fail loudly instead of silently falling back.

## 8. Using a real camera

Nothing in the model changes — a camera is just another frame source. The steps
below assume an Intel RealSense (the D405 this was built around).

**1. Install the SDK** (not part of `requirements.txt`, because the model does
not need it):

```bash
pip install pyrealsense2
```

**2. Plug the camera into a USB 3 port** — a blue connector, and a USB 3 cable.
USB 2 either fails to start the streams or silently drops to a low frame rate.

**3. Check that it is seen:**

```bash
python -c "import pyrealsense2 as rs; print(rs.context().query_devices())"
python -c "from armsoft import camera_available; print(camera_available())"
```

The second line is what `--source auto` uses. `True` means the next run takes
live frames; `False` means it falls back to the simulator.

**4. Run it:**

```bash
python -m armsoft --source camera --frames 1          # one live inference
python -m armsoft --source camera --frames 0 --viz    # continuous, 3-D window
```

Use `--source camera` rather than `auto` while testing, so a camera problem
fails loudly instead of quietly switching to simulation.

**5. Calibrate the table.** The grasp is expressed relative to the work
surface, so the model needs its normal. Either measure it once and pass it:

```bash
python -m armsoft --source camera --table-normal 0,-1,0
```

or show the camera a chessboard and let it fit the plane:

```bash
python -m armsoft --source camera --calibrate --frames 0
```

Without either, it assumes `0,-1,0` (camera upright, table below the lens).

**6. Record while you have the camera**, so you can keep working without it:

```bash
python -m armsoft --source camera --frames 200 --record rec/
python -m armsoft --source replay --replay-path rec/ --frames 0   # anywhere, later
```

This is the most useful thing to do on a first session with real hardware: 200
frames of the real objects give you a regression set the simulator cannot match.

**7. Publish to the robot**, once the grasp looks right:

```bash
python -m armsoft --source camera --frames 0 --ros2      # needs a ROS 2 install
```

### What to expect on real data

The object must be **red** (HSV segmentation), between **7 cm and 70 cm** away,
and at least **500 px** in the image. Results on real frames will be noisier
than the simulator suggests — see *Known limitations*. Watch the `stable` flag
rather than individual frames: only a stable grasp is worth acting on, and only
stable results are published.

## 9. Visual check of the simulated scenes

```bash
python tools/save_sim_results.py              # writes temp-sim-res/
```

Runs a list of simulated scenes and saves one annotated PNG per scene, so the
whole pipeline can be inspected by eye without a camera or a 3-D window. Each
image is a four-panel strip:

| panel | shows |
| --- | --- |
| 1. camera view | the colour image, the red mask, the bounding box, and the classifier's label with its confidence — marked `REJECTED` when it falls below the 0.70 threshold |
| 2. depth | the depth image as a colour map; black where the sensor returned nothing |
| 3. fit + grasp | the fitted wireframe (cyan) and gripper (orange) projected back onto the image |
| 4. top-down | a bird's-eye view on the table plane — the measured surface points, the fitted footprint and the jaws. **This is the panel to read**, because in the camera view the gripper points nearly at the lens and collapses to a line |

A caption under each strip prints the ground truth next to what the model
produced, so errors are obvious. `summary.md` tabulates every scene and
`_contact_sheet.png` stacks them all.

Every scene is rendered at four corruption levels, saved side by side:

| suffix | corruption |
| --- | --- |
| *(none)* | the default 0.8 mm depth noise |
| `__noise` | 3 mm depth noise |
| `__worms` | 2 mm noise **+ 12 worms** |
| `__worms_heavy` | 4 mm noise + 28 thick worms + 6 depth holes |

so `cylinder_default.png` and `cylinder_default__worms.png` sit next to each
other for comparison. Use `--variants clean,worms` to render a subset.

Edit the `SCENES` list at the top of the script to add your own cases, or pass
`--frames`, `--out` and `--classifier`.

What the current set shows:

* Cylinders are fitted accurately — 61 × 99 mm for a true 60 × 100 mm at 35 cm,
  and 41 × 139 mm for a 40 × 140 mm.
* The cuboid footprint is overestimated (121 mm fitted for a 60 mm cube),
  clearly visible in panel 4 as a cyan square much larger than the measured
  points. Only two faces are ever visible from one viewpoint.
* The classifier rejects a face-on cuboid (0.54) and a 45°-diagonal one (0.63),
  because from those angles the silhouette is a plain rectangle. It also
  rejects a cylinder at 20 cm (0.66), where the top cap leaves the frame.

### Worms: structured corruption

Gaussian depth noise is the easy case — it is zero-mean, so the temporal EMA
averages it away. Real depth cameras fail in *shaped* ways: thin dark streaks
where the projected pattern is lost, ragged holes at grazing angles. Those are
correlated in space and persist across frames, so no amount of averaging helps.

`armsoft/sources/artifacts.py` renders that: short wandering black curves
("worms") painted over the colour image, which also punch the depth out
underneath (a real dropout reports "no reading", not a wrong reading). About
60 % of them are aimed at the object, so they actually bite into the mask
instead of decorating the background.

```bash
python -m armsoft --source sim --sim-worms 12
python -m armsoft --source sim --sim-worms 28 --sim-worm-thickness 6 \
                  --sim-worm-length 90 --sim-depth-holes 6
python -m armsoft --source sim --sim-worms 12 --sim-worm-keep-depth   # colour only
```

They are far more damaging than gaussian noise, and the failure is the
dangerous kind — **confidently wrong rather than uncertain**:

* 12 worms turn a cylinder into a confident `cuboid` at **conf 0.99**, because
  the broken silhouette no longer looks round. The fitted width jumps from
  61 mm to 127 mm and the jaw opens to 143 mm instead of 77 mm.
* The stability gate does catch it — every worm case reports `stable=false`,
  so nothing would be published to the robot. That gate is the main defence
  against this failure mode.
* Heavy worms mostly end in `no_shape` or `no_object`, which is the safe
  outcome.
* Curiously, worms *help* the two scenes the clean classifier rejects
  (`cuboid_face_on`, `cuboid_diagonal`): the extra texture pushes a plain
  rectangular silhouette over the confidence threshold.

### Moving camera (animated)

```bash
python tools/make_sim_video.py                              # -> approach_cylinder.gif
python tools/make_sim_video.py --shape cuboid
python tools/make_sim_video.py --shape cylinder --worms 8
python tools/make_sim_video.py --frames 80 --fps 12 --mp4 --no-settle
```

Renders a hand-held camera shaking its way in towards the object — the
body-worn case — and writes a GIF with the same panels as the stills plus a
fifth one plotting the reported parameters as they update. `--mp4` also writes
a video file.

The motion comes from `armsoft/sources/trajectory.py`, which layers a smooth
eased approach, a slow body sway, and a damped random walk in all six axes (so
the shake drifts like a hand rather than vibrating in place). The camera pose
is a real pose: the scene is re-rendered from it each frame, the table texture
stays put in the world, and the table normal is recomputed in the camera frame.

Three GIFs are checked in:

| file | what it shows |
| --- | --- |
| `approach_cylinder.gif` | 42 → 22 cm. Fit holds at 61 × 99 mm the whole way; the gate latches green while the hand is steady |
| `approach_cuboid.gif` | the same move on a cuboid |
| `approach_cylinder_worms.gif` | with worm dropouts — the label flips to a confident `cuboid 0.98`, the jaw oscillates, and the gate never latches |

Worth watching for: the classifier flips to `cuboid` in the last few frames of
the clean cylinder run too. Very close in, the top cap leaves the view and the
silhouette is a plain rectangle — the same ambiguity as `cuboid_face_on`. The
stability gate catches it, but it argues for a minimum working distance.

### Noise robustness

```bash
python tools/noise_sweep.py                    # writes noise_sweep.md + .png
python tools/noise_sweep.py --levels 0,1,2,5 --repeats 10
```

Sweeps the simulator's depth noise and re-runs each scene with several random
seeds, reporting detection rate, shape accuracy, fitted size, the spread of the
grasp point across seeds, and how often the stability gate passes. A real D405
sits around 1–3 mm of depth noise at these ranges.

Results of the default sweep (5 seeds × 12 frames per point):

* **Position is very stable.** The grasp point moves under 1.3 mm across seeds
  at every noise level tested, including 10 mm — the temporal EMA absorbs
  zero-mean noise well. Shape classification is likewise flat.
* **Size is not.** The fitted width shrinks steadily as noise rises: a true
  60 mm cylinder is fitted at 61 mm with clean data, 58 mm at 3 mm noise, and
  **43 mm at 10 mm noise** — a 28 % underestimate. Since the jaw opening is
  derived from that width, heavy noise makes the gripper close *too far*, which
  is the failure mode to watch on real data. The same trend holds for the
  cuboid (121 → 107 mm).
* **Zero noise is the worst case for the classifier.** With noise switched off
  entirely, the cuboid is rejected in every run (conf 0.65); adding 0.5 mm
  lifts it to 0.75 and 80 % detection. A perfectly clean synthetic render is
  out of distribution for a model trained on real photographs, so do not read
  `--sim-noise 0` as the easy case.
* The cuboid's stability gate is the weak link — 40 % at normal noise, 20 % at
  10 mm — because the footprint fit jitters between frames even when the
  centroid does not.

## 10. What is in this folder

```
Fresh-Branch/
  README.md              this file
  requirements.txt       runtime dependencies
  setup_env.sh           creates .venv and installs everything
  .venv/                 ready-to-activate environment
  models/
    shape_classifier.pt  trained YOLOv8-cls weights (cylinder / cuboid)
  examples/
    minimal_inference.py smallest possible end-to-end use
    custom_source.py     plugging in your own frame source
  tools/
    panels.py            shared drawing code for the two tools below
    save_sim_results.py  render simulated scenes to annotated PNGs
    make_sim_video.py    animated moving-camera run, written as a GIF
    noise_sweep.py       robustness sweep over sensor noise
  temp-sim-res/          output of the above: PNGs, GIFs, summary.md
  armsoft/
    cli.py                 command-line tool (`python -m armsoft`)
    sources/               WHERE FRAMES COME FROM
      base.py                RGBDFrame, CameraIntrinsics, FrameSource
      simulated.py           synthetic RGB-D camera
      artifacts.py           worm / hole corruption applied to a frame
      trajectory.py          camera motion (shaky approach) for the simulator
      realsense.py           live Intel RealSense camera
      recorded.py            replay and recording of .npz frames
      __init__.py            create_frame_source(), camera_available()
    core/                  THE MODEL
      isolation.py           red-object isolation and deprojection
      classifier.py          YOLO, geometric and fixed shape-hint providers
      shape_fitter.py        primitive fitting and temporal smoothing
      grasp_geometry.py      grasp pose, gripper skeleton, joint targets
      pipeline.py            GraspPipeline and GraspResult
      table_plane.py         chessboard calibration of the work surface
    sinks/                 WHERE RESULTS GO
      results.py             stdout / JSONL / callback / ROS 2 / serial
      viewer.py              optional live 3-D window
    tests/
      selftest.py            no-hardware smoke test
```

## 11. Known limitations

* **Red objects only.** Isolation keys on red via HSV thresholds
  (`armsoft/isolation.py`, top of the file) — adjust them for other colours.
* **Two shapes only.** The classifier knows `cylinder` and `cuboid`.
* **Partial views inflate cuboids.** From a single viewpoint only two faces are
  visible, and the cuboid fitter overestimates the footprint: a simulated 60 mm
  cube comes out at about 94 mm wide, which inflates the jaw opening. The
  cylinder fit is accurate to roughly 1 mm.
* **The geometric classifier is the weaker one.** On a clean simulated cylinder
  it reports a cuboid, because the top-cap silhouette registers as corners.
  Prefer the trained model when torch is available.
* **Everything is in the camera frame.** There is no camera-to-arm calibration
  here; the grasp is expressed relative to the camera.
* **Structured corruption flips the classifier, confidently.** Worm-shaped
  dropouts make a cylinder read as a cuboid at 0.99 confidence and roughly
  double the jaw opening. The stability gate rejects these frames, so nothing
  reaches the robot, but the classifier itself gives no warning.
* **Noise shrinks the fitted size.** See the sweep above: past ~3 mm of depth
  noise the estimated object width is increasingly underestimated, and the jaw
  opening with it.
* **Speed.** About 50 ms per frame on a laptop CPU — isolation ≈ 40 ms,
  the classifier ≈ 6 ms, the fit ≈ 3 ms. Isolation dominates and scales with
  image area, so lowering the resolution is the obvious lever.

## 12. Porting to a smaller board

If this has to run on a microcontroller-class board, in rough order of effort:

1. **Open3D is the biggest obstacle** — it is used for voxel downsampling,
   DBSCAN clustering, normal estimation and the `LineSet` container, and it is
   both heavy and hard to build for small ARM targets. Replacing those four
   uses with numpy/scipy equivalents (hash-grid voxel filter, `cKDTree` plus
   local PCA for normals, connected components, a plain vertex/edge array) is
   the highest-value task. `grasp_geometry.wireframe_vertices()` already
   accepts a plain `(N, 3)` array, so the downstream half is ready.
2. **torch + ultralytics** are about 2 GB installed. Either run with
   `--classifier geometric` (weaker; see above) or export the classifier to
   ONNX/TFLite — the network is tiny, YOLOv8n-cls at 128×128.
3. **OpenCV** is available on ARM Linux but not on bare metal.
4. **SciPy** is used for exactly one call, `least_squares` in the cylinder fit.
5. **The camera.** A USB 3 depth camera may not be hostable; if not, frames
   must arrive from elsewhere, which is what `FrameSource` is for.
6. **Numerics.** Everything is float64 numpy; a float32 or fixed-point port is
   untested.

## 13. Troubleshooting

**`ImportError: libEGL.so.1: cannot open shared object file`** — Open3D's
OpenGL dependency is missing. Install it (`sudo apt install libegl1 libgl1`).
Without root, fetch the libraries into a local folder and point the loader at
them:

```bash
mkdir -p vendor && cd vendor
apt-get download libegl1 libglvnd0 libglx0 libgl1 libopengl0 libglapi-mesa
for f in *.deb; do dpkg-deb -x "$f" .; done
cd ..
export LD_LIBRARY_PATH="$PWD/vendor/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
```

**`--viz` fails, or OpenCV complains about a display** — you are on a headless
machine. Drop `--viz` and use `--save-preview out.png` instead.

### Camera

**`no RealSense camera — falling back to SIMULATION mode`** with a camera
plugged in — in order of likelihood: `pyrealsense2` is not installed in *this*
environment (`pip install pyrealsense2`); the camera is on a USB 2 port or
cable; another process already holds it (close the RealSense Viewer); or, on
Linux, the udev rules are missing — see the librealsense install notes.

**`RuntimeError: No device connected` / `Couldn't resolve requests`** — the
device was found but the requested streams were not. Usually USB 2, or a model
that does not do 640×480 @ 30 fps for both depth and colour. Try
`RealSenseSource(width=848, height=480)` or another frame rate.

**Frames arrive but everything is `no_object`** — see the last entry in this
section; the object must be red, 7–70 cm away, and reasonably large.

**The camera works on Windows but not in WSL** — expected. USB devices are not
visible to WSL2 unless forwarded with usbipd-win. Use native Windows or Linux
for live capture, and `--source replay` in WSL.

**`[classifier] ultralytics unavailable (operator torchvision::nms does not
exist)`** — `torch` and `torchvision` came from different indexes. Reinstall
them together:

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**`--ros2` fails with `ModuleNotFoundError: rclpy`** — `rclpy` comes from a ROS 2
installation, not from pip. Source your ROS 2 setup file in the same shell
before activating this environment.

**The model reports `no_object` on every frame** — the object is not red enough
for the HSV thresholds, is smaller than 500 px in the image, or is outside the
7 cm – 70 cm depth window. The first two live at the top of
`armsoft/core/isolation.py`, the depth window just below them. Run with
`--save-preview p.png` and look at the mask.

### Platform

**`.venv` does not activate, or imports fail after copying this folder** — the
shipped environment is Linux x86-64 and stores absolute paths. Delete `.venv`
and run `setup_env.sh` again.

**`setup_env.sh` will not run on Windows** — it is a bash script. Use Git Bash,
or the PowerShell steps in section 2.

**`python: command not found` / wrong Python** — use `python3` on Linux and
macOS, `py` on Windows, and make sure the environment is activated: the prompt
should show `(.venv)`.
