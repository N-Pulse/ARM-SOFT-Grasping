# ARM-SOFT-Grasping

Real-time computer-vision grasping pipeline for the **n-pulse** soft/dexterous
robotic hand. A single Intel RealSense **D405** RGB-D camera isolates a target
object, estimates its geometry, computes a grasp pose, and publishes joint
commands to the robot over ROS 2.

Two grasp-planning back-ends exist in this repo:

- **Shape-fitting** — segments the object, fits a geometric primitive
  (cylinder or cuboid), and derives an analytic grasp from the fitted shape.
  Lightweight, deterministic, and runs on-device (e.g. Jetson).
  **This is the pipeline used in the final deployment**
  (`test/test_lineset_live_shape.py`).
- **GraspNet** — runs the [GraspNet-baseline](https://github.com/graspnet/graspnet-baseline)
  network on the isolated point cloud and projects the best predicted
  parallel-jaw grasp onto the hand. Explored during development
  (`test/test_lineset_live.py`) but **not used in the final system**; kept for
  reference and comparison.

## How it works

Every frame flows through the same front-end and one of two planners:

```
RealSense D405 (aligned RGB-D)
        │
        ▼
ObjectIsolator                 capture/object_isolation.py
  · depth-discontinuity + white-background removal
  · HSV red-colour segmentation → object closest to frame centre
  · masked, subsampled point cloud
  · YOLOv8-cls shape hint (cylinder / cuboid)
        │
        ├──────────────► Shape-fitting planner
        │                  ShapeTracker + fit_and_track   capture/shape_fitter.py
        │                    ① SOR ② normals ③ classify
        │                    ④ primitive fit ⑤ height ⑥ EMA track ⑦ wireframe
        │                  grasp from fitted geometry      capture/grasp_projection.py
        │
        └──────────────► GraspNet planner
                           GraspNet inference              capture/grasp_pipeline_graspnet.py
                           grasp → hand projection         capture/grasp_projection.py
        │
        ▼
Open3D live viewer (point cloud + grasp/wireframe overlay)  helper/pcd_visualizer.py
        │
        ▼
ROS 2 publish → n-pulse robot                              test/test_lineset_live_shape.py
```

The **table plane** (captured from a chessboard at startup) provides a stable
`table_normal` used both for primitive classification and for orienting the
grasp relative to the work surface.

## Repository layout

```
capture/
  object_isolation.py         RealSense capture + red-object isolation (threaded)
  shape_classifier.py         YOLOv8-cls wrapper → "cylinder" | "cuboid"
  shape_fitter.py             primitive fitting + temporal ShapeTracker
  grasp_projection.py         parallel-jaw grasp → dexterous-hand projection
  grasp_pipeline_graspnet.py  GraspNet-baseline inference pipeline
helper/
  pcd_visualizer.py           reusable Open3D viewer for an ObjectIsolator stream
test/
  test_obj_iso.py             isolation-only smoke test
  test_shape_fit.py           shape fitting + live wireframe
  test_lineset_live_shape.py  full shape-based grasp + ROS 2 publishing
  test_lineset_live.py        full GraspNet grasp overlay
old_test/                     earlier experiments (pointcloud, upsampling, hand draw)
collect_shape_data.py         interactive dataset collector for the shape classifier
realsense_viewer.py           raw depth+color viewer (camera sanity check)
stl/                          reference primitive meshes (cube, cylinder, rod)
data/                         YOLO classify dataset (train/val, cylinder/cuboid)
runs/classify/                trained classifier runs + metrics
Dockerfile                    unused (experimental ROS 2 install attempt; not part of the workflow)
```

## Requirements

**On the Jetson, everything is already installed** — all dependencies live in
the `realsense` virtual environment, and ROS 2 Humble is already set up. You do
**not** need to install or download anything again; just activate the
environment (see below) and run. The list below is only for reference or for
reproducing the setup on a fresh machine.

- Intel RealSense **D405** (aligned depth + color, 640×480 @ 30 fps)
- Python 3, with `pyrealsense2`, `open3d`, `numpy`, `scipy`, `opencv-python`,
  `torch`, `ultralytics`
- **ROS 2 Humble** (only needed for publishing to the robot;
  `rclpy`, `std_msgs`, `trajectory_msgs`)
- **GraspNet planner only:** a local checkout of `graspnet-baseline` and its
  dependencies (`graspnetAPI`, PointNet2 ops), plus a trained checkpoint

The repo runs directly in the `realsense` virtual environment on the Jetson —
**no Docker is used**. The `Dockerfile` was an experimental attempt at
installing ROS 2 (not the venv); it is not part of the final workflow and can be
ignored.

## Environment setup

Before running **any** script, activate the `realsense` virtual environment and
source ROS 2 in the same shell:

```bash
# If it is a plain Python venv:
source <path-to>/realsense/bin/activate
# If it is a conda environment instead:
conda activate realsense

# Then source ROS 2 Humble:
source <path-to-ros>/setup.bash        # e.g. /opt/ros/humble/setup.bash
```

> **Check on the Jetson first.** It is not currently confirmed whether
> `realsense` is a plain Python venv or a conda environment — verify on the
> device and use the matching activation command above. Likewise, the exact
> location of the ROS 2 `setup.bash` needs to be checked on the Jetson.

The `realsense` env on the Jetson already contains every required library, so
there is nothing to install — just activate it. Both lines must be sourced in
every new terminal session: the env provides `pyrealsense2` / `open3d` /
`torch` etc., and the ROS setup provides `rclpy` and the message types used for
publishing.

## Quick start

Verify the camera first:

```bash
python realsense_viewer.py           # raw depth + color, press q to quit
```

Test object isolation only:

```bash
python test/test_obj_iso.py          # isolated point cloud in an Open3D window
```

### Shape-fitting grasp — the deployed pipeline

```bash
python test/test_shape_fit.py                    # fitting + live wireframe (no ROS)
python test/test_shape_fit.py --debug
python test/test_shape_fit.py --board-cols 10 --board-rows 7

python test/test_lineset_live_shape.py           # ← final system: grasp + ROS 2 publish
```

### GraspNet grasp (reference only, not used in the final system)

```bash
python test/test_lineset_live.py --checkpoint /path/to/checkpoint.tar
python test/test_lineset_live.py --checkpoint /path/to/checkpoint.tar --device cpu
```

In all live tools: close the Open3D window, press **Ctrl+C**, or press
**ESC / q** in the camera preview to quit.

## Training the shape classifier

The shape-fitting planner uses a small YOLOv8-classify model as a shape hint
(with a geometric fallback). To (re)train it:

1. Collect labelled crops from the live camera:

   ```bash
   python collect_shape_data.py
   ```

   Place one object in front of the camera at a time and press **C** for
   cylinder, **B** for cuboid, **S / Space** to skip, **Q / ESC** to quit.
   Aim for ~80–100 images per class into `data/train/{cylinder,cuboid}`.

2. Train with the Ultralytics CLI (auto 80/20 train/val split):

   ```bash
   yolo classify train \
       model=yolov8n-cls.pt \
       data=./data/train \
       epochs=50 imgsz=128 batch=16 fraction=0.8
   ```

3. Point `ShapeClassifier` at the resulting weights,
   e.g. `runs/classify/shape/weights/best.pt`.

## ROS 2 interface

`test/test_lineset_live_shape.py` publishes on lock of a stable grasp:

| Topic            | Type                    | Meaning                                                            |
| ---------------- | ----------------------- | ----------------------------------------------------------------- |
| `/cv/model/pose` | `Float64MultiArray`     | Object model: `[is_cylinder, diameter_m, …]` (shape + dimensions) |
| `/cv/base/pose`  | `JointTrajectory`       | Target joint trajectory for the arm/base                          |
| `/cv/hand/pose`  | `Int8`                  | Hand pose / grasp trigger flag                                    |

The base-roll joint is driven to `π/2 − alpha`, where `alpha` is the signed
angle between the gripper closing axis and the table plane, so the jaw line is
rotated vertical relative to the work surface.

## Notes

- Object isolation currently keys on **red** objects via HSV segmentation;
  adjust the thresholds in `capture/object_isolation.py` for other colours.
- Heavy computation (fitting, inference) runs on a background worker thread; the
  Open3D render loop only applies geometry updates, keeping the display
  responsive.
- The `ShapeTracker` convergence constants (`ALPHA`, `N_LOCK`, `N_UNLOCK`) are
  tuned — change them only deliberately.

## License

No license has been assigned to this repository. All rights are reserved by the
authors; contact the maintainer before reuse or redistribution.

## Discussion

Open questions and design notes for the project. For more information, see the
**n-pulse Notion page**.

**What is the ultimate goal of the project?**
The ultimate goal is a computer-vision pipeline that suggests prosthesis
positioning and lets the user grasp an object based on a trigger detected by the
camera.

**For the CV part, is the idea that the arm/prosthesis is worn directly by the
person? If so, have you already thought about the position of the camera
relative to the arm/prosthesis?**
Yes, the camera is worn directly by the person. Our plan is to strap it onto the
user's chest. Right now, for simplicity, we've fixed the distance between the
camera and the object/prosthesis. Going forward, the goal is to support
arbitrary distances, roughly in the 15 to 70 cm range.

**Also, if the whole system is worn on the person, understanding the space
between the camera and the arm is likely to be a major issue. Had you already
considered a solution for the calibration/transformation between the camera
frame and the arm frame?**
Since the whole system is body-worn, calibrating the spatial relationship
between the camera and the arm is quite difficult, and we haven't found an ideal
solution for it yet. Currently, we haven't implemented that calibration;
everything is computed in the camera's reference frame. A related limitation is
that the depth camera only sees the object from a single viewpoint, so the point
cloud covers just the visible surface (and it is fairly noisy). To handle this,
we fit geometric primitives by analyzing the object's surface curvature and
other geometric cues, which both reconstructs an approximate full 3D shape from
the partial view and smooths out the point-cloud noise, giving us a stable shape
to plan the grasp with. A future improvement would be adding a second camera to
get a better depth and position estimation of the object and prosthesis.

**1. Last year, which pipeline actually worked in the demo? Was the demo running
only in simulation/on a computer, or was there a test with a real physical
setup?**
The grasping motion itself ran entirely in simulation. The camera and object
detection, however, were tested with a real physical setup: we captured live
camera input of a real object, used that to generate movement instructions for
the prosthesis, and then executed those instructions in the simulation.

**2. Regarding visual recognition and grasping pose estimation, what was
finished last year? And what remains to be done or improved this year?**
Completed: we trained a custom AI model to recognize two object shapes,
cylinders and cubes, and overlay a matching 3D model of the same shape onto the
camera image. Remaining: expand recognition to a much wider variety of objects.
Existing libraries can do general object recognition, but for our specific use
case (matching 2D input to an approximate 3D shape for grasping), we found it
more effective to train our own model. (The libraries could be a great option,
but they are a pain to download and use on the Jetson.)

**3. For object recognition (YOLO) and the grasping logic, what was your overall
approach? Do you think this is the right direction and that we should keep going
with it, or should we also consider other approaches?**
For object recognition, we ended up using only a small YOLO classifier built
into our own model. Ideally, we'd like to use a library like YOLO instead of a
custom model, since it's a well-established tool with a large range of
pre-trained object classes. However, as mentioned earlier, we ran into
difficulties getting it to run efficiently on the Jetson. Our custom model was
more of a workaround, put together because we were running out of time, not
necessarily because we think it's the better long-term approach. For the future,
it would be great to get YOLO (or a similar library) working properly on the
Jetson, since that would give us much broader object-recognition capability
without having to train and maintain our own model. Also, for YOLO we need to
figure out the issue of only having a 2D view of the object; using multiple
cameras from different angles could be a potential fix for this.

**4. How was the training data for cylinder/cuboid collected? Were the objects
used only red?**
We collected training data by photographing the cylinder and cuboid from many
different angles to give the model a variety of viewpoints to learn from. The
objects were deliberately colored red. This made recognition easier, since red
stood out clearly against the black-and-white checkerboard background we used
for calibration. That said, this was a simplification, not a long-term design
choice. Ideally, the model should be able to recognize objects of any color, but
that requires improving the underlying recognition algorithm so it isn't relying
on color contrast as a shortcut. That's an area we'd want to address if we had
more time.

**5. Is the long-term goal to extend recognition to everyday objects, like a
cup, a bottle, a box, a phone, etc.?**
Yes. We need to check with the prosthesis team which objects are feasible.

**6. For the team handling the movement of the arm/prosthesis, what parameters
does the CV team need to provide as output? For example: object position, object
type, grasping orientation, hand opening, etc. Has there already been a
discussion or integration with the hardware/control team?**
This is best discussed directly with the firmware/prosthesis team, but based on
our current understanding, the key outputs the CV team needs to provide are:

- **Finger closure:** an angle (or set of angles) per finger, indicating how
  much each finger should close to grasp the object.
- **Wrist rotation:** an angle indicating how the wrist should orient itself
  relative to the object.

One important distinction: the positioning of the arm relative to the object
(i.e., moving the arm to the right distance and location) is not something the CV
system controls. That's driven by the prosthesis wearer themselves, since it's
the user who moves their arm to reach the object. Our system's role is to provide
the grasp parameters (finger angles, wrist orientation) once the arm is already
in range, not to control the arm's positioning itself.

We haven't yet had a detailed integration discussion with the hardware/control
team; this is something that should happen directly with them to confirm exact
parameter formats, units, and any additional data they might need (e.g., timing,
feedback signals, or force limits). At the time we were working on it, they
weren't that advanced with the prosthesis yet, so there is definitely lots to
work on next semester on this front.

**7. In your opinion, what was the biggest problem or blocker last year?**
Getting the code to work on the Jetson. Be really careful to be consistent about
where things get downloaded on the Jetson.

**8. For this year, what should be the CV team's first concrete objective?**
Make the object detection better and create better pose estimations.
