# Generate synthetic RGB-D data for N-Pulse with Fusion

Start with three hand poses to generate **18 frames**. Check the images, depth, and keypoints before expanding to **60–90 frames** from 10–15 hand poses, with six arm placements per pose. Fusion solves the linked finger joints. A Python script then renders the solved CAD geometry from a simulated chest-mounted camera and writes 21 keypoints per frame. The images cover the hand, wrist, and visible forearm.

## 1. Set up your workspace

- Install Autodesk Fusion desktop and Python 3.12 on Windows or macOS. Have a local copy of this repository ready. The workflow uses a simulated camera, so you can run it without a D405.
- Run the terminal commands from the root of the `ARM-SOFT-Grasping` repository. The guide stores exports and generated data in `ARM-SOFT-data`, a folder next to the repository.

## 2. Export the three baseline poses in Fusion

1. Open the **Data Panel** at the top left of Fusion and go to `EPFL N-Pulse / 2026-2027 / Prosthesis / Assembly`. Open **`Assembly_hand_old_motors` v22**. If Fusion lists a newer version, open v22 from **History**. In the education edition, start from the version number below the file. Leave the assembly in its saved pose. [Autodesk guide to opening older versions](https://help.autodesk.com/cloudhelp/ENU/Fusion-Designs/files/ASM-OPEN-OLDER-VERSION.htm)
2. Create `ARM-SOFT-data/fusion-v22-baseline/` next to the repository and copy [baseline_poses.json](../tools/fusion/baseline_poses.json) into it. The file defines an open hand, the pose saved in the assembly, and a grasp.
3. In Fusion's **Design** workspace, go to **Utilities → Add-Ins → Scripts and Add-Ins → Scripts**. Click **+** and select [NPulseExport.py](../tools/fusion/NPulseExport/NPulseExport.py) from the repository. Keep [NPulseExport.manifest](../tools/fusion/NPulseExport/NPulseExport.manifest) in the same folder as the script. Select **NPulseExport**, click **Run**, and choose the copy of `baseline_poses.json` you made in the previous step. [Autodesk guide to adding and running scripts](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/UsingSamplesFromGitHub_UM.htm)
4. Confirm the export. For each pose, the script starts from saved v22. It exports `as_saved` directly and moves the finger joints in small increments for the other two poses. After each export, it closes the assembly tab without saving the temporary pose. When it finishes, look for `fusion_export/` next to the JSON file. It should contain `extension/`, `as_saved/`, `grasp/`, and `export_report.json`. Open the report and check that all three poses have a `measured_deg` entry. Review any entries under `error`, `errors`, or `skipped_bodies`.

If the export stops partway through, keep the report for troubleshooting. For a retry, copy the pose configuration into a fresh working folder. Fusion creates `fusion_export/` during each run.

## 3. Generate the 18 baseline frames

Use the commands for your operating system. They read the Fusion export and create a new `rgbd/` folder.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r pose_estimation/tools/fusion/requirements.txt
.\.venv\Scripts\python.exe pose_estimation/tools/fusion/generate.py --input "../ARM-SOFT-data/fusion-v22-baseline/fusion_export" --out "../ARM-SOFT-data/fusion-v22-baseline/rgbd"
```

**macOS Terminal**

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r pose_estimation/tools/fusion/requirements.txt
.venv/bin/python pose_estimation/tools/fusion/generate.py --input "../ARM-SOFT-data/fusion-v22-baseline/fusion_export" --out "../ARM-SOFT-data/fusion-v22-baseline/rgbd"
```

When the terminal prints `3 accepted groups, 0 rejected groups, 18 frames`, open `rgbd/gallery.png`. Each row shows one hand pose. From left to right, the placements show the palm facing the camera, the palm at an angle, the thumb side, the back of the hand at an angle, the palm close to the chest, and a forward reach. The camera stays fixed while the hand and forearm move. Compare **001 (open hand)** with **013 (grasp)**, then check that `rgbd/validation.json` reports 18 frames and 378 keypoints.

| File in `rgbd/` | What to expect |
| --- | --- |
| `rgb/001.png` | An 848 × 480 color image without colored markers or a skeleton overlay. |
| `depth_mm/001.png` | A 16-bit depth image aligned with the RGB image. Each value is distance along the camera's optical axis in millimeters. Zero means no surface. A regular image viewer may display it as very dark. |
| `labels/001.json` | Pixel coordinates and 3D camera-frame coordinates for 21 keypoints on the wrist axis, palm, and five fingers. The 3D values are in meters. Camera X points right, Y points down, and Z points forward. |
| `overlay/001.png` | The 21 keypoints drawn on the RGB image for review. The overlay includes occluded keypoints. |
| `finger_side_review.png` | A CAD side view of each long finger in each pose. From left to right, the columns show the index, middle, ring, and little fingers. |
| `geometry_review.json` and `joint_audit.json` | The actual drive angles read back from Fusion, signed bend angles for the four long fingers, raw joint records, and the reasons any poses were rejected. |

## 4. Expand the dataset to 60–90 frames

1. Create `ARM-SOFT-data/fusion-v22-batch01/` and copy the baseline configuration there as `poses.json`. The example below adds a half-grip pose to try. Enter the target drive angles in degrees for the **index, middle, ring, little finger, and thumb**, in that order. Fusion's Motion Links drive the remaining finger joints.

   ```json
   {
     "poses": {
       "extension": [0, 0, 0, 0, 0],
       "as_saved": null,
       "grasp": [52, 55, 58, 54, 48],
       "half_grip": [25, 28, 30, 27, 24]
     }
   }
   ```

2. Start with two or three new poses. Follow Step 2 using the new `poses.json`. Then follow Step 3 and replace both occurrences of `fusion-v22-baseline` in the generation command with `fusion-v22-batch01`. Once the trial passes, use fresh working folders and updated configurations until you have **10–15 poses that pass review**. Include an open hand, partial grips, a grasp, and poses where different fingers bend by different amounts. The script generates the same six arm placements for each pose.
3. Review `geometry_review.json` and inspect each new pose in `finger_side_review.png` and in at least one full-hand overlay. Look for a fingertip bending backward, parts intersecting, missing parts, or a keypoint following the wrong fingertip. In an earlier batch, the middle joint bent toward the palm while the last joint bent backward, even though the 2D projection looked plausible. The script now checks the signed 3D bend at both joints of each long finger. Record why any pose failed, adjust its target angles, and solve it again from v22. The automated angle checks apply to these CAD poses. The physical joint limits still need to be verified.
4. Record which keypoints are observable from the simulated chest view. You can use an `observability_review.csv` with columns `frame,keypoint_id,status,reason`. Keep uncertain points marked `unreviewed`. An occluded joint still has a 3D location in CAD. A true `surface_depth_valid` value tells you there is a surface at that pixel, which may belong to a part in front of the joint. The thumb connection between `Pouce C` (the joint linked to the hidden old tip) and `NewTip` (the visible tip) still needs verification. The camera uses simulated intrinsics and a 35° downward tilt. The physical D405 parameters still need to be measured.
5. Create `split.json` with the `pose_group` names assigned to `train`, `val`, and `test`. Keep all six frames of a hand pose in the same split. Put near-identical hand poses in the same split as well. Exclude poses rejected during manual review. Keep the separate 18-frame baseline run for reproducibility checks.

## 5. Hand over the batch

Send the following to the project lead through the team's data-sharing channel.

- `poses.json`, `fusion_export/`, `rgbd/`, and a command you ran successfully to reproduce the batch.
- The two review images (`gallery.png` and `finger_side_review.png`), `validation.json`, `geometry_review.json`, the manual review notes, and `split.json`.
- A short summary with the machine and Fusion version, counts of accepted and rejected poses, the final frame count, and the reasons for rejection. Include separate timings for Fusion's pose solving, mesh export, and Python rendering. List any remaining thumb, missing-reference, or observability issues and the frame IDs they affect.

If Fusion reports a version mismatch, update the desktop app and reopen v22. If an external reference fails to load, send the project lead the assembly name, version, and exact message. The earlier baseline had an unresolved reference named `°Coque2o`. If a visible part is missing, keep the error report and pause the expansion until the assembly can be checked.
