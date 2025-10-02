# Intent Prediction Project by Darius Giannoli & Gabriel Taieb

This repository provides a modular pipeline for 3D object manipulation and voice-controlled robotics, integrating RGB-D perception, YOLO-based object detection, and voice recognition. The project is organized into several main components, each responsible for a specific part of the workflow.

---

## Project Structure

```
.gitignore
requirements.txt
pipeline_rgbd/
    main_pipeline.py
    pipeline_no_visu.py
    torque_ini.py
    handle_detection/
        split.py
        cup_handle/
        runs/
    partial_pcd/
        isolate.py
        v2.py
        captures/
    robot/
        hook_main.py
        hook2_main.py
        reset.py
        CUBOID/
        CUBOID2/
        ...
    shape_identification/
        ...
pipeline_yolo/
    main_pipeline_yolo.py
    requirements.txt
    actions/
    game/
    voice_recognition/
        eval_model.py
        dataset_recorder_flexible.py
        preprocess.py
        train_commands.py
        ...
stls/
    PIece1.stl
    Piece2.stl
    Piece3.stl
```

---

## Main Components

### 1. `pipeline_rgbd/`
- **Purpose:** Handles 3D perception and manipulation using RGB-D data.
- **main_pipeline.py:** Main entry point for the RGB-D pipeline; coordinates object detection, shape identification, and robot control.
- **pipeline_no_visu.py:** Variant of the main pipeline without visualization.
- **torque_ini.py:** Initializes or configures robot torque settings.
- **handle_detection/:** Tools and scripts for detecting object handles (e.g., cup handles).
- **partial_pcd/:** Scripts for working with partial point clouds, including isolation and capture utilities.
- **robot/:** Robot control scripts for different gripper types (hook, cuboid, cylindrical, spherical), as well as reset and initialization scripts.
- **shape_identification/:** Modules for identifying object shapes from point clouds.

### 2. `pipeline_yolo/`
- **Purpose:** Contains the YOLO-based object detection pipeline and voice recognition modules.
- **main_pipeline_yolo.py:** Main entry point for the YOLO pipeline.
- **actions/, game/:** Additional modules for robot actions and game logic.
- **voice_recognition/:**
    - **eval_model.py:** Evaluation scripts for voice recognition models.
    - **dataset_recorder_flexible.py:** Tool for recording custom voice datasets.
    - **preprocess.py:** Preprocessing utilities for audio data.
    - **train_commands.py:** Training scripts for command recognition models.

### 3. `stls/`
- **Purpose:** Contains 3D model files (STL format) used for simulation or reference.

### 4. Root Files
- **requirements.txt:** Lists Python dependencies for the project.
- **.gitignore:** Specifies files and directories to be ignored by git.

---

## Getting Started

1. **Install dependencies:**  
   ```
   pip install -r requirements.txt
   ```

2. **Run the main pipelines:**  
   - For RGB-D pipeline:  
     ```
     python pipeline_rgbd/main_pipeline.py
     ```
   - For YOLO pipeline:  
     ```
     python pipeline_yolo/main_pipeline_yolo.py
     ```

3. **Voice dataset recording and training:**  
   See scripts in `pipeline_yolo/voice_recognition/`.

---

## More Information

A detailed documentation website will be available soon, providing in-depth explanations and usage
