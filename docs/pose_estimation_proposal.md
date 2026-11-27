---
created: 2026-11-08
updated: 2026-11-08
tags:
  - N-pulse
  - proposal
  - robot-vision
status: draft
language:
  - en
---

# N-Pulse Robotic Arm Visual Pose Estimation — Project Proposal

## 1. Project Background and Objectives

The N-Pulse robotic arm requires visual estimates of its joint positions, hand opening state, and spatial configuration to support interaction, grasp analysis, and system integration. We will capture RGB-D data with a chest-mounted RealSense D405 and deploy the main vision pipeline on the Arduino UNO Q 4 GB.

We will follow an approach that combines **multiview 3D training labels, mechanically constrained pose estimation, rendering-based supervision, and single-camera operation**. The project will aim to reduce manual 3D annotation and improve pose reliability, with independent evaluation used to establish the actual benefits.

The project objectives are:

- **Joint localization:** estimate the 3D positions of the target arm and finger joints in the camera coordinate frame.
- **Hand opening analysis:** identify open and closed states and quantify the opening of individual fingers and the hand as a whole.
- **Spatial configuration:** estimate the positions and orientations of the main links through a consistent representation of the mechanical state.
- **Autonomous operation:** after deployment, use chest-mounted RGB-D data as the only live sensing input and perform all subsequent processing automatically.

This phase will not depend on encoder readings or motor commands. Additional views will be used for training-data collection and validation; normal operation will use only the chest-mounted camera. The system will report joints with insufficient evidence as “not observable.”

## 2. Available Resources and Key Challenges

The project has initial tools for RGB-D capture, keypoint processing, annotation, training, and 3D fusion. The hardware team can support data collection, changes to component colors, and access to mechanical information. These resources provide a starting point for automatic labeling and structured pose estimation, but the required accuracy and runtime performance have not yet been demonstrated.

| Key challenge | Issue to resolve |
|---|---|
| Obtaining 3D labels | Validate the accuracy of automatically generated multiview labels and measure the manual review effort |
| Connecting visual observations to mechanical structure | Establish accurate relationships between surface markers and internal joint centers |
| Motion, occlusion, and missing depth | Handle camera motion and invisible finger segments, assess validity, and reacquire the target |
| Embedded deployment | Validate camera connectivity, image processing, inference, and sustained operation together on the UNO Q |

The working distance, target accuracy, output rate, and latency requirements are not yet fixed. We will establish them during the first phase using the intended application scenarios and measured results.

## 3. Technical Approach and Implementation

### 3.1 Overall Architecture

The approach consists of **data construction, model training and validation, and deployment inference**. We will first generate reviewable 3D marker labels from multiview observations and train an initial pose model. Once mechanical information is available, we will add structural constraints and rendering supervision, then use the evaluated model for automatic single-camera pose estimation.

Marker identities, camera calibration, coordinate frames, units, and validity definitions will remain consistent throughout the pipeline. Mechanical information will be introduced in the later structural modeling and rendering stage. Additional training views will not be required as live inputs after deployment. The system will extract 2D coordinates, depth, and validity automatically, without manual frame-by-frame input.

**Main modules.** The training and delivery workflow consists of five main modules.

| Module | Input | Main method and purpose | Output |
|---|---|---|---|
| Data acquisition and 3D labeling | Multiview images, video and depth | Detect fixed markers, reconstruct them across views, and review reference labels | 3D training data |
| Pose network training | 2D points, depth, validity and 3D labels | Train a lightweight pose network and evaluate RGB image features | Initial 3D pose estimates |
| Mechanical modeling | Predicted pose, geometry and joint definitions | Apply the mechanical structure to recover joint and link states | Structurally consistent 3D pose |
| Rendering-based refinement | Predicted mechanical model and original RGB-D | Compare projections, silhouettes and depth, then use the errors to refine the network | A model trained with rendering supervision |
| Evaluation and deployment | Trained models and independent evaluation data | Check accuracy, stability and runtime, then select a deployment version | A deployable pose estimation model |

### 3.2 Marker Design and Multiview 3D Labeling

We will place distinguishable colors or visual markers at fixed locations on the palm, fingers, and main links. The system will identify their 2D coordinates, extract corresponding depth, and record the validity of 2D detections and depth measurements separately. Marker design will prioritize stable physical locations and clear identities, with each marker assigned to a component and a fixed location.

During capture, we will observe the same mechanical configuration from several calibrated viewpoints. Corresponding markers will be triangulated to recover 3D surface positions and fused with valid depth. Once mechanical information is available, we will establish the relationships between markers and actual joint and link states. For static capture, we will hold the arm stationary while moving the camera. For dynamic capture, we will use synchronized views and correctly record the relative camera poses.

We will transform the reconstructed results into the chest-camera coordinate frame at the corresponding capture time to produce candidate 3D labels. Quality checks will use held-out views and a small number of independent reference measurements. Structural consistency checks will be added once mechanical information is available. Manual review will focus on low-quality samples and random checks of the remaining samples. Incorrect results will be corrected or removed, and reliable portions will be used for training.

We will iterate through “automatic candidate generation → manual review or correction → training and evaluation” and record the actual review cost. The first deployed version will retain the necessary markers. If marker removal is required later, we will validate visual detection separately.

### 3.3 Mechanically Constrained Pose Network

We will first build a coordinate-based pose network using **2D coordinates, valid depth, and validity flags**, with initial training supervised by multiview-reconstructed 3D marker positions. Separate flags will indicate whether a 2D point was detected and whether its depth is available, preventing missing values from being treated as genuine observations.

Each training sample will pair a chest-camera observation with reviewed multiview 3D labels from the same moment. The network will predict 3D positions from the chest-camera observation, compare them with the reference positions of the same markers, and use the errors to update its parameters. Multiview labels will serve only as training supervision and evaluation references. During operation, the network will receive only observations extracted automatically from the chest-mounted RGB-D camera.

Observation-quality indicators will come from existing detection outputs and depth statistics, such as available detection scores, the proportion of valid depth samples, and local depth variation. We will initially use these indicators to filter observations and weight observation-consistency losses. Controlled experiments will then determine whether they also improve performance as additional network inputs. We will weight 3D supervision according to the quality of the multiview labels themselves. We will not add a separate quality-scoring network or treat uncalibrated scores as probabilities of correctness.

After establishing the coordinate network, we will add a shared lightweight RGB feature extractor. It will extract appearance, component-identity, and occlusion cues from the arm image region and fuse them with coordinate features. We will compare accuracy, stability, and runtime with and without RGB features, then select the deployed version according to the measured benefit. This will test the value of image information while keeping model complexity under control.

Once mechanical information is available, we will extend the model to structured pose estimation, predicting the arm’s overall position and orientation relative to the camera together with its joint states. Forward kinematics will then generate joint centers and link poses. Confirmed link lengths and connectivity will be built directly into the output representation.

We will verify the mechanical model against the physical device before fixing confirmed structural relationships. We will allow appropriate deviations for assembly tolerances, flexible deformation, or approximate joint coupling. Opening degree will be quantified between reference open and closed configurations using identifiable joint states or fingertip-to-palm geometry.

### 3.4 Rendering-Based Supervision with the Full Mechanical Model

Once an initial pose model is available and the mechanical information has been verified, we will update the 3D link models according to the predicted state, render them using the actual camera parameters, and compare the renderings with the original RGB-D observations. The full model will provide silhouette and surface information beyond sparse joint points, helping constrain link orientation, finger configuration, and depth ordering.

We will retain 3D label supervision and introduce three complementary observation comparisons. **2D reprojection** will project predicted surface markers into the image and compare them with detections of the same markers, checking image-position agreement. **Silhouette comparison** will compare the rendered arm shape with the visible silhouette, constraining link orientation and finger configuration. **Depth comparison** will compare visible model surfaces with valid measured depth, constraining depth ordering and spatial distance. Together, these signals will supplement supervision from sparse 3D markers.

The resulting errors will form a combined training objective. Differentiable kinematics and rendering will pass these errors back to the pose network to update its parameters, progressively bringing predictions closer to both the 3D references and the real observations. Verified mechanical dimensions and connectivity will remain fixed. Supervision terms will be weighted by validity and quality to prevent incorrect observations from dominating training.

We will account for occlusion, missing data, and incorrect correspondences when comparing silhouettes and depth. Surface markers and internal joint centers will be modeled separately. The RGB feature branch will supply network inputs, while rendering supervision will compare predicted states with observations; they serve different purposes. The initial phase will not require pixel-perfect agreement across the full RGB image.

We will first train with 3D labels and then progressively introduce rendering constraints. Controlled experiments will measure their effect on localization and stability. Hidden joints that cannot be determined from the current observations will remain marked as not observable. Structural plausibility or rendering agreement alone will not be treated as successful localization.

### 3.5 Single-Camera Operation and UNO Q Deployment

After deployment to the UNO Q, the device will automatically detect the arm, identify markers or keypoints, process depth, estimate pose, and compute kinematics. It will detect the target again after tracking is lost. Outputs will use the current camera coordinate frame and update as the camera moves. When valid observations are lost, the system will stop presenting old coordinates as current results.

We will use approximately one million parameters or fewer as the initial design budget for the coordinate-based pose network and validate its accuracy through training. Versions with an RGB branch will have their total parameter count and computational cost measured separately. All detection, image-processing, and inference components will be included in the UNO Q system-level performance evaluation.

## 4. Implementation Plan and Milestones

| Stage | Main activities | Deliverables and progression criteria |
|---|---|---|
| M1: Observation and label validation | Confirm marker placement; start static capture with the palm and one finger; validate multiview 3D marker labels; test UNO Q camera connectivity and basic processing | A small reviewable dataset, label-error report, and initial runtime measurements; confirmation that observation and labeling prerequisites are met |
| M2: Hand model training | Expand finger and pose coverage; complete initial pose training; add mechanical constraints and rendering supervision once structural information is available, and evaluate RGB features | A single-camera hand prototype, with measured accuracy benefits, runtime costs, and manual review effort |
| M3: Arm integration and deployment | Extend coverage to the main links and target joints; optimize the pipeline; test chest motion, occlusion, reacquisition, and sustained operation | An integrated UNO Q prototype, a common output interface, a performance report, and a defined operating scope |

We will begin with a small capture pilot and determine the subsequent dataset size and schedule from label quality, actual review time, and hardware performance. At each stage, we will retain direct geometric estimation with the same mechanical constraints as a baseline to test whether learning provides a measurable improvement.

## 5. Resource Requirements and Team Coordination

We need the robotic arm hardware team to apply agreed colors or fixed markers around the joints and assist with image and video collection. When we begin the later rendering and reprojection training stage, the hardware team will provide geometry models and dimensions matching the physical arm, together with joint motion definitions, so that we can construct the mechanical model and train with rendering supervision.

The project requires computing resources for model training.

## 6. Deliverables and Acceptance Evaluation

We will deliver:

1. **An automatic labeling and review workflow:** traceable multiview 3D data with documented label quality and valid coverage.
2. **A single-camera pose estimation prototype:** automatic outputs for target joint positions, main-link poses, and hand opening state.
3. **A common output interface:** explicit coordinate frames, units, timestamps, joint identities, validity, confidence information, and the distinction between observation and inference.
4. **A deployment and evaluation report:** UNO Q performance, supported scenarios, failure boundaries, and directions for further improvement.

| Evaluation dimension | Main checks |
|---|---|
| Label quality and data cost | Error against independent reference measurements, valid coverage, error rates, and manual correction time |
| 3D pose and hand opening | Position, orientation, and opening errors in the camera frame; aligned shape similarity will not substitute for absolute localization accuracy |
| Method contribution | Comparison of the geometric baseline and learned models, with separate evaluation of quality inputs, RGB features, and rendering constraints |
| Stability and failure handling | Jitter, valid-output rate, incorrect outputs reported as valid, and reacquisition performance |
| Complete-system operation | End-to-end latency, output rate, resource use, and sustained performance |

We will split evaluation data by pose group or capture session and cover chest motion, varying distances and backgrounds, finger occlusion, and missing depth. When evidence is insufficient, the system will report the affected state as not observable rather than treating a plausible but unsupported pose as reliable.

We will complete the feasibility phase first, then use measured results to establish formal accuracy, runtime, and coverage requirements. This will guide development toward a deployable, inspectable, and maintainable visual perception system for the robotic arm.
