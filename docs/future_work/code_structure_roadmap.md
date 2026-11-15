# Code structure roadmap

Status: **incremental future refactoring**

## Completed foundation

The first two structural changes are now in place:

1. Dataset Studio keeps `tools/dataset_web.py` as its stable launch command, while HTTP routing is isolated in `armsoft/studio/`.
2. Detection, segmentation, and pose results share a task-independent contract in `armsoft/perception/`, including explicit depth validity and camera-space XYZ.

These changes preserve the existing pages, endpoints, storage layout, and startup command.

## Next priorities

### 1. Split the remaining Studio state by workflow

Move capture, sessions, preprocessing, annotation, dataset export, training, and live-test lifecycle code into small services under `armsoft/studio/`. Keep one coordinator that owns the camera and shared job state.

Do this workflow by workflow with API regression tests; avoid a single large rewrite.

### 2. Introduce versioned API and artifact contracts

Define schemas for web job status, perception frames, dataset manifests, training runs, point clouds, and failure cases. Validate data at module boundaries so frontend and backend changes fail clearly instead of producing partially compatible files.

### 3. Simplify the legacy core package

Review `armsoft/core/` after the active pipelines are stable. Keep proven geometry and grasping logic, move reusable pieces to clearly named packages, and delete abandoned experimental paths only after tests show they have no callers.

### 4. Unify training entry points

Keep one supported local/Colab training interface and have the web page call the same Python service. Dataset version, initial weights, hyperparameters, output directory, and Git revision should always be recorded in one run manifest.

### 5. Separate desktop and target-device runtimes

Treat data preparation/training and deployed inference as different runtime profiles. The desktop profile may include annotation and training dependencies; the target profile should contain only camera, preprocessing, selected model runtime, perception contracts, and outputs.

### 6. Retire compatibility wrappers carefully

The old helper imports and command paths should remain while teammates migrate. Remove them only after documentation, tests, and downstream scripts use the new packages and a deprecation window has passed.

## Acceptance criteria for future refactors

- Existing Dataset Studio URLs and launch commands continue to work or have an explicit migration guide.
- Raw RGB-D data is never duplicated or overwritten by refactoring.
- Unit tests cover each extracted service and versioned contract.
- Camera-free simulation can validate the complete web workflow.
- Architecture and user guides are updated in the same change.
