# Model artifacts

Model weights and downloaded task bundles are not stored in normal Git.
Publish reproducible weights through Git LFS or a versioned release and record:

- source commit;
- dataset version and split manifests;
- training configuration;
- evaluation metrics;
- weight checksum.

The existing `shape_classifier.pt` belongs to the grasp pipeline. Mechanical
hand pose weights should use names such as `mechanical_hand_pose_v1.pt`.
