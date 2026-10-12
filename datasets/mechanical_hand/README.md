# Mechanical-hand dataset

Large images, videos and generated labels are intentionally not committed.
Keep local data in this directory using the following layout:

```text
raw_videos/          original recordings, immutable
extracted_frames/    selected, de-duplicated frames
annotations/         canonical 17-keypoint source annotations
yolo_pose/           generated YOLO Pose dataset
splits/              versioned train/val/test recording lists
```

Split by recording session, never by randomly mixing adjacent video frames.
Only `README.md` and split manifest text files belong in Git.
