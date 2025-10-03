````markdown
# Helper Scripts #
# Make scripts executable
chmod +x lerobot_setup.sh lerobot_il_pipeline.sh
````

---

## `setup/lerobot_setup.sh`

Configuration for the follower and the leader arm + calibration:

```bash
# Option 1: Phospho AI guided
./lerobot_setup.sh --option phospho

# Option 2: Official path (will guide you through install, ports, IDs, calibration)
./lerobot_setup.sh --option official \
  --follower-port /dev/tty.usbmodemXXXX \
  --leader-port   /dev/tty.usbmodemYYYY \
  --follower-id   my_awesome_follower_arm \
  --leader-id     my_awesome_leader_arm
```

---

## `setup/lerobot_il_pipeline.sh`

Script for the full training pipeline:

```bash
# Teleoperate
./lerobot_il_pipeline.sh --teleop \
  --follower-port /dev/tty.usbmodemXXXX \
  --leader-port   /dev/tty.usbmodemYYYY

# Find cameras
./lerobot_il_pipeline.sh --find-cameras

# Login to Hugging Face (for dataset uploads)
./lerobot_il_pipeline.sh --hf-login --hf-token "$HUGGINGFACE_TOKEN"

# Record a small dataset and push to Hub
./lerobot_il_pipeline.sh --record --dataset-repo so101_test --episodes 5

# Get a ready-to-paste Google Colab training cell (train ACT in the cloud)
./lerobot_il_pipeline.sh --print-colab-guide

# Replay an episode locally
./lerobot_il_pipeline.sh --replay --dataset-repo so101_test --replay-episode 0
```

---

# Trouble Shooting

* **Ports/IDs:** Use the exact serial ports for your follower/leader; keep IDs consistent across teleop/record/eval.
* **Colab path:** Training runs in Colab with `policy.device=cuda`; no local GPU required.
* **RealSense on macOS:** If you see a power-state error, try `lerobot-find-cameras realsense` with `sudo`.

