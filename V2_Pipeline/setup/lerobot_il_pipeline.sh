#!/usr/bin/env bash
#
# lerobot_il_pipeline.sh — Teleop, Cameras, Recording, Colab Training, Replay for LeRobot
#
# This script wraps common LeRobot workflows. It assumes you already installed LeRobot
# and focuses on teleoperation + dataset workflow. It also prints a Colab block
# for training when you don't have a powerful local GPU.
#
# Usage examples:
#   ./lerobot_il_pipeline.sh --teleop
#   ./lerobot_il_pipeline.sh --find-cameras
#   ./lerobot_il_pipeline.sh --teleop-with-cameras
#   ./lerobot_il_pipeline.sh --hf-login --hf-token $HUGGINGFACE_TOKEN
#   ./lerobot_il_pipeline.sh --record --episodes 5 --dataset-repo record-test
#   ./lerobot_il_pipeline.sh --replay --episode 0 --dataset-repo record-test
#   ./lerobot_il_pipeline.sh --print-colab-guide
#
set -Eeuo pipefail

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
info() { printf "[INFO] %s\n" "$*"; }
warn() { printf "\033[33m[WARN]\033[0m %s\n" "$*"; }
err()  { printf "\033[31m[ERROR]\033[0m %s\n" "$*" >&2; }

cleanup() { :; }
trap cleanup EXIT
trap 'err "Command failed on line $LINENO"; exit 1' ERR

# ---------- defaults ----------
ROBOT_FOLLOWER_TYPE="${ROBOT_FOLLOWER_TYPE:-so101_follower}"
ROBOT_LEADER_TYPE="${ROBOT_LEADER_TYPE:-so101_leader}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/tty.usbmodem58760431541}"
LEADER_PORT="${LEADER_PORT:-/dev/tty.usbmodem58760431551}"
FOLLOWER_ID="${FOLLOWER_ID:-my_awesome_follower_arm}"
LEADER_ID="${LEADER_ID:-my_awesome_leader_arm}"

CAM_INDEX="${CAM_INDEX:-0}"
CAM_FPS="${CAM_FPS:-30}"
CAM_WIDTH="${CAM_WIDTH:-1920}"
CAM_HEIGHT="${CAM_HEIGHT:-1080}"

DATASET_REPO="${DATASET_REPO:-so101_test}"
NUM_EPISODES="${NUM_EPISODES:-5}"
SINGLE_TASK="${SINGLE_TASK:-Grab the black cube}"

REPLAY_EPISODE="${REPLAY_EPISODE:-0}"

HF_TOKEN="${HF_TOKEN:-}"
HF_USER="${HF_USER:-}"
POLICY_REPO="${POLICY_REPO:-my_policy}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/act_${DATASET_REPO}}"
JOB_NAME="${JOB_NAME:-act_${DATASET_REPO}}"

DO_TELEOP=false
DO_FIND_CAMERAS=false
DO_TELEOP_WITH_CAMERAS=false
DO_HF_LOGIN=false
DO_RECORD=false
DO_REPLAY=false
DO_PRINT_COLAB=false

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { err "Missing required command: $1"; return 1; }
}

detect_hf_user() {
  need_cmd huggingface-cli
  HF_USER="$(huggingface-cli whoami 2>/dev/null | head -n 1 || true)"
  if [[ -z "${HF_USER}" ]]; then
    warn "HF user not found. Run with --hf-login --hf-token <TOKEN> or ensure 'huggingface-cli whoami' works."
  fi
}

teleop() {
  bold "I — Teleoperate"
  need_cmd lerobot-teleoperate
  info "Using follower: type=${ROBOT_FOLLOWER_TYPE}, port=${FOLLOWER_PORT}, id=${FOLLOWER_ID}"
  info "Using leader:   type=${ROBOT_LEADER_TYPE},   port=${LEADER_PORT},   id=${LEADER_ID}"
  lerobot-teleoperate \
    --robot.type="${ROBOT_FOLLOWER_TYPE}" \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id="${FOLLOWER_ID}" \
    --teleop.type="${ROBOT_LEADER_TYPE}" \
    --teleop.port="${LEADER_PORT}" \
    --teleop.id="${LEADER_ID}"
  bold "Teleoperation finished."
}

find_cameras() {
  bold "II — Find cameras"
  warn "On macOS, Intel RealSense can be unstable; if you hit a power-state error try with sudo."
  need_cmd lerobot-find-cameras
  info "Scanning OpenCV cameras..."
  lerobot-find-cameras opencv || true
  info "To scan Intel RealSense cameras:"
  info "  lerobot-find-cameras realsense   # may require sudo on macOS"
  bold "Camera scan complete."
}

teleop_with_cameras() {
  bold "III — Teleoperate with cameras + rerun visualization"
  need_cmd lerobot-teleoperate
  local spec
  spec="{ front: {type: opencv, index_or_path: ${CAM_INDEX}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}}}"
  info "Camera spec: ${spec}"
  info "Using follower: type=koch_follower (example), port=${FOLLOWER_PORT}, id=${FOLLOWER_ID}"
  info "Using leader:   type=koch_leader   (example), port=${LEADER_PORT}, id=${LEADER_ID}"
  lerobot-teleoperate \
    --robot.type=koch_follower \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id="${FOLLOWER_ID}" \
    --robot.cameras="${spec}" \
    --teleop.type=koch_leader \
    --teleop.port="${LEADER_PORT}" \
    --teleop.id="${LEADER_ID}" \
    --display_data=true
  bold "Teleop-with-cameras finished."
}

hf_login() {
  bold "IV — Hugging Face login"
  need_cmd huggingface-cli
  if [[ -n "${HF_TOKEN}" ]]; then
    info "Logging into Hugging Face CLI with provided token..."
    huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential
  fi
  detect_hf_user()
  if [[ -z "${HF_USER}" ]]; then
    err "HF user unknown. Ensure 'huggingface-cli whoami' works or pass --hf-token."
    exit 1
  fi
  bold "Logged in as: ${HF_USER}"
}

record_dataset() {
  bold "V — Record a dataset"
  need_cmd lerobot-record
  detect_hf_user
  if [[ -z "${HF_USER}" ]]; then
    err "HF user unknown. Run --hf-login first."
    exit 1
  fi
  local spec
  spec="{ front: {type: opencv, index_or_path: ${CAM_INDEX}, width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${CAM_FPS}}}"
  info "Recording ${NUM_EPISODES} episodes to ${HF_USER}/${DATASET_REPO}"
  lerobot-record \
    --robot.type="${ROBOT_FOLLOWER_TYPE}" \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id="${FOLLOWER_ID}" \
    --robot.cameras="${spec}" \
    --teleop.type="${ROBOT_LEADER_TYPE}" \
    --teleop.port="${LEADER_PORT}" \
    --teleop.id="${LEADER_ID}" \
    --display_data=true \
    --dataset.repo_id="${HF_USER}/${DATASET_REPO}" \
    --dataset.num_episodes="${NUM_EPISODES}" \
    --dataset.single_task="${SINGLE_TASK}"
  bold "Recording complete. Data will be at ~/.cache/huggingface/lerobot/${HF_USER}/${DATASET_REPO} and on your HF hub."
}

replay_episode() {
  bold "VII — Replay an episode"
  need_cmd lerobot-replay
  detect_hf_user
  if [[ -z "${HF_USER}" ]]; then
    err "HF user unknown. Run --hf-login first."
    exit 1
  fi
  info "Replaying episode ${REPLAY_EPISODE} from ${HF_USER}/${DATASET_REPO}"
  lerobot-replay \
    --robot.type="${ROBOT_FOLLOWER_TYPE}" \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id="${FOLLOWER_ID}" \
    --dataset.repo_id="${HF_USER}/${DATASET_REPO}" \
    --dataset.episode="${REPLAY_EPISODE}"
  bold "Replay done."
}

print_colab_guide() {
  bold "VI — Google Colab Training Guide (no local GPU needed)"
  cat <<'COLAB'
=================== COPY INTO A GOOGLE COLAB NOTEBOOK ===================

# 1) Runtime → Change runtime type → GPU (T4/A100/etc.)
# 2) Install LeRobot + deps
!pip -q install -U "lerobot" "lerobot[all]" wandb

# 3) Login to Hugging Face (paste a write token)
import os
try:
    from google.colab import userdata  # Colab-only
    token = userdata.get('HUGGINGFACE_TOKEN')
except Exception:
    token = None
if token is None:
    token = input("Paste your HF token: ").strip()

!huggingface-cli login --token "$token" --add-to-git-credential

# 4) Get your HF username
import subprocess
HF_USER = subprocess.run("huggingface-cli whoami | head -n 1", shell=True, capture_output=True, text=True).stdout.strip()
print("HF_USER =", HF_USER)

# 5) Train ACT policy on your recorded dataset
DATASET_REPO = "so101_test"     # <- set to the repo you used locally
OUTPUT_DIR   = f"outputs/train/act_{DATASET_REPO}"
JOB_NAME     = f"act_{DATASET_REPO}"
POLICY_REPO  = "my_policy"      # <- name of the model repo to create on HF

!lerobot-train \
  --dataset.repo_id=${HF_USER}/{DATASET_REPO} \
  --policy.type=act \
  --output_dir={OUTPUT_DIR} \
  --job_name={JOB_NAME} \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/{POLICY_REPO}

# 6) (Optional) Resume from last checkpoint
#!lerobot-train \
#  --config_path={OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json \
#  --resume=true

# 7) Upload the final checkpoint (if push_to_hub was disabled)
#!huggingface-cli upload ${HF_USER}/{JOB_NAME} \
#  {OUTPUT_DIR}/checkpoints/last/pretrained_model

# 8) After training, you can run inference locally by pointing to the HF repo in --policy.path
#    e.g., --policy.path=${HF_USER}/{POLICY_REPO}

=========================================================================
COLAB
}

usage() {
  cat <<'USAGE'
lerobot_il_pipeline.sh — Wrap common LeRobot imitation learning steps.

Flags (actions):
  --teleop                 Run basic teleoperation (auto-calibrates if needed).
  --find-cameras           List available cameras (OpenCV; try 'realsense' too).
  --teleop-with-cameras    Teleop with camera viz and joint plots (uses Koch example).
  --hf-login               Log into Hugging Face CLI (use with --hf-token TOKEN).
  --record                 Record a dataset and push to HF hub.
  --replay                 Replay one recorded episode from a dataset.
  --print-colab-guide      Print a ready-to-copy Google Colab training block.

Options:
  --follower-port PATH     Default: /dev/tty.usbmodem58760431541
  --leader-port PATH       Default: /dev/tty.usbmodem58760431551
  --follower-id NAME       Default: my_awesome_follower_arm
  --leader-id NAME         Default: my_awesome_leader_arm
  --robot-follower TYPE    Default: so101_follower
  --robot-leader TYPE      Default: so101_leader

  --cam-index N            Default: 0
  --cam-fps N              Default: 30
  --cam-width N            Default: 1920
  --cam-height N           Default: 1080

  --hf-token TOKEN         HF token for CLI login.
  --dataset-repo NAME      Dataset repository name suffix (Default: so101_test)
  --episodes N             Number of episodes to record (Default: 5)
  --single-task TEXT       Task description for the dataset

  --replay-episode N       Episode index to replay (Default: 0)

  --policy-repo NAME       HF model repo name (Default: my_policy)
  --output-dir PATH        Training output dir (Default: outputs/train/act_${DATASET_REPO})
  --job-name NAME          Training job name (Default: act_${DATASET_REPO})

Examples:
  ./lerobot_il_pipeline.sh --hf-login --hf-token $HUGGINGFACE_TOKEN
  ./lerobot_il_pipeline.sh --record --dataset-repo record-test --episodes 5
  ./lerobot_il_pipeline.sh --replay --dataset-repo record-test --replay-episode 0
  ./lerobot_il_pipeline.sh --print-colab-guide

USAGE
}

if [[ $# -eq 0 ]]; then
  bold "No flags provided. Opening quick menu..."
  PS3="Select a step: "
  select opt in "Teleop" "Find Cameras" "Teleop + Cameras" "HF Login" "Record Dataset" "Replay Episode" "Print Colab Guide" "Quit"; do
    case "$REPLY" in
      1) DO_TELEOP=true; break;;
      2) DO_FIND_CAMERAS=true; break;;
      3) DO_TELEOP_WITH_CAMERAS=true; break;;
      4) DO_HF_LOGIN=true; break;;
      5) DO_RECORD=true; break;;
      6) DO_REPLAY=true; break;;
      7) DO_PRINT_COLAB=true; break;;
      8) exit 0;;
      *) echo "Invalid selection";;
    esac
  done
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --teleop) DO_TELEOP=true; shift;;
    --find-cameras) DO_FIND_CAMERAS=true; shift;;
    --teleop-with-cameras) DO_TELEOP_WITH_CAMERAS=true; shift;;
    --hf-login) DO_HF_LOGIN=true; shift;;
    --record) DO_RECORD=true; shift;;
    --replay) DO_REPLAY=true; shift;;
    --print-colab-guide) DO_PRINT_COLAB=true; shift;;

    --follower-port) FOLLOWER_PORT="${2:-}"; shift 2;;
    --leader-port)   LEADER_PORT="${2:-}"; shift 2;;
    --follower-id)   FOLLOWER_ID="${2:-}"; shift 2;;
    --leader-id)     LEADER_ID="${2:-}"; shift 2;;
    --robot-follower) ROBOT_FOLLOWER_TYPE="${2:-}"; shift 2;;
    --robot-leader)   ROBOT_LEADER_TYPE="${2:-}"; shift 2;;

    --cam-index)  CAM_INDEX="${2:-}"; shift 2;;
    --cam-fps)    CAM_FPS="${2:-}"; shift 2;;
    --cam-width)  CAM_WIDTH="${2:-}"; shift 2;;
    --cam-height) CAM_HEIGHT="${2:-}"; shift 2;;

    --hf-token) HF_TOKEN="${2:-}"; shift 2;;
    --dataset-repo) DATASET_REPO="${2:-}"; shift 2;;
    --episodes) NUM_EPISODES="${2:-}"; shift 2;;
    --single-task) SINGLE_TASK="${2:-}"; shift 2;;

    --replay-episode) REPLAY_EPISODE="${2:-}"; shift 2;;

    --policy-repo) POLICY_REPO="${2:-}"; shift 2;;
    --output-dir)  OUTPUT_DIR="${2:-}"; shift 2;;
    --job-name)    JOB_NAME="${2:-}"; shift 2;;

    --help|-h) usage; exit 0;;
    *) err "Unknown argument: $1"; usage; exit 1;;
  esac
done

$DO_TELEOP && teleop
$DO_FIND_CAMERAS && find_cameras
$DO_TELEOP_WITH_CAMERAS && teleop_with_cameras
$DO_HF_LOGIN && hf_login
$DO_RECORD && record_dataset
$DO_REPLAY && replay_episode
$DO_PRINT_COLAB && print_colab_guide

bold "All done ✅"
