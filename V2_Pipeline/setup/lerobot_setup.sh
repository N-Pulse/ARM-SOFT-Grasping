#!/usr/bin/env bash
#
# lerobot_setup.sh — One-stop installer & setup helper for LeRobot (macOS/Linux)
#
# Usage examples:
#   ./lerobot_setup.sh --option phospho
#   ./lerobot_setup.sh --option official \
#       --follower-port /dev/tty.usbmodem585A0076841 \
#       --leader-port /dev/tty.usbmodem575E0031751 \
#       --follower-id my_awesome_follower_arm \
#       --leader-id my_awesome_leader_arm
#
# If you omit ports/ids, the script will prompt you.
#
set -Eeuo pipefail

# ------------- styling helpers -------------
bold() { printf "\033[1m%s\033[0m\n" "$*"; }
info() { printf "[INFO] %s\n" "$*"; }
warn() { printf "[WARN] %s\n" "$*"; }
err()  { printf "\033[31m[ERROR]\033[0m %s\n" "$*" >&2; }

cleanup() { :; }
trap cleanup EXIT
trap 'err "Command failed on line $LINENO"; exit 1' ERR

# ------------- defaults -------------
CHOICE=""
CONDA_ENV="lerobot"
PYVER="3.10"
REPO_URL="https://github.com/huggingface/lerobot.git"
REPO_DIR="${HOME}/src/lerobot"
FOLLOWER_PORT=""
LEADER_PORT=""
FOLLOWER_ID="my_awesome_follower_arm"
LEADER_ID="my_awesome_leader_arm"

# ------------- arg parsing -------------
usage() {
  cat <<'USAGE'
lerobot_setup.sh

Options:
  --option [phospho|official]    Choose installation path (required).
  --follower-port PORT           Serial port for follower (e.g., /dev/tty.usbmodemXXXX).
  --leader-port PORT             Serial port for leader.
  --follower-id NAME             Unique name for follower (default: my_awesome_follower_arm).
  --leader-id NAME               Unique name for leader (default: my_awesome_leader_arm).
  --repo-dir DIR                 Where to clone lerobot (default: ~/src/lerobot).
  --help                         Show this help.

Examples:
  ./lerobot_setup.sh --option phospho
  ./lerobot_setup.sh --option official --follower-port /dev/tty.usbmodem585A0076841

USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --option) CHOICE="${2:-}"; shift 2;;
    --follower-port) FOLLOWER_PORT="${2:-}"; shift 2;;
    --leader-port)   LEADER_PORT="${2:-}"; shift 2;;
    --follower-id)   FOLLOWER_ID="${2:-}"; shift 2;;
    --leader-id)     LEADER_ID="${2:-}"; shift 2;;
    --repo-dir)      REPO_DIR="${2:-}"; shift 2;;
    --help|-h) usage; exit 0;;
    *) err "Unknown argument: $1"; usage; exit 1;;
  esac
done

# ------------- helpers -------------
need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    err "Missing required command: $1";
    return 1;
  }
}

ensure_git() {
  if ! command -v git >/dev/null 2>&1; then
    warn "git not found; attempting to install via conda..."
    need_cmd conda || { err "conda required to install git automatically"; return 1; }
    conda install -y git || { err "Failed to install git"; return 1; }
  fi
}

ensure_conda_in_shell() {
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "${CONDA_BASE}/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  err "Could not source conda.sh. Make sure Miniconda/Mambaforge is installed and 'conda init' has been run."
  err "See: https://docs.conda.io/projects/conda/en/latest/user-guide/install/"
  return 1
}

activate_env() {
  local env_name="$1"
  ensure_conda_in_shell
  if conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
    info "Activating existing env: ${env_name}"
  else
    info "Creating env '${env_name}' (python=${PYVER})"
    conda create -y -n "${env_name}" "python=${PYVER}"
  fi
  conda activate "${env_name}"
}

clone_or_update_repo() {
  local url="$1" dir="$2"
  mkdir -p "$(dirname "${dir}")"
  if [[ -d "${dir}/.git" ]]; then
    info "Updating existing repo at ${dir}"
    git -C "${dir}" pull --ff-only
  else
    info "Cloning ${url} into ${dir}"
    git clone "${url}" "${dir}"
  fi
}

prompt_if_empty() {
  local varname="$1" prompt="$2"
  local val="${!varname}"
  if [[ -z "$val" ]]; then
    read -r -p "$prompt: " val
    printf -v "$varname" '%s' "$val"
  fi
}

# ------------- Option 1: Phospho AI guided -------------
run_phospho() {
  bold "Option 1 — Phospho AI guided"
  need_cmd curl
  info "Installing phosphobot..."
  # The install script may ask for confirmation — it runs non-interactively with -fsSL
  bash -lc 'curl -fsSL https://raw.githubusercontent.com/phospho-app/phosphobot/main/install.sh | bash'
  info "Launching phosphobot..."
  phosphobot run
  bold "Phospho flow completed."
}

# ------------- Option 2: Official documentation path -------------
install_lerobot() {
  bold "I — Install LeRobot (conda + pip)"
  need_cmd conda
  activate_env "${CONDA_ENV}"
  info "Installing ffmpeg via conda-forge..."
  conda install -y -c conda-forge ffmpeg

  ensure_git

  clone_or_update_repo "${REPO_URL}" "${REPO_DIR}"
  cd "${REPO_DIR}"

  info "Upgrading pip tooling..."
  python -m pip install --upgrade pip setuptools wheel

  # Order chosen to keep editable install last (so it "wins")
  info "Installing PyPI packages 'lerobot' and extras..."
  pip install -U "lerobot" "lerobot[all]"
  info "Editable install of local repo..."
  pip install -e .
  info "Adding Feetech extras (editable)..."
  pip install -e ".[feetech]"

  bold "LeRobot installation complete."
}

configure_motors_find_ports() {
  bold "II — Configure the motors (find ports)"
  info "Connect a MotorBus via USB and power. The tool will guide you."
  info "When prompted, disconnect/ reconnect as instructed."
  need_cmd lerobot-find-port
  lerobot-find-port
  bold "Port discovery finished."
}

setup_motors_ids() {
  bold "III — Set the motors ID (Follower arm)"
  prompt_if_empty FOLLOWER_PORT "Enter follower port (e.g., /dev/tty.usbmodem585A0076841)"
  need_cmd lerobot-setup-motors
  lerobot-setup-motors --robot.type=so101_follower --robot.port="${FOLLOWER_PORT}"
  bold "Follower motor IDs configured."

  bold "IV — Repeat for Leader arm"
  prompt_if_empty LEADER_PORT "Enter leader port (e.g., /dev/tty.usbmodem575E0031751)"
  lerobot-setup-motors --teleop.type=so101_leader --teleop.port="${LEADER_PORT}"
  bold "Leader motor IDs configured."
}

calibrate_both() {
  bold "V — Calibrate"
  need_cmd lerobot-calibrate

  info "Calibrating follower..."
  lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port="${FOLLOWER_PORT}" \
    --robot.id="${FOLLOWER_ID}"

  info "Calibrating leader..."
  lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port="${LEADER_PORT}" \
    --teleop.id="${LEADER_ID}"

  bold "Calibration complete for both arms."
}

run_official_flow() {
  install_lerobot
  configure_motors_find_ports
  setup_motors_ids
  calibrate_both
}

# ------------- main -------------
if [[ -z "${CHOICE}" ]]; then
  bold "No --option provided."
  echo "Choose an option:"
  echo "  1) Phospho AI guided"
  echo "  2) Official documentation"
  read -r -p "Enter 1 or 2: " ans
  case "$ans" in
    1) CHOICE="phospho" ;;
    2) CHOICE="official" ;;
    *) err "Invalid choice"; exit 1 ;;
  esac
fi

case "${CHOICE}" in
  phospho|phospho-ai|phospho_ai)
    run_phospho
    ;;
  official|docs|documentation)
    run_official_flow
    ;;
  *)
    err "Unknown --option '${CHOICE}'"; usage; exit 1;;
esac

bold "All done ✅"
