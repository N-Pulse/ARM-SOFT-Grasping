#!/usr/bin/env bash
# Create ./.venv and install everything the model needs.
#
#   bash setup_env.sh
#   source .venv/bin/activate
#   python -m armsoft.selftest
#
# Re-running is safe: it reuses an existing .venv and upgrades in place.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
PY="${PYTHON:-python3}"

echo "==> Using $($PY -V) at $(command -v "$PY")"

if [ ! -d "$VENV" ]; then
    echo "==> Creating virtual environment in $VENV"
    "$PY" -m venv "$VENV" || {
        echo "venv creation failed — on Debian/Ubuntu: sudo apt install python3-venv"
        exit 1
    }
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

# torch AND torchvision must come from the same index, or ultralytics fails at
# runtime with "operator torchvision::nms does not exist".
echo "==> Installing CPU builds of torch and torchvision"
"$VENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing the remaining requirements"
"$VENV/bin/pip" install -r "$HERE/requirements.txt"

echo
echo "==> Verifying"
"$VENV/bin/python" - <<'PYCHECK' || {
import numpy, scipy, cv2, open3d, torch, torchvision, ultralytics
torch.ops.torchvision.nms          # fails loudly on a torch/torchvision mismatch
print("all imports OK")
PYCHECK
    cat <<'MSG'

Import failed.  The usual cause on a bare Linux server is Open3D's OpenGL
dependency; install the system libraries and re-run this check:

    sudo apt install libegl1 libgl1 libglib2.0-0

MSG
    exit 1
}

cat <<MSG

Done.  Next:

    source "$VENV/bin/activate"
    python -m armsoft.selftest
    python -m armsoft.run_inference --frames 1

MSG
