#!/usr/bin/env bash
# Reproduce every porting result from a fresh clone, in order.
#
#   bash setup_env.sh                        # once: host .venv (see main README §2)
#   bash porting/reproduce.sh                # ~45 min, mostly emulated pip installs
#   TFENV=/path/to/tfenv bash porting/reproduce.sh   # also redo the TFLite export
#
# Host requirements: Linux x86-64, python3, curl, qemu-aarch64-static
# (Debian/Ubuntu: apt install qemu-user-static).  No root, no Docker.
# Outputs land in porting/results/.  Every step can also be run by hand —
# porting/README.md lists them one by one.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
HOSTPY=.venv/bin/python
[ -x "$HOSTPY" ] || { echo "no .venv — run: bash setup_env.sh"; exit 1; }
ENV="bash porting/unoq_env.sh"
R=porting/results; mkdir -p "$R"
step() { echo; echo "=== $*"; }

step "1/8 host: export tools (side dir, the .venv is not modified)"
[ -d porting/cache/hostpkgs/onnx ] || .venv/bin/pip install -q --target porting/cache/hostpkgs onnx onnxruntime onnxslim ai-edge-litert
for p in numpy numpy.libs sympy mpmath isympy.py typing_extensions.py packaging; do
    rm -rf "porting/cache/hostpkgs/$p"            # never shadow the .venv's own copies
done
export PYTHONPATH=porting/cache/hostpkgs YOLO_OFFLINE=1

step "2/8 host: export the classifier to ONNX (fp32 + int8)"
$HOSTPY porting/export_classifier.py --int8
if [ -n "${TFENV:-}" ]; then
    step "2b host: ONNX -> TFLite"
    "$TFENV/bin/python" porting/export_tflite.py --int8
fi

step "3/8 host: record simulated RGB-D test sequences"
PYTHON=$HOSTPY bash porting/record_sim.sh

step "4/8 host: classifier parity + replays (reference results)"
$HOSTPY porting/test_classifier.py --json $R/classifier_x86_64.json | grep -v '^\[' || true
$HOSTPY porting/run_replay.py --quiet --o3d real --classifier yolo \
    --jsonl $R/replay_x86_64_real_yolo.jsonl --summary-json $R/replay_x86_64_real_yolo.json
$HOSTPY porting/run_replay.py --quiet --o3d lite --classifier onnx \
    --jsonl $R/replay_x86_64_lite_onnx.jsonl --summary-json $R/replay_x86_64_lite_onnx.json
unset PYTHONPATH

step "5/8 arm64: build the emulated UNO Q (Debian 13, Cortex-A53)"
$ENV create
$ENV install full
$ENV install minimal

step "6/8 arm64: dependency check"
$ENV run python porting/check_env.py --json > $R/check_env_aarch64.json
$ENV run python porting/check_env.py

step "7/8 arm64: classifier + replays"
$ENV run python porting/test_classifier.py --json $R/classifier_aarch64.json | grep -v '^\[' || true
$ENV run python porting/run_replay.py --quiet --o3d real --classifier yolo \
    --jsonl $R/replay_aarch64_real_yolo.jsonl --summary-json $R/replay_aarch64_real_yolo.json
UNOQ_VENV=minimal $ENV run python porting/run_replay.py --quiet --o3d lite --classifier onnx \
    --jsonl $R/replay_aarch64_minimal.jsonl --summary-json $R/replay_aarch64_minimal.json

step "8/8 compare: x86 PyTorch+Open3D  vs  arm64 minimal (ONNX, no Open3D, no torch)"
$HOSTPY porting/compare_results.py $R/replay_x86_64_real_yolo.jsonl $R/replay_aarch64_minimal.jsonl \
    --json $R/diff_x86yolo_vs_arm_minimal.json
echo; echo "done — see porting/README.md for how to read these numbers"
