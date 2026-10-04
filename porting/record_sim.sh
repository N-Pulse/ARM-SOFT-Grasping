#!/usr/bin/env bash
# (Re)create the recorded RGB-D test sequences in porting/testdata/rec/.
# They are .npz (git-ignored), 20 frames each, ~16 MB per sequence.
#
#   bash porting/record_sim.sh                  # on any machine with the model installed
#
# Real D405 recordings belong next to them: copy a `--record` directory into
# porting/testdata/rec/<name>/ and run_replay.py picks it up automatically.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
R=porting/testdata/rec
PY="${PYTHON:-python}"
rec() { local name=$1; shift; rm -rf "$R/$name"
        "$PY" -m armsoft --source sim --frames 20 --record "$R/$name" "$@" >/dev/null
        echo "$R/$name: $(ls "$R/$name" | wc -l) frames"; }
rec cylinder       --sim-shape cylinder
rec cuboid         --sim-shape cuboid --sim-yaw 30
rec cylinder_worms --sim-shape cylinder --sim-worms 12
