#!/usr/bin/env bash
# Zip the training/ CODE for Colab  ->  training/colab/training_code.zip
# Re-make it after every code change and upload it to Google Drive: MyDrive/npulse/
#
#   bash training/colab/make_code_zip.sh
#
# Files are taken from git (tracked + new, non-ignored), so datasets, runs,
# weights, .venv-download, the FiftyOne cache and the zips themselves are
# never included.  training/results/ is left out too (not needed to train).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && (pwd -W 2>/dev/null || pwd))"
REPO="$(cd "$HERE/../.." && (pwd -W 2>/dev/null || pwd))"
OUT="training/colab/training_code.zip"
PY="${PYTHON:-python}"

cd "$REPO"
git ls-files --cached --others --exclude-standard -- training \
    | grep -v '^training/results/' \
    | "$PY" -c '
import sys
import zipfile
from pathlib import Path

out = Path(sys.argv[1])
banned = ("/datasets/", "/runs/", "/weights/", ".venv", "_fiftyone_cache")
files = [f for f in sys.stdin.read().splitlines()
         if f and Path(f).is_file() and not f.endswith((".zip", ".pt", ".onnx", ".tflite"))]
bad = [f for f in files if any(b in f for b in banned)]
if bad:
    sys.exit(f"refusing to zip ignored-looking files: {bad[:5]}")
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(files):
        z.write(f, f)
print(f"{len(files)} files -> {out} ({out.stat().st_size / 1e3:.0f} kB)")
' "$OUT"

echo "Upload to Google Drive: MyDrive/npulse/training_code.zip"
