#!/usr/bin/env bash
# Zip ONE YOLO dataset for Colab:  training/datasets/<name>/  ->  training/colab/<name>.zip
# The zip root is <name>/, so the notebook unzips it straight into training/datasets/.
# Upload the result ONCE to Google Drive: MyDrive/npulse/data/<name>.zip
#
#   bash training/colab/make_data_zip.sh                 # openimages4 (default)
#   bash training/colab/make_data_zip.sh inhouse_d405    # any other dataset folder
#
# Needs only Python (zipfile is in the standard library); set PYTHON=... to pick one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && (pwd -W 2>/dev/null || pwd))"
TRAINING="$(dirname "$HERE")"
NAME="${1:-openimages4}"
SRC="$TRAINING/datasets/$NAME"
OUT="$HERE/$NAME.zip"
PY="${PYTHON:-python}"

if [ ! -d "$SRC/images" ] || [ ! -d "$SRC/labels" ]; then
    echo "no YOLO dataset (images/ + labels/) at $SRC" >&2
    exit 1
fi

"$PY" - "$SRC" "$OUT" <<'PYZIP'
import sys
import zipfile
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:   # JPEGs do not compress
    for p in sorted(src.rglob("*")):
        # *.cache = Ultralytics label caches with local absolute paths; rebuilt on Colab
        if p.is_file() and p.suffix != ".cache":
            z.write(p, p.relative_to(src.parent).as_posix())
            n += 1
print(f"{n} files -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
PYZIP

echo "Upload to Google Drive: MyDrive/npulse/data/$NAME.zip"
