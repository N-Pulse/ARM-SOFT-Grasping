"""
training/register_in_studio.py
==============================
Make a trained detection run selectable in the Dataset Studio Live Test:

    python training/register_in_studio.py --run iter1 --name multiclass_det_iter1

copies training/runs/detect/<run>/ into <data root>/runs/object_segmentation/<name>/
    weights/best.pt
    results.csv               so the Studio shows mAP50 / precision / recall
    armsoft_training.json     same metadata the Studio writes after its own training
The data root is the Studio's (armsoft.data.data_root(): ARM_SOFT_DATA_ROOT or
../ARM-SOFT-data).  Nothing in the repo changes; delete the folder to undo.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.paths import RUNS_DIR  # noqa: E402

# same rule as the Studio's run names (tools/dataset_web.py SAFE)
SAFE_NAME = re.compile(r"^(?!\.{1,2}$)(?!\.)[\w.-]{1,80}$")


def studio_metadata(run_dir: Path, target: Path) -> dict:
    """armsoft_training.json content for a run trained outside the Studio."""
    args = {}
    if (run_dir / "args.yaml").is_file():
        args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8")) or {}
    return {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": "training/data/data.yaml (Open Images, 4 classes)",
        "base_model": str(args.get("model", "")),
        "task": str(args.get("task", "detect")),
        "epochs": args.get("epochs"),
        "imgsz": args.get("imgsz"),
        "batch": args.get("batch"),
        "source_run": run_dir.name,
        "best_weights": str(target / "weights" / "best.pt"),
    }


def register(run_dir: Path, studio_runs: Path, name: str, force: bool = False) -> Path:
    """Copy ``run_dir`` (an Ultralytics run) to ``studio_runs/name``; return the new folder."""
    if not SAFE_NAME.fullmatch(name):
        raise ValueError(f"invalid name for the Studio: {name!r}")
    weights = run_dir / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"no weights at {weights}")
    target = studio_runs / name
    if target.exists():
        if not force:
            raise FileExistsError(f"{target} already exists (use --force to replace it)")
        shutil.rmtree(target)
    (target / "weights").mkdir(parents=True)
    shutil.copy2(weights, target / "weights" / "best.pt")
    if (run_dir / "results.csv").is_file():
        shutil.copy2(run_dir / "results.csv", target / "results.csv")
    metadata = studio_metadata(run_dir, target)
    (target / "armsoft_training.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                  encoding="utf-8")
    return target


def main() -> int:
    from armsoft.data import data_root  # the Studio's data root

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="iter1", help="folder under training/runs/detect/")
    ap.add_argument("--name", default=None, help="name in the Studio (default: multiclass_det_<run>)")
    ap.add_argument("--force", action="store_true", help="replace an existing Studio run of that name")
    args = ap.parse_args()

    name = args.name or f"multiclass_det_{args.run}"
    target = register(RUNS_DIR / "detect" / args.run,
                      data_root() / "runs" / "object_segmentation", name, args.force)
    print(f"[register] {name} -> {target}")
    print("[register] start the Studio (python tools/dataset_web.py) and pick it in Live Test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
