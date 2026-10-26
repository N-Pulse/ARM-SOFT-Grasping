"""
training/train.py
=================
Fine-tune YOLO26n (COCO-pretrained) on a 4-class YOLO dataset.

    python training/train.py --name iter1                       # defaults (CPU)
    python training/train.py --epochs 1 --fraction 0.05 --name smoke
    python training/train.py --data training/data/inhouse.yaml --name inhouse1

Weights end up in training/runs/detect/<name>/weights/best.pt (git-ignored).
Defaults are CPU-friendly; pass --device 0 on a CUDA machine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.paths import DEFAULT_DATA, DEFAULT_MODEL, RUNS_DIR, resolve_data_yaml  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="starting weights (COCO yolo26n.pt, or a previous best.pt)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=10, help="early-stopping epochs")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="fraction of the train split to use (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="iter1", help="run name under training/runs/detect/")
    ap.add_argument("--exist-ok", action="store_true", help="overwrite a run with the same name")
    return ap.parse_args(argv)


def build_train_kwargs(args: argparse.Namespace, data_path) -> dict:
    return dict(data=str(data_path), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=args.workers,
                patience=args.patience, fraction=args.fraction, seed=args.seed,
                deterministic=True, project=str(RUNS_DIR / "detect"),
                name=args.name, exist_ok=args.exist_ok, plots=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    from ultralytics import YOLO  # heavy; imported only when actually training
    data = resolve_data_yaml(args.data)
    model = YOLO(args.model)
    model.train(**build_train_kwargs(args, data))
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"best weights: {best}")


if __name__ == "__main__":
    main()
