#!/usr/bin/env python3
"""Train and validate a YOLO segmentation model from an exported dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="exported data.yaml")
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", type=Path, default=Path("runs/object_segmentation"))
    parser.add_argument("--name", default="yolo_seg_v1")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"dataset config not found: {args.data}")
    from ultralytics import YOLO  # imported late so --help stays lightweight

    model = YOLO(args.model)
    result = model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, project=str(args.project),
        name=args.name,
    )
    run_dir = Path(result.save_dir)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.data.resolve()),
        "base_model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "best_weights": str(run_dir / "weights" / "best.pt"),
    }
    (run_dir / "armsoft_training.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[train] complete; best weights: {metadata['best_weights']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
