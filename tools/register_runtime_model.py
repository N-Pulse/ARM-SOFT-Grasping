#!/usr/bin/env python3
"""Validate and register a trained YOLO model for Dataset Studio Live Test."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from armsoft.data import data_root  # noqa: E402
from armsoft.models import register_runtime_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--name", required=True, help="stable model id")
    parser.add_argument("--display-name")
    parser.add_argument("--task", choices=("segmentation", "detection", "pose"))
    parser.add_argument("--dataset", default="", help="dataset version or identifier")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--author", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--data-root", type=Path, default=data_root())
    args = parser.parse_args()
    record = register_runtime_model(
        args.weights, args.data_root / "models" / "runtime", model_id=args.name,
        display_name=args.display_name, task=args.task, dataset_version=args.dataset,
        model_version=args.version, author=args.author, notes=args.notes,
    )
    print(json.dumps(record.public(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
