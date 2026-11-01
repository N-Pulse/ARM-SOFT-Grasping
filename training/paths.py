"""
training/paths.py
=================
Folder layout of the training workspace, and the data.yaml resolver.

Ultralytics resolves a relative ``path:`` in a dataset yaml against its own
global datasets directory, not against the yaml's folder.
:func:`resolve_data_yaml` writes a copy with an absolute ``path:`` so our
yamls can stay relative to themselves and work from any clone.
"""

from __future__ import annotations

from pathlib import Path

import yaml

TRAINING_DIR = Path(__file__).resolve().parent
REPO_DIR     = TRAINING_DIR.parent
DATASETS_DIR = TRAINING_DIR / "datasets"   # git-ignored
RUNS_DIR     = TRAINING_DIR / "runs"       # git-ignored
WEIGHTS_DIR  = TRAINING_DIR / "weights"    # git-ignored
RESULTS_DIR  = TRAINING_DIR / "results"    # committed (small files only)

DEFAULT_DATA  = TRAINING_DIR / "data" / "data.yaml"
DEFAULT_MODEL = "yolo26n.pt"               # COCO-pretrained, auto-downloaded


def resolve_data_yaml(data_yaml: str | Path, out_dir: str | Path = RUNS_DIR) -> Path:
    """Copy ``data_yaml`` to ``out_dir`` with ``path:`` made absolute."""
    data_yaml = Path(data_yaml).resolve()
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    cfg["path"] = root.as_posix()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{data_yaml.stem}.resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out
