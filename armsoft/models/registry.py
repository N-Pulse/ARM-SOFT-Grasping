"""Validated local registry for deployable Ultralytics models."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

MODEL_SCHEMA = "armsoft.runtime-model.v1"
MODEL_ID = re.compile(r"^(?!\.)[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
TASKS = {"detect": "detection", "detection": "detection", "segment": "segmentation",
         "segmentation": "segmentation", "pose": "pose"}


@dataclass(frozen=True)
class RuntimeModel:
    id: str
    display_name: str
    task: str
    weights: Path
    metadata: Path
    base_model: str = ""
    dataset_version: str = ""
    model_version: str = "1.0.0"
    sha256: str = ""
    classes: dict | None = None
    keypoint_count: int | None = None

    def public(self) -> dict:
        result = asdict(self)
        result["weights"] = str(self.weights)
        result["metadata"] = str(self.metadata)
        return result


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _record(folder: Path) -> RuntimeModel | None:
    metadata = folder / "model.json"
    try:
        data = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("schema") != MODEL_SCHEMA or data.get("id") != folder.name:
        return None
    task = TASKS.get(str(data.get("task", "")).lower())
    weights = (folder / str(data.get("weights", "best.pt"))).resolve()
    if not task or folder.resolve() not in weights.parents or not weights.is_file():
        return None
    return RuntimeModel(
        id=folder.name, display_name=str(data.get("display_name") or folder.name),
        task=task, weights=weights, metadata=metadata.resolve(),
        base_model=str(data.get("base_model") or ""),
        dataset_version=str(data.get("dataset_version") or ""),
        model_version=str(data.get("model_version") or "1.0.0"),
        sha256=str(data.get("sha256") or ""), classes=data.get("classes"),
        keypoint_count=data.get("keypoint_count"),
    )


def discover_runtime_models(root: str | Path) -> list[RuntimeModel]:
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return []
    records = [_record(path) for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")]
    return sorted((record for record in records if record), key=lambda item: item.id)


def register_runtime_model(weights: str | Path, root: str | Path, *, model_id: str,
                           display_name: str | None = None, task: str | None = None,
                           dataset_version: str = "", model_version: str = "1.0.0",
                           author: str = "", notes: str = "") -> RuntimeModel:
    if not MODEL_ID.fullmatch(model_id):
        raise ValueError("model id may only contain letters, numbers, dots, dashes, and underscores")
    source = Path(weights).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".pt":
        raise ValueError("weights must be an existing .pt file")
    from ultralytics import YOLO
    model = YOLO(str(source))
    detected = TASKS.get(str(getattr(model, "task", "")).lower())
    requested = TASKS.get(str(task).lower()) if task else detected
    if requested not in {"segmentation", "detection", "pose"}:
        raise ValueError("model task must be segmentation, detection, or pose")
    if detected and requested != detected:
        raise ValueError(f"requested task {requested} does not match weights task {detected}")
    root = Path(root).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    target = (root / model_id).resolve()
    if root not in target.parents or target.exists():
        raise ValueError("a runtime model with this id already exists")
    names = getattr(model, "names", {}) or {}
    classes = {str(key): str(value) for key, value in (names.items() if isinstance(names, dict) else enumerate(names))}
    configuration = getattr(getattr(model, "model", None), "yaml", {}) or {}
    shape = configuration.get("kpt_shape") or []
    keypoints = int(shape[0]) if requested == "pose" and shape else None
    digest = _digest(source)
    payload = {
        "schema": MODEL_SCHEMA, "id": model_id,
        "display_name": display_name or model_id, "task": requested,
        "framework": "ultralytics", "weights": "best.pt",
        "base_model": source.name, "dataset_version": dataset_version,
        "model_version": model_version, "created_utc": datetime.now(timezone.utc).isoformat(),
        "author": author, "notes": notes, "sha256": digest,
        "classes": classes, "keypoint_count": keypoints,
    }
    stage = Path(tempfile.mkdtemp(prefix=f".{model_id}-", dir=root))
    try:
        shutil.copy2(source, stage / "best.pt")
        (stage / "model.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stage.rename(target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    record = _record(target)
    if record is None:
        raise RuntimeError("registered model failed validation")
    return record
