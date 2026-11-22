"""Download, build and validate the versioned Open Images + D405 YOLO dataset.

Examples:
    python training/data/prepare_dataset.py check
    python training/data/prepare_dataset.py all --per-class 50     # quick run, separate names
    python training/data/prepare_dataset.py all
    python training/data/prepare_dataset.py validate

A run with ``--per-class`` different from the config is a *quick run*: every output
(Open Images source, D405 and mixed datasets, YOLO yaml) gets a ``_pc<N>`` suffix, so it
never collides with the pinned full dataset.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.data.build_d405_datasets import build_datasets  # noqa: E402
from training.data.validate_yolo_dataset import validate_dataset  # noqa: E402
from training.data.yolo_format import CLASSES  # noqa: E402
from training.paths import DATASETS_DIR, REPO_DIR  # noqa: E402

DEFAULT_CONFIG = REPO_DIR / "configs" / "datasets" / "openimages_d405_v1.yaml"
DEFAULT_START_MODEL = "models/catalog/yolo26s-mix-openimages-d405-v1.0.0/best.pt"


# --------------------------------------------------------------------------- config

def load_config(path: Path) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("classes") != CLASSES:
        raise ValueError(f"classes must exactly match the fixed order: {CLASSES}")
    d405 = config.get("d405", {})
    for target in d405.get("class_map", {}).values():
        if target not in CLASSES:
            raise ValueError(f"d405.class_map points to an unknown class: {target}")
    splits = d405.get("session_splits")
    if splits is not None and set(splits) != {"train", "val", "test"}:
        raise ValueError("d405.session_splits must define exactly train, val and test")
    return config


def effective_config(config: dict, per_class: int | None = None,
                     link_mode: str | None = None) -> dict:
    """Apply command-line overrides; a different ``per_class`` renames every output."""
    config = copy.deepcopy(config)
    full = int(config["openimages"]["per_class"])
    config["quick_run"] = False
    if per_class is not None and per_class != full:
        if per_class < 1:
            raise ValueError("--per-class must be at least 1")
        suffix = f"_pc{per_class}"
        config["quick_run"] = True
        config["name"] += suffix
        config["openimages"]["per_class"] = per_class
        config["openimages"]["output"] += suffix
        config["d405"]["output_name"] += suffix
        config["mixed"]["output_name"] += suffix
        data_yaml = Path(config["mixed"]["yaml"])
        config["mixed"]["yaml"] = str(data_yaml.with_name(f"{data_yaml.stem}{suffix}{data_yaml.suffix}"))
    if link_mode is not None:
        config["d405"]["link_mode"] = link_mode
    return config


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_DIR / path


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# --------------------------------------------------------------------------- sources

def find_openimages_python(explicit: str | None = None) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"Open Images Python does not exist: {candidate}")
        return candidate
    candidates = [
        REPO_DIR / "training" / ".venv-download" / "bin" / "python",
        REPO_DIR / "training" / ".venv-download" / "Scripts" / "python.exe",
    ]
    if importlib.util.find_spec("fiftyone"):
        return Path(sys.executable)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "FiftyOne environment not found. Create training/.venv-download and install "
        "fiftyone + pyyaml, or pass --openimages-python."
    )


def openimages_request(config: dict) -> dict:
    settings = config["openimages"]
    return {"per_class": int(settings["per_class"]), "seed": int(settings.get("seed", 0)),
            "split": "train"}


def download_openimages(config: dict, python_executable: str | None, refresh: bool) -> Path:
    output = repo_path(config["openimages"]["output"])
    requested = openimages_request(config)
    marker = output / "DOWNLOAD.json"
    if (output / "stats.json").is_file() and not refresh:
        if _read_json(marker) == requested:
            print(f"[skip] Open Images already prepared: {output}")
            return output
        raise RuntimeError(
            f"Open Images output already exists with different or unknown settings: {output}. "
            "Use --refresh-sources to rebuild it."
        )
    python = find_openimages_python(python_executable)
    command = [str(python), str(REPO_DIR / "training/data/download_openimages.py"),
               "--per-class", str(requested["per_class"]),
               "--seed", str(requested["seed"]), "--out", str(output)]
    print("[run]", " ".join(command))
    subprocess.run(command, cwd=REPO_DIR, check=True)
    marker.write_text(json.dumps(requested, indent=2) + "\n", encoding="utf-8")
    return output


def download_huggingface(config: dict, refresh: bool) -> Path:
    settings = config["huggingface"]
    output = repo_path(settings["output"])
    marker = output / "DOWNLOAD.json"
    recorded = _read_json(marker)
    if recorded and not refresh and recorded.get("revision") == settings["revision"]:
        print(f"[skip] Hugging Face revision already downloaded: {output}")
        return output
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install it in the pipeline environment with "
            "python -m pip install huggingface_hub"
        ) from exc
    output.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(repo_id=settings["repo_id"],
                                 repo_type=settings.get("repo_type", "dataset"),
                                 revision=settings["revision"], local_dir=output)
    marker.write_text(json.dumps({
        "repo_id": settings["repo_id"],
        "repo_type": settings.get("repo_type", "dataset"),
        "revision": settings["revision"],
        "resolved_path": str(Path(resolved).resolve()),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    return output


# --------------------------------------------------------------------------- build

def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_dataset_yaml(config: dict, destination: Path, datasets_dir: Path | None = None) -> None:
    """Write a YOLO data yaml whose ``path`` is relative to the yaml itself."""
    mixed = (datasets_dir or DATASETS_DIR) / config["mixed"]["output_name"]
    try:
        relative = Path(_relpath(mixed, destination.parent))
    except ValueError:                      # different drive on Windows
        relative = mixed
    payload = {
        "path": relative.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(config["classes"])},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        f"# Generated by training/data/prepare_dataset.py for {config['name']}; do not edit.\n"
        + yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _relpath(target: Path, start: Path) -> str:
    import os  # noqa: PLC0415
    return os.path.relpath(target.resolve(), start.resolve())


def _publish_all(pairs: list[tuple[Path, Path]], force: bool) -> None:
    """Move every staged dataset to its final place, or none of them.

    Existing finals are archived under ``.previous`` first (only with ``force``). If any
    move fails, published datasets go back to staging and archived ones are restored.
    """
    existing = [final for _, final in pairs if final.exists()]
    if existing and not force:
        raise FileExistsError("dataset already exists; use --force to archive and replace: "
                              + ", ".join(str(path) for path in existing))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    previous = DATASETS_DIR / ".previous"
    archived: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for _, final in pairs:
            if final.exists():
                previous.mkdir(parents=True, exist_ok=True)
                destination = previous / f"{final.name}-{stamp}"
                shutil.move(str(final), str(destination))
                archived.append((destination, final))
        for staged, final in pairs:
            shutil.move(str(staged), str(final))
            published.append((staged, final))
    except BaseException:
        for staged, final in reversed(published):
            shutil.move(str(final), str(staged))
        for destination, final in reversed(archived):
            shutil.move(str(destination), str(final))
        raise
    for destination, _ in archived:
        print(f"[archive] previous version kept at {destination}")


def _split_counts(report: dict) -> dict:
    return {split: {"images": info["images"], "boxes": info["boxes"],
                    "instances": info["instances"]}
            for split, info in report["splits"].items()}


def build(config: dict, force: bool) -> tuple[Path, Path]:
    base = repo_path(config["openimages"]["output"])
    hf_root = repo_path(config["huggingface"]["output"])
    if not (base / "stats.json").is_file():
        raise FileNotFoundError(f"Open Images source is not ready: {base}")
    if not (hf_root / "annotations").is_dir() or not (hf_root / "raw").is_dir():
        raise FileNotFoundError(f"Hugging Face Dataset Studio root is not ready: {hf_root}")

    d405_settings = config["d405"]
    d405_final = DATASETS_DIR / d405_settings["output_name"]
    mixed_final = DATASETS_DIR / config["mixed"]["output_name"]
    existing = [path for path in (d405_final, mixed_final) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "built dataset already exists; use --force to archive and replace: "
            + ", ".join(str(path) for path in existing)
        )

    stage_root = DATASETS_DIR / ".staging" / uuid.uuid4().hex
    d405_stage = stage_root / d405_settings["output_name"]
    mixed_stage = stage_root / config["mixed"]["output_name"]
    succeeded = False
    try:
        stats = build_datasets(
            hf_root, base, d405_stage, mixed_stage,
            train_repeat=int(d405_settings.get("train_repeat", 5)),
            link_mode=d405_settings.get("link_mode", "hardlink"),
            session_splits=d405_settings.get("session_splits"),
            excluded_frames=d405_settings.get("excluded_frames"),
            class_map=d405_settings.get("class_map"),
        )
        reports = {
            "d405": validate_dataset(d405_stage, config["classes"]),
            "mixed": validate_dataset(mixed_stage, config["classes"]),
        }
        for name, stage in (("d405", d405_stage), ("mixed", mixed_stage)):
            (stage / "validation_report.json").write_text(
                json.dumps(reports[name], indent=2) + "\n", encoding="utf-8")
        for name, report in reports.items():
            for warning in report["warnings"]:
                print(f"[warning] {name}: {warning}")
        if not all(report["valid"] for report in reports.values()):
            for name, report in reports.items():
                for error in report["errors"][:20]:
                    print(f"[error] {name}: {error}")
            raise RuntimeError("dataset validation failed")

        (mixed_stage / "stats.json").write_text(json.dumps({
            "splits": _split_counts(reports["mixed"]),
            "openimages_source": _read_json(base / "stats.json"),
            "d405": stats,
        }, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "pipeline_version": config.get("version", 1),
            "dataset_name": config["name"],
            "quick_run": config.get("quick_run", False),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_revision": _git_revision(),
            "sources": {
                "openimages": {"config": config["openimages"],
                               "download": _read_json(base / "DOWNLOAD.json")},
                "huggingface": {"config": config["huggingface"],
                                "download": _read_json(hf_root / "DOWNLOAD.json")},
            },
            "build": {key: d405_settings.get(key) for key in
                      ("train_repeat", "link_mode", "class_map", "session_splits",
                       "excluded_frames")},
            "outputs": {"d405": d405_final.name, "mixed": mixed_final.name},
            "d405_stats": stats,
            "validation": {key: _split_counts(value) for key, value in reports.items()},
        }
        for stage in (d405_stage, mixed_stage):
            (stage / "dataset_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        _publish_all([(d405_stage, d405_final), (mixed_stage, mixed_final)], force)
        data_yaml = repo_path(config["mixed"]["yaml"])
        write_dataset_yaml(config, data_yaml)
        succeeded = True
    finally:
        if succeeded:
            shutil.rmtree(stage_root, ignore_errors=True)
        elif stage_root.exists():
            failed = DATASETS_DIR / ".failed" / datetime.now().strftime("%Y%m%d-%H%M%S")
            failed.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stage_root), str(failed))
            print(f"[failed] partial build and validation reports kept at {failed}")
            print("[failed] delete that folder when you no longer need it")

    print(f"[ready] D405 dataset: {d405_final}")
    print(f"[ready] mixed dataset: {mixed_final}")
    print(f"[ready] YOLO yaml: {data_yaml}")
    print_train_command(config, data_yaml)
    return d405_final, mixed_final


def print_train_command(config: dict, data_yaml: Path) -> None:
    try:
        data_arg = data_yaml.resolve().relative_to(REPO_DIR.resolve()).as_posix()
    except ValueError:
        data_arg = str(data_yaml)
    lines = ["python training/train.py", f"  --name {config['name']}_run1"]
    if (REPO_DIR / DEFAULT_START_MODEL).is_file():
        lines.append(f"  --model {DEFAULT_START_MODEL}")
    lines.append(f"  --data {data_arg}")
    if config.get("quick_run"):
        lines.append("  --epochs 1 --fraction 0.1      # quick run: smoke test only")
    print("\nNext, train with:\n    " + " \\\n    ".join(lines) + "\n")


# --------------------------------------------------------------------------- check

def check_environment() -> dict:
    status = {name: importlib.util.find_spec(name) is not None
              for name in ("yaml", "huggingface_hub", "ultralytics", "cv2")}
    try:
        status["fiftyone_python"] = str(find_openimages_python())
    except (RuntimeError, FileNotFoundError):
        status["fiftyone_python"] = None
    status["hf_token"] = None
    if status["huggingface_hub"]:
        try:
            from huggingface_hub import get_token  # noqa: PLC0415
            status["hf_token"] = get_token() is not None
        except ImportError:
            pass
    return status


def check(config: dict) -> dict:
    env = check_environment()
    hints = {
        "huggingface_hub": "python -m pip install huggingface_hub",
        "ultralytics": "bash setup_env.sh (needed to train, not to build)",
        "cv2": "python -m pip install opencv-python (enables the image decode check)",
        "fiftyone_python": "python3 -m venv training/.venv-download && "
                           "training/.venv-download/bin/python -m pip install fiftyone pyyaml",
    }
    print("Environment")
    for name in ("yaml", "huggingface_hub", "ultralytics", "cv2", "fiftyone_python"):
        ok = bool(env[name])
        detail = env[name] if isinstance(env[name], str) else ""
        print(f"  {'OK' if ok else 'MISSING':7} {name:16} {detail or ('' if ok else hints.get(name, ''))}")
    token = env["hf_token"]
    print(f"  {'OK' if token else 'INFO':7} {'hf_token':16} "
          + ("logged in" if token else "not logged in; needed only if the HF dataset is private "
                                      "(huggingface-cli login)"))
    paths = {
        "openimages": repo_path(config["openimages"]["output"]),
        "huggingface": repo_path(config["huggingface"]["output"]),
        "d405": DATASETS_DIR / config["d405"]["output_name"],
        "mixed": DATASETS_DIR / config["mixed"]["output_name"],
        "yaml": repo_path(config["mixed"]["yaml"]),
    }
    print(f"Outputs for {config['name']}" + (" (quick run)" if config.get("quick_run") else ""))
    status = {name: path.exists() for name, path in paths.items()}
    for name, path in paths.items():
        print(f"  {'READY' if status[name] else 'MISSING':7} {name:16} {path}")
    return {"environment": env, "outputs": status}


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["check", "download-openimages", "download-hf",
                                           "build", "validate", "all"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--per-class", type=int,
                        help="Open Images images per class; a value different from the config "
                             "is a quick run with _pc<N> output names")
    parser.add_argument("--link-mode", choices=["hardlink", "copy"],
                        help="override the config; use copy before zipping for Colab")
    parser.add_argument("--openimages-python")
    parser.add_argument("--refresh-sources", action="store_true",
                        help="download sources again even if they look ready")
    parser.add_argument("--force", action="store_true",
                        help="archive existing built datasets under .previous and replace them")
    args = parser.parse_args(argv)
    config = effective_config(load_config(args.config.resolve()), args.per_class, args.link_mode)
    if config["quick_run"]:
        print(f"[quick run] outputs use the name {config['name']}")

    if args.action == "check":
        check(config)
    elif args.action == "download-openimages":
        download_openimages(config, args.openimages_python, args.refresh_sources)
    elif args.action == "download-hf":
        download_huggingface(config, args.refresh_sources)
    elif args.action == "build":
        build(config, args.force)
    elif args.action == "validate":
        mixed = DATASETS_DIR / config["mixed"]["output_name"]
        report = validate_dataset(mixed, config["classes"])
        print(json.dumps({key: report[key] for key in
                          ("dataset", "valid", "errors", "warnings", "corrupt_images")}, indent=2))
        print(json.dumps(_split_counts(report), indent=2))
        return 0 if report["valid"] else 1
    elif args.action == "all":
        download_openimages(config, args.openimages_python, args.refresh_sources)
        download_huggingface(config, args.refresh_sources)
        build(config, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
