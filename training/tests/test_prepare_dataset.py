import json
import sys
import types
from pathlib import Path
from unittest import mock

import yaml

from training.data import prepare_dataset as P
from training.data.yolo_format import CLASSES
from training.paths import REPO_DIR
from training.tests._png import write_png

CONFIG = REPO_DIR / "configs" / "datasets" / "openimages_d405_v1.yaml"
SPLITS = {"train": ["s_train"], "val": ["s_val"], "test": ["s_test"]}


def test_pipeline_config_keeps_fixed_class_order():
    config = P.load_config(CONFIG)
    assert config["classes"] == CLASSES
    assert config["huggingface"]["revision"].startswith("ac4f808")


def test_config_matches_the_reviewed_studio_constants():
    from training.data.studio_format import EXCLUDED_FRAMES, SESSION_SPLITS, STUDIO_TO_CLASS
    d405 = P.load_config(CONFIG)["d405"]
    assert d405["session_splits"] == SESSION_SPLITS
    assert d405["excluded_frames"] == EXCLUDED_FRAMES
    assert d405["class_map"] == STUDIO_TO_CLASS


def test_generated_yaml_resolves_to_the_mixed_dataset():
    config = P.load_config(CONFIG)
    destination = REPO_DIR / "training" / "data" / "openimages6_d405_v1.yaml"
    with mock.patch.object(Path, "write_text") as written, mock.patch.object(Path, "mkdir"):
        P.write_dataset_yaml(config, destination)
    data = yaml.safe_load(written.call_args.args[0])
    assert data["path"] == "../datasets/openimages6_d405_v1"
    assert list(data["names"].values()) == CLASSES


def test_quick_run_never_uses_the_full_dataset_names():
    full = P.load_config(CONFIG)
    quick = P.effective_config(full, per_class=50)
    assert quick["quick_run"]
    for section, key in (("openimages", "output"), ("d405", "output_name"),
                         ("mixed", "output_name"), ("mixed", "yaml")):
        assert quick[section][key] != full[section][key]
        assert "_pc50" in quick[section][key]
    same = P.effective_config(full, per_class=full["openimages"]["per_class"])
    assert not same["quick_run"] and same["mixed"] == full["mixed"]


# --------------------------------------------------------------------------- fake sources

def _fake_sources(root: Path) -> dict:
    base, hf = root / "oi", root / "hf"
    for value, split in enumerate(("train", "val", "test"), 1):
        write_png(base / "images" / split / f"oi_{split}.png", value)
        (base / "labels" / split).mkdir(parents=True)
        (base / "labels" / split / f"oi_{split}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (base / "stats.json").write_text("{}")
    (base / "DOWNLOAD.json").write_text(json.dumps({"per_class": 7, "seed": 0, "split": "train"}))
    value = 100
    for sessions in SPLITS.values():
        for session in sessions:
            folder = hf / "annotations" / "object_segmentation" / session / "annotations"
            folder.mkdir(parents=True)
            for frame in range(2):
                value += 1
                frame_id = f"frame_{frame:06d}"
                write_png(hf / "raw" / "object" / session / "color" / f"{frame_id}.png", value)
                (folder / f"{frame_id}.json").write_text(json.dumps({
                    "review_status": "approved", "frame_id": frame_id,
                    "source_session": f"C:\\data\\raw\\object\\{session}",
                    "source_color": f"color/{frame_id}.png", "width": 640, "height": 480,
                    "objects": [{"class_name": "can", "bbox_xyxy": [10, 10, 100, 200]}]}))
    (hf / "DOWNLOAD.json").write_text(json.dumps({"revision": "abc"}))
    config = P.load_config(CONFIG)
    config["openimages"]["output"] = str(base)
    config["huggingface"]["output"] = str(hf)
    config["mixed"]["yaml"] = str(root / "data.yaml")
    config["d405"]["session_splits"] = SPLITS
    config["d405"]["excluded_frames"] = {"s_train": {"frame_000001": "bad box"}}
    return P.effective_config(config)


def _build(tmp_path, config, force=False):
    with mock.patch.object(P, "DATASETS_DIR", tmp_path / "datasets"):
        return P.build(config, force)


def test_build_publishes_validated_datasets_with_manifest(tmp_path):
    config = _fake_sources(tmp_path)
    d405, mixed = _build(tmp_path, config)
    manifest = json.loads((mixed / "dataset_manifest.json").read_text())
    assert manifest["sources"]["openimages"]["download"]["per_class"] == 7   # what was downloaded
    assert manifest["sources"]["huggingface"]["download"]["revision"] == "abc"
    report = json.loads((mixed / "validation_report.json").read_text())
    assert report["valid"]
    # 1 Open Images + 1 kept D405 train frame x 5 repeats; frame_000001 is excluded
    assert report["splits"]["train"]["images"] == 6
    assert report["splits"]["val"]["images"] == 3                 # 1 Open Images + 2 D405
    assert report["splits"]["test"]["images"] == 1                # D405 test never mixed in
    assert (mixed / "stats.json").is_file()
    train = list((mixed / "images" / "train").glob("s_train*"))
    assert len({path.stat().st_ino for path in train}) == 1      # hard links, one copy on disk
    data = yaml.safe_load((tmp_path / "data.yaml").read_text())
    assert (tmp_path / data["path"]).resolve() == mixed.resolve()
    assert not any((tmp_path / "datasets" / ".staging").iterdir())


def test_existing_dataset_requires_force_and_is_archived(tmp_path):
    config = _fake_sources(tmp_path)
    _build(tmp_path, config)
    try:
        _build(tmp_path, config)
        raise AssertionError("second build without --force must fail")
    except FileExistsError:
        pass
    _build(tmp_path, config, force=True)
    archived = sorted(p.name for p in (tmp_path / "datasets" / ".previous").iterdir())
    assert len(archived) == 2 and archived[0].startswith("d405_v1-")


def test_failed_validation_keeps_reports_and_publishes_nothing(tmp_path):
    config = _fake_sources(tmp_path)
    (Path(config["openimages"]["output"]) / "labels" / "train" / "oi_train.txt").write_text(
        "9 0.5 0.5 0.2 0.2\n")
    try:
        _build(tmp_path, config)
        raise AssertionError("invalid dataset must not be published")
    except RuntimeError:
        pass
    datasets = tmp_path / "datasets"
    assert not (datasets / "openimages6_d405_v1").exists()
    reports = [json.loads(path.read_text())
               for path in (datasets / ".failed").glob("*/*/validation_report.json")]
    assert len(reports) == 2                                       # d405 and mixed
    assert not all(report["valid"] for report in reports)


def test_publish_rolls_back_when_a_move_fails(tmp_path):
    staged = [tmp_path / "stage" / name for name in ("a", "b")]
    finals = [tmp_path / "final" / name for name in ("a", "b")]
    for path in staged + finals:
        path.mkdir(parents=True)
        (path / "marker").write_text(str(path))
    real_move = P.shutil.move
    calls = {"n": 0}

    def flaky_move(src, dst):
        calls["n"] += 1
        if calls["n"] == 4:                       # 2 archives, 1 publish, then fail
            raise OSError("disk full")
        return real_move(src, dst)

    with mock.patch.object(P, "DATASETS_DIR", tmp_path / "datasets"), \
            mock.patch.object(P.shutil, "move", side_effect=flaky_move):
        try:
            P._publish_all(list(zip(staged, finals)), force=True)
            raise AssertionError("publish must fail")
        except OSError:
            pass
    for stage, final in zip(staged, finals):
        assert (final / "marker").read_text() == str(final)       # old versions restored
        assert (stage / "marker").read_text() == str(stage)       # new versions back in staging


# --------------------------------------------------------------------------- downloads

def test_openimages_download_is_skipped_only_for_identical_settings(tmp_path):
    config = _fake_sources(tmp_path)
    with mock.patch.object(P.subprocess, "run") as run:
        try:
            P.download_openimages(config, None, refresh=False)     # marker says 7, config 500
            raise AssertionError("different settings must not be reused silently")
        except RuntimeError:
            pass
        config["openimages"]["per_class"] = 7
        P.download_openimages(config, None, refresh=False)
        run.assert_not_called()


def test_huggingface_download_pins_and_records_the_revision(tmp_path):
    config = P.load_config(CONFIG)
    config["huggingface"]["output"] = str(tmp_path / "hf")
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return kwargs["local_dir"]

    fake = types.SimpleNamespace(snapshot_download=snapshot_download)
    with mock.patch.dict(sys.modules, {"huggingface_hub": fake}):
        P.download_huggingface(config, refresh=False)
        P.download_huggingface(config, refresh=False)             # second call is skipped
    assert len(calls) == 1
    assert calls[0]["revision"] == config["huggingface"]["revision"]
    marker = json.loads((tmp_path / "hf" / "DOWNLOAD.json").read_text())
    assert marker["revision"] == config["huggingface"]["revision"]
