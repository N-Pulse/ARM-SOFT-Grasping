import os
from pathlib import Path

from training.data.validate_yolo_dataset import _image_decoder, validate_dataset
from training.tests._png import write_png


def _write_dataset(root: Path, duplicate_across_splits=False):
    values = {"train": 10, "val": 20, "test": 30}
    if duplicate_across_splits:
        values["val"] = values["train"]
    for split, value in values.items():
        write_png(root / "images" / split / f"{split}.png", value)
        (root / "labels" / split).mkdir(parents=True)
        (root / "labels" / split / f"{split}.txt").write_text(
            "1 0.500000 0.500000 0.250000 0.250000\n")


def test_valid_yolo_dataset(tmp_path):
    _write_dataset(tmp_path)
    report = validate_dataset(tmp_path)
    assert report["valid"], report["errors"]
    assert report["splits"]["train"]["instances"]["can"] == 1


def test_cross_split_duplicate_is_rejected(tmp_path):
    _write_dataset(tmp_path, duplicate_across_splits=True)
    report = validate_dataset(tmp_path)
    assert not report["valid"]
    assert "image contents occur in more than one split" in " ".join(report["errors"])


def test_invalid_coordinate_is_rejected(tmp_path):
    _write_dataset(tmp_path)
    (tmp_path / "labels" / "train" / "train.txt").write_text("1 1.2 0.5 0.2 0.2\n")
    report = validate_dataset(tmp_path)
    assert not report["valid"]
    assert "outside [0, 1]" in " ".join(report["errors"])


def test_invalid_class_and_missing_label_are_rejected(tmp_path):
    _write_dataset(tmp_path)
    (tmp_path / "labels" / "val" / "val.txt").write_text("6 0.5 0.5 0.2 0.2\n")
    (tmp_path / "labels" / "test" / "test.txt").unlink()
    errors = " ".join(validate_dataset(tmp_path)["errors"])
    assert "invalid class id 6" in errors
    assert "test: 1 images have no label file" in errors


def test_corrupt_image_is_rejected_when_a_decoder_is_installed(tmp_path):
    _write_dataset(tmp_path)
    (tmp_path / "images" / "train" / "train.png").write_bytes(b"not an image")
    report = validate_dataset(tmp_path)
    if _image_decoder() is None:
        assert any("decoding skipped" in warning for warning in report["warnings"])
        return
    assert not report["valid"]
    assert report["corrupt_images"] == [str(tmp_path / "images" / "train" / "train.png")]


def test_hard_links_are_decoded_once(tmp_path):
    _write_dataset(tmp_path)
    source = tmp_path / "images" / "train" / "train.png"
    for n in range(3):
        os.link(source, source.parent / f"copy{n}.png")
        (tmp_path / "labels" / "train" / f"copy{n}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    report = validate_dataset(tmp_path)
    assert report["valid"], report["errors"]
    assert report["splits"]["train"]["images"] == 4
    if report["images_decoded"] is not None:
        assert report["images_decoded"] == 3      # train (shared by 4 names), val, test
