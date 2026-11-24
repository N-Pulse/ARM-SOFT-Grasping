from pathlib import Path

import pytest

from training.train import build_train_kwargs, parse_args


def test_train_defaults_preserve_ultralytics_learning_rate():
    args = parse_args([])
    kwargs = build_train_kwargs(args, "dataset.yaml")

    assert kwargs["optimizer"] == "auto"
    assert "lr0" not in kwargs


def test_train_accepts_finetuning_optimizer_and_learning_rate():
    args = parse_args(["--optimizer", "AdamW", "--lr0", "0.0002"])
    kwargs = build_train_kwargs(args, "dataset.yaml")

    assert kwargs["optimizer"] == "AdamW"
    assert kwargs["lr0"] == 0.0002


def test_train_loads_local_augmentation_profile(tmp_path: Path):
    profile = tmp_path / "augmentation.yaml"
    profile.write_text("degrees: 8\nfliplr: 0.5\nmosaic: 0.5\n")
    args = parse_args(["--augmentation-profile", str(profile)])
    kwargs = build_train_kwargs(args, "dataset.yaml")
    assert kwargs["degrees"] == 8 and kwargs["fliplr"] == 0.5 and kwargs["mosaic"] == 0.5


def test_train_rejects_conflicting_or_unknown_augmentation(tmp_path: Path):
    profile = tmp_path / "augmentation.yaml"
    profile.write_text("unsafe_option: 1\n")
    with pytest.raises(ValueError, match="unknown augmentation"):
        build_train_kwargs(parse_args(["--augmentation-profile", str(profile)]), "dataset.yaml")
    profile.write_text("degrees: 8\n")
    with pytest.raises(ValueError, match="choose either"):
        build_train_kwargs(parse_args(["--augmentation-profile", str(profile),
                                       "--no-augmentation"]), "dataset.yaml")
