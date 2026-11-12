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
