import json

import pytest

from training.register_in_studio import register


def _run(tmp_path):
    run = tmp_path / "runs" / "detect" / "iter1"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"weights")
    (run / "results.csv").write_text("epoch,metrics/mAP50(B)\n1,0.5\n", encoding="utf-8")
    (run / "args.yaml").write_text("task: detect\nmodel: yolo26n.pt\nepochs: 30\nimgsz: 416\nbatch: 32\n",
                                   encoding="utf-8")
    return run


def test_register_writes_what_the_studio_lists(tmp_path):
    studio = tmp_path / "data" / "runs" / "object_segmentation"
    target = register(_run(tmp_path), studio, "multiclass_det_iter1")
    assert target == studio / "multiclass_det_iter1"
    assert (target / "weights" / "best.pt").read_bytes() == b"weights"
    assert (target / "results.csv").is_file()
    meta = json.loads((target / "armsoft_training.json").read_text(encoding="utf-8"))
    assert meta["task"] == "detect" and meta["base_model"] == "yolo26n.pt"
    assert (meta["epochs"], meta["imgsz"]) == (30, 416)
    assert meta["best_weights"] == str(target / "weights" / "best.pt")


def test_register_refuses_to_overwrite_unless_forced(tmp_path):
    run, studio = _run(tmp_path), tmp_path / "studio"
    register(run, studio, "m")
    with pytest.raises(FileExistsError):
        register(run, studio, "m")
    register(run, studio, "m", force=True)


def test_register_rejects_bad_name_and_missing_weights(tmp_path):
    with pytest.raises(ValueError):
        register(_run(tmp_path), tmp_path / "studio", "../escape")
    with pytest.raises(FileNotFoundError):
        register(tmp_path / "nothing", tmp_path / "studio", "m")
