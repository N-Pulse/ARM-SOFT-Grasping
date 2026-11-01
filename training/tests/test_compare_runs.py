import json

from training.compare_runs import FAR_DATA, NEAR_DATA, load_runs, table


def _metrics(eval_data, map50=.5):
    return {"eval_data": eval_data, "map50": map50, "map50_95": .3, "start_weights": "yolo11n.pt",
            "train_data": "openimages6.yaml", "imgsz": 416, "seed": 0, "n_params": 2_600_000,
            "weights_mb": 5.4, "inference_ms": 40.0,
            "per_class": {"ball": {"recall": .7}, "can": None}}


def _write(results, tag, metrics, split="val"):
    d = results / tag / split
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_near_and_far_are_joined_on_the_run_name(tmp_path):
    _write(tmp_path, "a", _metrics(NEAR_DATA))
    _write(tmp_path, "a__far", _metrics(FAR_DATA, .2))
    runs = load_runs(tmp_path, "val")
    assert set(runs) == {"a"}
    assert runs["a"]["far"]["map50"] == .2


def test_runs_on_other_evaluation_data_are_skipped(tmp_path):
    _write(tmp_path, "iter1", _metrics("training/data/data.yaml"))
    assert load_runs(tmp_path, "val") == {}


def test_d405_fine_tuning_runs_stay_out_of_the_benchmark(tmp_path):
    for train_data in ("d405.yaml", "openimages6_d405.yaml"):
        _write(tmp_path, train_data, {**_metrics(NEAR_DATA), "train_data": train_data})
    assert load_runs(tmp_path, "val") == {}


def test_far_tag_evaluated_on_near_data_is_skipped(tmp_path):
    _write(tmp_path, "a__far", _metrics(NEAR_DATA))
    assert load_runs(tmp_path, "val") == {}


def test_table_ranks_by_far_map_and_shows_missing_values(tmp_path):
    _write(tmp_path, "low", _metrics(FAR_DATA, .1))
    runs = {"low": {"far": _metrics(FAR_DATA, .1)},
            "high": {"far": _metrics(FAR_DATA, .9)},
            "not_far_yet": {"near": _metrics(NEAR_DATA)}}
    lines = table(runs).splitlines()
    assert [l.split("|")[1].strip() for l in lines[2:5]] == ["high", "low", "not_far_yet"]
    assert "0.70 / –" in table({"x": {"near": _metrics(NEAR_DATA)}})
    assert "– / –" in table({"x": {"near": _metrics(NEAR_DATA)}})
