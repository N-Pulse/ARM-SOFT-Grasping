import csv

import pytest

from training.eval import EVAL_COLUMNS, append_evaluation_row, evaluation_row, summarise

NAMES = ["ball", "can", "box", "bottle"]


def _row(**over):
    s = summarise(NAMES, 0.61234, 0.4, {0: (.9, .8, .7, .5), 3: (.5, .4, .3, .2)})
    kw = dict(run="iter1", data="training/data/data.yaml", start_weights="yolo26n.pt",
              imgsz=416, epochs=30, split="val", notes="baseline, CPU", date="2026-10-10")
    kw.update(over)
    return evaluation_row(s, **kw)


def test_append_writes_header_once_and_rows_in_order(tmp_path):
    p = tmp_path / "results" / "evaluations.csv"
    append_evaluation_row(p, _row())
    append_evaluation_row(p, _row(run="iter2", split="test", notes="imgsz 640"))
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == EVAL_COLUMNS
    assert [(r["run"], r["split"]) for r in rows] == [("iter1", "val"), ("iter2", "test")]
    assert rows[0]["map50"] == "0.6123" and rows[0]["ap50_ball"] == "0.7000"
    assert rows[0]["ap50_can"] == ""                       # class absent from split
    assert rows[0]["notes"] == "baseline, CPU"             # comma survives quoting
    assert p.read_text(encoding="utf-8").count("date,run,") == 1


def test_append_rejects_unknown_column(tmp_path):
    with pytest.raises(ValueError):
        append_evaluation_row(tmp_path / "evaluations.csv", {**_row(), "oops": 1})
