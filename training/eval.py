"""
training/eval.py
================
Evaluate trained weights on a dataset split and save the results:

    python training/eval.py --weights training/runs/detect/iter1/weights/best.pt --split val --notes "baseline"

writes training/results/<tag>/<split>/  (tag defaults to the run name, e.g. "iter1")
    metrics.json                    mAP50, mAP50-95, P/R/AP per class, top confusions
    metrics.md                      the same, as a table (for analysis in log.md)
    confusion_matrix.png            raw counts (Ultralytics plot)
    confusion_matrix_normalized.png
and appends one row to training/results/evaluations.csv, the ONLY source for
numbers reported anywhere (log.md, README, slides).  Iterations are compared
on --split val (all decisions); test is also run each iteration, for information only.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.data.yolo_format import CLASSES  # noqa: E402
from training.paths import (DEFAULT_DATA, REPO_DIR, RESULTS_DIR, RUNS_DIR,  # noqa: E402
                            resolve_data_yaml)

EVALUATIONS_CSV = RESULTS_DIR / "evaluations.csv"
EVAL_COLUMNS = ["date", "run", "data", "start_weights", "imgsz", "epochs",
                       "split", "map50", "map50_95",
                       *[f"ap50_{c}" for c in CLASSES], "notes"]

# Ultralytics ConfusionMatrix.matrix is [predicted, true] (checked for 8.4 in
# the plan, Task 7 Step 1).  The last row/column is "background".
ROWS_ARE_PREDICTED = True


def summarise(names, map50, map50_95, per_class) -> dict:
    return {
        "map50": float(map50),
        "map50_95": float(map50_95),
        "per_class": {
            name: (dict(zip(("precision", "recall", "ap50", "ap50_95"),
                            map(float, per_class[i])))
                   if i in per_class else None)
            for i, name in enumerate(names)
        },
    }


def top_confusions(matrix, names, k: int = 5) -> list[tuple[str, str, int]]:
    """Largest off-diagonal cells as (true, predicted, count)."""
    labels = list(names) + ["background"]
    cells = []
    for p in range(len(labels)):
        for t in range(len(labels)):
            n = int(matrix[p][t] if ROWS_ARE_PREDICTED else matrix[t][p])
            if p != t and n > 0:
                cells.append((labels[t], labels[p], n))
    cells.sort(key=lambda c: -c[2])
    return cells[:k]


def to_markdown(summary: dict, confusions) -> str:
    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    rows = ["| class | precision | recall | AP50 | AP50-95 |",
            "|---|---|---|---|---|"]
    for name, m in summary["per_class"].items():
        vals = ([fmt(m[k]) for k in ("precision", "recall", "ap50", "ap50_95")]
                if m else ["n/a"] * 4)
        rows.append(f"| {name} | " + " | ".join(vals) + " |")
    errors = [f"- {t} → predicted {p}: {n}" for t, p, n in confusions] or ["- none"]
    return "\n".join([
        f"mAP50: **{fmt(summary['map50'])}** · mAP50-95: **{fmt(summary['map50_95'])}**",
        "", *rows, "",
        "Main errors (true → predicted; 'background' = missed / false positive):",
        *errors, ""])


def evaluation_row(summary: dict, *, run, data, start_weights, imgsz, epochs,
                    split, notes: str = "", date: str | None = None) -> dict:
    pc = summary["per_class"]
    row = {"date": date or datetime.date.today().isoformat(), "run": run,
           "data": data, "start_weights": start_weights, "imgsz": imgsz,
           "epochs": epochs, "split": split,
           "map50": f"{summary['map50']:.4f}",
           "map50_95": f"{summary['map50_95']:.4f}"}
    for c in CLASSES:
        row[f"ap50_{c}"] = f"{pc[c]['ap50']:.4f}" if pc.get(c) else ""
    row["notes"] = notes
    return row


def append_evaluation_row(csv_path, row: dict) -> None:
    unknown = set(row) - set(EVAL_COLUMNS)
    if unknown:
        raise ValueError(f"unknown evaluation columns {sorted(unknown)}")
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EVAL_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in EVAL_COLUMNS})


def read_run_args(weights) -> dict:
    """Ultralytics saves the training args next to weights/: <run>/args.yaml."""
    p = Path(weights).resolve().parent.parent / "args.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def _rel(path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_DIR).as_posix()
    except ValueError:
        return p.as_posix()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"],
                    help="val for every decision; test is reported for information")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", help="results sub-folder (default: run name)")
    ap.add_argument("--notes", default="", help="free text for the evaluations.csv row")
    ap.add_argument("--no-record", action="store_true",
                    help="do not append to evaluations.csv (smoke runs only)")
    args = ap.parse_args(argv)
    tag = args.tag or args.weights.resolve().parent.parent.name

    from ultralytics import YOLO
    r = YOLO(str(args.weights)).val(
        data=str(resolve_data_yaml(args.data)), split=args.split,
        imgsz=args.imgsz, device=args.device, batch=16, plots=True,
        project=str(RUNS_DIR / "eval"), name=f"{tag}-{args.split}", exist_ok=True)

    names = [r.names[i] for i in range(len(r.names))]
    per_class = {int(c): r.box.class_result(j) for j, c in enumerate(r.box.ap_class_index)}
    summary = summarise(names, r.box.map50, r.box.map, per_class)
    confusions = top_confusions(r.confusion_matrix.matrix, names)
    summary.update(split=args.split, weights=str(args.weights), imgsz=args.imgsz,
                   top_confusions=[list(c) for c in confusions])

    out = RESULTS_DIR / tag / args.split
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "metrics.md").write_text(f"# {tag} ({args.split} split)\n\n"
                                    + to_markdown(summary, confusions), encoding="utf-8")
    for png in ("confusion_matrix.png", "confusion_matrix_normalized.png"):
        if (Path(r.save_dir) / png).exists():
            shutil.copy2(Path(r.save_dir) / png, out / png)

    if not args.no_record:
        run_args = read_run_args(args.weights)
        append_evaluation_row(EVALUATIONS_CSV, evaluation_row(
            summary, run=tag, data=_rel(args.data),
            start_weights=run_args.get("model", args.weights.name),
            imgsz=args.imgsz, epochs=run_args.get("epochs", ""),
            split=args.split, notes=args.notes))
    print(to_markdown(summary, confusions))
    print(f"results saved to {out}"
          + ("" if args.no_record else f"; row appended to {EVALUATIONS_CSV}"))


if __name__ == "__main__":
    main()
