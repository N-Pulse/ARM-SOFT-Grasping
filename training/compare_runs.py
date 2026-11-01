"""
training/compare_runs.py
========================
Build the benchmark table from the metrics.json files written by eval.py:

    python training/compare_runs.py                 # val split (all decisions)
    python training/compare_runs.py --split test    # information only

Each run is evaluated twice with eval.py: as is (near, Open Images sizes) and with
``--data training/data/openimages6_far.yaml --tag <run>__far`` (far, apparent size at
50 cm-1 m).  Writes training/results/benchmark_<split>.md; iter* runs on the older
4-class data are skipped because their val and test images differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.data.yolo_format import CLASSES  # noqa: E402
from training.paths import RESULTS_DIR  # noqa: E402

FAR_SUFFIX = "__far"
NEAR_DATA = "training/data/openimages6.yaml"
FAR_DATA = "training/data/openimages6_far.yaml"
# D405 fine-tuning runs are reported apart, by eval_d405.py.
BENCHMARK_TRAIN_DATA = {"openimages6.yaml", "openimages6_large.yaml", "openimages6_far.yaml"}


def load_runs(results_dir: Path, split: str) -> dict[str, dict]:
    """{run: {"near": metrics, "far": metrics}} for runs on the frozen evaluation sets."""
    runs: dict[str, dict] = {}
    for path in sorted(results_dir.glob(f"*/{split}/metrics.json")):
        tag = path.parent.parent.name
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("train_data") not in BENCHMARK_TRAIN_DATA:
            continue
        if tag.endswith(FAR_SUFFIX) and metrics.get("eval_data") == FAR_DATA:
            runs.setdefault(tag[:-len(FAR_SUFFIX)], {})["far"] = metrics
        elif not tag.endswith(FAR_SUFFIX) and metrics.get("eval_data") == NEAR_DATA:
            runs.setdefault(tag, {})["near"] = metrics
    return runs


def _num(metrics, key, digits=3) -> str:
    return f"{metrics[key]:.{digits}f}" if metrics and metrics.get(key) is not None else "–"


def _recall(metrics, cls) -> str:
    per_class = (metrics or {}).get("per_class", {}).get(cls)
    return f"{per_class['recall']:.2f}" if per_class else "–"


def table(runs: dict[str, dict]) -> str:
    """Markdown tables, best far mAP50 first (runs not yet evaluated far come last)."""
    order = sorted(runs, key=lambda r: -(runs[r].get("far") or {}).get("map50", -1))
    summary = ["| run | model | train data | imgsz | seed | params (M) | weights (MB) "
               "| CPU ms/img | near mAP50 | near mAP50-95 | far mAP50 | far mAP50-95 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    recall = ["| run | " + " | ".join(f"{c} near / far" for c in CLASSES) + " |",
              "|---|" + "---|" * len(CLASSES)]
    for run in order:
        near, far = runs[run].get("near"), runs[run].get("far")
        info = near or far
        params = f"{info['n_params'] / 1e6:.2f}" if info.get("n_params") else "–"
        summary.append(
            f"| {run} | {info.get('start_weights', '–')} | {info.get('train_data', '–')} "
            f"| {info.get('imgsz', '–')} | {info.get('seed', '–')} | {params} "
            f"| {_num(info, 'weights_mb', 1)} | {_num(info, 'inference_ms', 1)} "
            f"| {_num(near, 'map50')} | {_num(near, 'map50_95')} "
            f"| {_num(far, 'map50')} | {_num(far, 'map50_95')} |")
        recall.append(f"| {run} | " + " | ".join(
            f"{_recall(near, c)} / {_recall(far, c)}" for c in CLASSES) + " |")
    return "\n".join(summary + ["", "Recall per class:", ""] + recall) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args(argv)
    runs = load_runs(RESULTS_DIR, args.split)
    if not runs:
        sys.exit("no benchmark run evaluated yet: run training/eval.py first")
    out = RESULTS_DIR / f"benchmark_{args.split}.md"
    note = ("Web photos only (Open Images): this ranks the options; the D405 verdict comes later. "
            "CPU time compares models with each other on the dev PC, not the UNO Q.\n\n")
    out.write_text(f"# Model and dataset benchmark ({args.split} split)\n\n{note}{table(runs)}",
                   encoding="utf-8")
    print(table(runs))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
