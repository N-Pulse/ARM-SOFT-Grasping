"""
training/eval_d405.py
=====================
Evaluate fine-tuning runs on D405 images, separately from the Open Images benchmark:

    python training/eval_d405.py --root ../ARM-SOFT-data-douaa \
        --runs b1_yolo26s_d1 b2_yolo26s_d405 b2_yolo26s_mix

writes training/results/d405_finetune/report.md and metrics.json:
    - results per class and per session (test, then val sessions)
    - false detections on sessions holding none of the six classes (hands, mug, egg)
    - Open Images near and far AP50, to see whether classes were forgotten

Nothing goes to evaluations.csv or benchmark_val.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.data.far_scale import parse_yolo  # noqa: E402
from training.data.studio_format import SESSION_SPLITS  # noqa: E402
from training.data.yolo_format import CLASSES  # noqa: E402
from training.eval import resolve_imgsz, read_run_args  # noqa: E402
from training.paths import DATASETS_DIR, RESULTS_DIR, RUNS_DIR, TRAINING_DIR  # noqa: E402

CONF = 0.25                             # Ultralytics default for prediction
IOU_MATCH = 0.5
NEGATIVE_SESSIONS = ["right-hand-3fingers-1", "right-hand-thumbs-up-1", "right-hand-thumbs-up-2",
                     "mug_1", "ellipsoid", "ellipsoid_2", "ellipsoid_3"]
OUT_DIR = RESULTS_DIR / "d405_finetune"
LIMITS = """## Limits

- Very small test: one object per class (tall can, red cube), no pen_pencil, no ball, bottle or
  door_handle. Indicative only, not a verdict on CV-REQ-002.
- Captures are close: about 20-35 cm on the object, measured from depth on 3 frames per session,
  so outside the required 50 cm-1 m range.
- Almost the same white table in every session. Frames of a session are strongly correlated.
- Counts use conf 0.25 and IoU 0.5. AP50 comes from Ultralytics validation.
"""


def to_xyxy(xc, yc, w, h):
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


def iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match_frame(truths, predictions, iou_min: float = IOU_MATCH) -> dict[str, Counter]:
    """
    ``truths``: [(class, box)], ``predictions``: [(class, box, conf)].  A prediction finds a
    truth of its class (highest confidence first, one prediction per truth).  One left
    overlapping another truth is a wrong class; any other is an extra detection.
    Returns Counters per true class: boxes, found, wrong_class; per predicted class: extra.
    """
    counts = {k: Counter() for k in ("boxes", "found", "wrong_class", "extra")}
    counts["boxes"].update(c for c, _ in truths)
    free = list(range(len(truths)))
    for cls, box, _ in sorted(predictions, key=lambda p: -p[2]):
        same = [i for i in free if truths[i][0] == cls and iou(truths[i][1], box) >= iou_min]
        if same:
            best = max(same, key=lambda i: iou(truths[i][1], box))
            free.remove(best)
            counts["found"][cls] += 1
        else:
            other = [i for i in free if truths[i][0] != cls and iou(truths[i][1], box) >= iou_min]
            if other:
                best = max(other, key=lambda i: iou(truths[i][1], box))
                free.remove(best)
                counts["wrong_class"][truths[best][0]] += 1
            else:
                counts["extra"][cls] += 1
    return counts


def add_counts(total: dict[str, Counter], counts: dict[str, Counter]) -> None:
    for key, value in counts.items():
        total.setdefault(key, Counter()).update(value)


def session_of(stem: str) -> str:
    return stem.split("__")[0]


def predictions_of(result) -> list[tuple[str, tuple, float]]:
    boxes = result.boxes
    return [(CLASSES[int(c)], tuple(map(float, xyxy)), float(p))
            for c, xyxy, p in zip(boxes.cls, boxes.xyxyn, boxes.conf)]


def session_list_yaml(images, name: str, work: Path) -> Path:
    """Dataset yaml whose test split is exactly ``images``."""
    work.mkdir(parents=True, exist_ok=True)
    listing = work / f"{name}.txt"
    listing.write_text("\n".join(p.resolve().as_posix() for p in images) + "\n")
    out = work / f"{name}.yaml"
    out.write_text(yaml.safe_dump({"path": DATASETS_DIR.resolve().as_posix(), "train": listing.as_posix(),
                                   "val": listing.as_posix(), "test": listing.as_posix(),
                                   "names": dict(enumerate(CLASSES))}, sort_keys=False))
    return out


def ap50_per_class(model, data, split, imgsz, name, device) -> dict:
    r = model.val(data=str(data), split=split, imgsz=imgsz, device=device, batch=16, plots=False,
                  verbose=False, project=str(RUNS_DIR / "eval_d405"), name=name, exist_ok=True)
    per_class = {CLASSES[int(c)]: float(r.box.class_result(j)[2])
                 for j, c in enumerate(r.box.ap_class_index)}
    return {"map50": float(r.box.map50), "ap50": per_class}


def evaluate_run(run: str, root: Path, device: str) -> dict:
    from ultralytics import YOLO
    weights = RUNS_DIR / "detect" / run / "weights" / "best.pt"
    model = YOLO(str(weights))
    imgsz = resolve_imgsz(None, read_run_args(weights))
    if [model.names[i] for i in range(len(model.names))] != CLASSES:
        raise SystemExit(f"{run}: classes {model.names} differ from {CLASSES}")
    out = {"imgsz": imgsz, "sessions": {}, "classes": {}, "negatives": {}, "open_images": {}}
    work = RUNS_DIR / "eval_d405" / "lists"
    data_dir = DATASETS_DIR / "d405"

    for split in ("test", "val"):
        split_total: dict[str, Counter] = {}
        images = sorted((data_dir / "images" / split).iterdir())
        for session in SESSION_SPLITS[split]:
            frames = [p for p in images if session_of(p.stem) == session]
            total: dict[str, Counter] = {}
            for path in frames:
                label = data_dir / "labels" / split / f"{path.stem}.txt"
                truths = [(CLASSES[c], to_xyxy(*box))
                          for c, *box in parse_yolo(label.read_text().splitlines())]
                result = model.predict(str(path), imgsz=imgsz, conf=CONF, device=device, verbose=False)[0]
                add_counts(total, match_frame(truths, predictions_of(result)))
            add_counts(split_total, total)
            data = session_list_yaml(frames, session, work)
            ap = ap50_per_class(model, data, "test", imgsz, f"{run}-{session}", device)
            out["sessions"][session] = {"split": split, "frames": len(frames), "ap50": ap["ap50"],
                                        **{k: dict(v) for k, v in total.items()}}
        data = session_list_yaml(images, f"d405_{split}", work)
        ap = ap50_per_class(model, data, "test", imgsz, f"{run}-d405-{split}", device)
        out["classes"][split] = {"ap50": ap["ap50"], **{k: dict(v) for k, v in split_total.items()}}

    for session in NEGATIVE_SESSIONS:
        frames = sorted((root / "raw" / "object" / session / "color").glob("*.png"))
        by_class, with_detection = Counter(), 0
        for path in frames:
            found = predictions_of(model.predict(str(path), imgsz=imgsz, conf=CONF, device=device,
                                                 verbose=False)[0])
            with_detection += bool(found)
            by_class.update(c for c, _, _ in found)
        out["negatives"][session] = {"frames": len(frames), "frames_with_detection": with_detection,
                                     "detections": sum(by_class.values()), "by_class": dict(by_class)}

    for name, yaml_name in (("near", "openimages6.yaml"), ("far", "openimages6_far.yaml")):
        from training.paths import resolve_data_yaml
        data = resolve_data_yaml(TRAINING_DIR / "data" / yaml_name, work)
        out["open_images"][name] = ap50_per_class(model, data, "val", imgsz, f"{run}-oi-{name}", device)
    return out


def _ap(metrics, cls) -> str:
    return f"{metrics['ap50'][cls]:.3f}" if cls in metrics.get("ap50", {}) else "–"


def _count_row(metrics, cls) -> str:
    n = metrics.get("boxes", {}).get(cls, 0)
    found = metrics.get("found", {}).get(cls, 0)
    wrong = metrics.get("wrong_class", {}).get(cls, 0)
    return f"{n} | {found} | {wrong} | {found / n:.2f} | {_ap(metrics, cls)}"


def report(results: dict[str, dict]) -> str:
    lines = ["# Fine-tuning on D405 images", "",
             "Separate from `benchmark_val.md`. Boxes come from approved, visually checked frames.", ""]
    head = "| run | {} | boxes | found | wrong class | recall | AP50 |"
    rule = "|---|---|---|---|---|---|---|"
    for split, title in (("test", "Test by class (objects never seen in train or val)"),
                         ("val", "Val by class (used to choose the epoch, so optimistic)")):
        lines += [f"## {title}", "", head.format("class"), rule]
        for run, m in results.items():
            data = m["classes"][split]
            for cls in CLASSES:
                if data.get("boxes", {}).get(cls):
                    lines.append(f"| {run} | {cls} | {_count_row(data, cls)} |")
        lines.append("")
    lines += ["## By session", "", "| run | session (split) | class | frames | boxes | found | wrong class "
              "| recall | AP50 | extra detections |", "|---|---|---|---|---|---|---|---|---|---|"]
    for run, m in results.items():
        for session, data in m["sessions"].items():
            extra = sum(data.get("extra", {}).values())
            for cls in CLASSES:
                if data.get("boxes", {}).get(cls):
                    lines.append(f"| {run} | {session} ({data['split']}) | {cls} | {data['frames']} "
                                 f"| {_count_row(data, cls)} | {extra} |")
    lines += ["", "## False detections on hands, mug and egg", "",
              "None of these objects is one of the six classes, so every detection counts as false. "
              "All frames of each session are used.", "",
              "| run | session | frames | frames with a detection | detections | by class |",
              "|---|---|---|---|---|---|"]
    for run, m in results.items():
        for session, data in m["negatives"].items():
            by_class = ", ".join(f"{c} {n}" for c, n in sorted(data["by_class"].items())) or "–"
            lines.append(f"| {run} | {session} | {data['frames']} | {data['frames_with_detection']} "
                         f"| {data['detections']} | {by_class} |")
    lines += ["", "## Open Images val: were classes forgotten?", "",
              "| run | set | mAP50 | " + " | ".join(CLASSES) + " |",
              "|---|---|---|" + "---|" * len(CLASSES)]
    for run, m in results.items():
        for name, data in m["open_images"].items():
            lines.append(f"| {run} | {name} | {data['map50']:.3f} | "
                         + " | ".join(_ap(data, c) for c in CLASSES) + " |")
    return "\n".join(lines + ["", LIMITS])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--root", type=Path, required=True, help="Dataset Studio data root")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    results = {run: evaluate_run(run, args.root, args.device) for run in args.runs}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT_DIR / "report.md").write_text(report(results), encoding="utf-8")
    print(report(results))
    print(f"saved {OUT_DIR}")


if __name__ == "__main__":
    main()
