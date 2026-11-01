"""
porting/compare_results.py
==========================
Frame-by-frame diff of two ``run_replay.py --jsonl`` outputs, e.g. real Open3D
vs open3d_lite, torch vs ONNX, or x86-64 vs arm64.

    python porting/compare_results.py A.jsonl B.jsonl [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def load(path):
    return [json.loads(line) for line in open(path)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    if len(A) != len(B):
        print(f"different frame counts: {len(A)} vs {len(B)}")
        return 1

    same = {k: 0 for k in ("status", "shape", "stable", "hand_pose")}
    both_ok, dpos, djaw, dw, dh, djoint = 0, [], [], [], [], []
    for a, b in zip(A, B):
        for k in same:
            same[k] += a[k] == b[k]
        if a["status"] == b["status"] == "ok":
            both_ok += 1
            dpos.append(np.abs(np.subtract(a["position_m"], b["position_m"])).max())
            djaw.append(abs(a["jaw_opening_m"] - b["jaw_opening_m"]))
            dw.append(abs(a["object_width_m"] - b["object_width_m"]))
            dh.append(abs(a["object_height_m"] - b["object_height_m"]))
            djoint.append(np.abs(np.subtract(a["joint_positions"], b["joint_positions"])).max())

    n = len(A)
    mx = lambda v: float(np.max(v) * 1e3) if v else 0.0   # noqa: E731  metres → mm
    res = dict(frames=n, both_ok=both_ok,
               **{f"same_{k}": v for k, v in same.items()},
               max_dpos_mm=mx(dpos), max_djaw_mm=mx(djaw),
               max_dwidth_mm=mx(dw), max_dheight_mm=mx(dh),
               max_djoint=float(np.max(djoint)) if djoint else 0.0)
    print(f"{args.a}\n  vs {args.b}")
    print(f"  frames {n}: status same {same['status']}/{n}, shape same {same['shape']}/{n}, "
          f"stable same {same['stable']}/{n}")
    print(f"  on {both_ok} frames ok in both: max |Δpos| {res['max_dpos_mm']:.3f} mm, "
          f"|Δjaw| {res['max_djaw_mm']:.3f} mm, |Δwidth| {res['max_dwidth_mm']:.3f} mm, "
          f"|Δheight| {res['max_dheight_mm']:.3f} mm, |Δjoint| {res['max_djoint']:.2e}")
    if args.json:
        json.dump(res, open(args.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
