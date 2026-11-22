"""
training/data/far_scale.py
==========================
Pure helpers to assemble labelled images into camera-sized grids, so objects get
the apparent size they have on the D405 at 50 cm-1 m (training/decisions.md).
A grid has no artificial background and every visible object stays labelled.
No image I/O: everything here is unit-tested.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

CANVAS = (640, 480)                     # D405 colour stream used by the pipeline
GRIDS = ((2, 2), (3, 2), (3, 3), (4, 3), (5, 4))   # (columns, rows)
MIN_BOX_PX = 2                          # smaller boxes are not learnable: label dropped


def parse_yolo(lines: Iterable[str]) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in lines:
        if line.strip():
            c, *box = line.split()
            rows.append((int(c), *map(float, box)))
    return rows


def plan_canvases(ids: Sequence[str], passes: int, seed: int) -> list[tuple[tuple[int, int], list[str]]]:
    """
    Group image ids into grids; each id is used exactly ``passes`` times.  Same
    input and seed give the same plan, whatever the input order.  Unfilled cells
    of the last grid of a pass stay empty.
    """
    rng = random.Random(seed)
    ordered = sorted(set(ids))
    plan = []
    for _ in range(passes):
        rest = ordered[:]
        rng.shuffle(rest)
        while rest:
            grid = rng.choice(GRIDS)
            n = grid[0] * grid[1]
            plan.append((grid, rest[:n]))
            rest = rest[n:]
    return plan


def fit_in_cell(src_wh, grid, index: int, canvas=CANVAS) -> tuple[int, int, int, int]:
    """Return (new_w, new_h, dx, dy): the image centred in cell ``index``, aspect kept."""
    cols, rows = grid
    cell_w, cell_h = canvas[0] // cols, canvas[1] // rows
    scale = min(cell_w / src_wh[0], cell_h / src_wh[1])
    new_w, new_h = max(1, round(src_wh[0] * scale)), max(1, round(src_wh[1] * scale))
    return (new_w, new_h,
            (index % cols) * cell_w + (cell_w - new_w) // 2,
            (index // cols) * cell_h + (cell_h - new_h) // 2)


def shrink_boxes(boxes, placement, canvas=CANVAS, min_px=MIN_BOX_PX) -> list[str]:
    """Express YOLO boxes in canvas coordinates; drop those under ``min_px``."""
    new_w, new_h, dx, dy = placement
    lines = []
    for c, xc, yc, bw, bh in boxes:
        w_px, h_px = bw * new_w, bh * new_h
        if w_px < min_px or h_px < min_px:
            continue
        lines.append(f"{c} {(xc * new_w + dx) / canvas[0]:.6f} {(yc * new_h + dy) / canvas[1]:.6f} "
                     f"{w_px / canvas[0]:.6f} {h_px / canvas[1]:.6f}")
    return lines
