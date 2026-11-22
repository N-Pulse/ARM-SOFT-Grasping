from collections import Counter

import pytest

from training.data.far_scale import (CANVAS, fit_in_cell, parse_yolo, plan_canvases,
                                     shrink_boxes)

IDS = [f"img{i:03d}" for i in range(100)]


def test_plan_is_deterministic_whatever_the_input_order():
    assert plan_canvases(IDS, 1, seed=0) == plan_canvases(list(reversed(IDS)), 1, seed=0)


def test_each_image_is_used_exactly_once_per_pass():
    for passes in (1, 3):
        used = Counter(i for _, ids in plan_canvases(IDS, passes, seed=0) for i in ids)
        assert set(used) == set(IDS) and set(used.values()) == {passes}


def test_grid_never_holds_more_images_than_cells():
    for (cols, rows), ids in plan_canvases(IDS, 3, seed=0):
        assert len(ids) <= cols * rows


def test_images_stay_inside_their_own_cell():
    for index in range(12):
        new_w, new_h, dx, dy = fit_in_cell((1024, 768), (4, 3), index)
        col, row = index % 4, index // 4
        assert col * 160 <= dx and dx + new_w <= (col + 1) * 160
        assert row * 160 <= dy and dy + new_h <= (row + 1) * 160


def test_fitting_keeps_the_aspect_ratio():
    new_w, new_h, _, _ = fit_in_cell((1000, 500), (2, 2), 0)
    assert (new_w, new_h) == (320, 160)
    assert new_w + 0 <= CANVAS[0] and new_h <= CANVAS[1]


def test_boxes_follow_the_shrunk_image():
    lines = shrink_boxes([(4, .5, .5, .5, .5)], (320, 240, 100, 50))
    c, xc, yc, w, h = parse_yolo(lines)[0]
    assert c == 4
    assert (xc * 640, yc * 480) == pytest.approx((260, 170), abs=1e-3)
    assert (w * 640, h * 480) == pytest.approx((160, 120), abs=1e-3)


def test_unlearnably_small_box_is_dropped():
    assert shrink_boxes([(0, .5, .5, .001, .5)], (320, 240, 0, 0)) == []
