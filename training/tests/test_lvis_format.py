import pytest

from training.data.lvis_format import (DEFAULT_RULES, LVIS_TO_CLASS, box_iou, category_map,
                                       check_image, coco_box_to_yolo, image_stem,
                                       select_per_class, visible_classes, yolo_lines)
from training.data.yolo_format import CLASSES

CATEGORIES = [{"id": i, "name": n} for i, n in enumerate(sorted(LVIS_TO_CLASS) + ["handle"], 1)]
RULES = dict(DEFAULT_RULES)


def _ids_of():
    cat_map = category_map(CATEGORIES)
    out = {}
    for cat_id, cls in cat_map.items():
        out.setdefault(cls, set()).add(cat_id)
    return cat_map, out


def test_mapping_covers_all_six_classes_but_not_generic_handles():
    cat_map, _ = _ids_of()
    assert set(cat_map.values()) == set(CLASSES)
    assert LVIS_TO_CLASS["doorknob"] == "door_handle"
    assert "handle" not in LVIS_TO_CLASS          # any cup or bag handle, too generic


def test_unknown_lvis_name_is_an_error():
    with pytest.raises(ValueError):
        category_map([{"id": 1, "name": "bottle"}])


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0
    assert box_iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(50 / 150)


def test_rules_reject_crowds_overlaps_and_incomplete_labels():
    _, ids_of = _ids_of()
    image = {"width": 100, "height": 100, "not_exhaustive_category_ids": []}
    one = [("can", [10, 10, 20, 20])]
    assert check_image(image, one, ids_of, RULES) is None
    assert check_image(image, [], ids_of, RULES) == "no_object"
    crowd = [("can", [x, 0, 5, 5]) for x in range(0, 40, 10)]          # 4 separate boxes
    assert check_image(image, crowd, ids_of, RULES) == "too_many_objects"
    stacked = [("can", [10, 10, 20, 20]), ("bottle", [12, 12, 20, 20])]
    assert check_image(image, stacked, ids_of, RULES) == "overlap"
    apart = [("can", [0, 0, 20, 20]), ("bottle", [50, 50, 20, 20])]
    assert check_image(image, apart, ids_of, RULES) is None
    incomplete = dict(image, not_exhaustive_category_ids=[next(iter(ids_of["can"]))])
    assert check_image(incomplete, one, ids_of, RULES) == "not_exhaustive"


def test_only_visible_objects_make_an_image_a_candidate():
    image = {"width": 100, "height": 100}
    boxes = [("can", [0, 0, 2, 2]), ("bottle", [50, 50, 10, 10])]       # 0.04 % and 1 %
    assert visible_classes(image, boxes, 0.005) == {"bottle"}


def test_selection_is_deterministic_capped_and_never_repeats_an_image():
    candidates = {"can": list(range(10)), "bottle": list(range(5, 20))}
    picked = select_per_class(candidates, per_class=4, seed=0)
    assert picked == select_per_class(dict(reversed(list(candidates.items()))), 4, seed=0)
    assert sum(cls == "can" for cls in picked.values()) == 4
    assert sum(cls == "bottle" for cls in picked.values()) == 4
    assert len(picked) == 8
    assert picked != select_per_class(candidates, per_class=4, seed=1)


def test_coco_box_becomes_a_clipped_relative_yolo_box():
    assert coco_box_to_yolo([10, 20, 30, 40], 100, 200) == pytest.approx((0.25, 0.2, 0.3, 0.2))
    assert coco_box_to_yolo([90, 0, 30, 10], 100, 100) == pytest.approx((0.95, 0.05, 0.1, 0.1))
    assert coco_box_to_yolo([120, 0, 10, 10], 100, 100) is None
    line = yolo_lines([("bottle", [10, 20, 30, 40])], 100, 200)[0]
    assert line.split()[0] == str(CLASSES.index("bottle"))


def test_stems_never_look_like_open_images_ids():
    assert image_stem(42) == "lvis_000000000042"
