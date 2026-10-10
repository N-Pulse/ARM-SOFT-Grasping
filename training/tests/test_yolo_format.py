import pytest

from training.data.yolo_format import (CLASSES, OI_TO_CLASS, filter_detections,
                                       oi_box_to_yolo, oi_labels_for, split_ids,
                                       yolo_lines)


def test_classes_order_is_fixed():
    assert CLASSES == ["ball", "can", "box", "bottle"]


def test_every_class_has_an_open_images_label():
    for cls in CLASSES:
        assert oi_labels_for(cls), cls
    assert set(OI_TO_CLASS.values()) == set(CLASSES)


def test_box_conversion_centre_and_size():
    assert oi_box_to_yolo([0.1, 0.2, 0.4, 0.2]) == pytest.approx((0.3, 0.3, 0.4, 0.2))


def test_box_is_clipped_to_image():
    xc, yc, w, h = oi_box_to_yolo([-0.1, 0.9, 0.3, 0.3])
    assert (xc, yc, w, h) == pytest.approx((0.1, 0.95, 0.2, 0.1))


def test_degenerate_box_is_dropped():
    assert oi_box_to_yolo([0.5, 0.5, 0.0, 0.1]) is None
    assert oi_box_to_yolo([1.2, 0.5, 0.1, 0.1]) is None


def test_yolo_lines_maps_classes_and_drops_others():
    dets = [("Bottle", [0.0, 0.0, 0.5, 0.5]),
            ("Person", [0.1, 0.1, 0.2, 0.2]),
            ("Tin can", [0.5, 0.5, 0.5, 0.5])]
    assert yolo_lines(dets) == ["3 0.250000 0.250000 0.500000 0.500000",
                                "1 0.750000 0.750000 0.500000 0.500000"]


def test_split_is_deterministic_disjoint_and_complete():
    ids = [f"img{i:03d}" for i in range(100)]
    a = split_ids(ids, seed=0)
    b = split_ids(reversed(ids), seed=0)
    assert a == b
    assert [len(a[k]) for k in ("train", "val", "test")] == [70, 15, 15]
    assert sorted(a["train"] + a["val"] + a["test"]) == ids


def test_split_removes_duplicates():
    s = split_ids(["a", "a", "b", "c"])
    assert sorted(s["train"] + s["val"] + s["test"]) == ["a", "b", "c"]


def test_data_yaml_names_match_classes():
    import yaml
    from training.paths import DEFAULT_DATA
    names = yaml.safe_load(DEFAULT_DATA.read_text())["names"]
    assert [names[i] for i in range(len(names))] == CLASSES


def test_filter_skips_group_of_and_depiction_boxes():
    dets = [("Bottle", [0, 0, .5, .5], {"IsGroupOf": False, "IsDepiction": False}),
            ("Box", [0, 0, 1, 1], {"IsGroupOf": True}),
            ("Ball", [.1, .1, .2, .2], {"IsDepiction": True}),
            ("Tin can", [.5, .5, .2, .2], {"IsGroupOf": True, "IsDepiction": True}),
            ("Ball", [.6, .1, .2, .2], {})]
    kept, skipped = filter_detections(dets)
    assert kept == [("Bottle", [0, 0, .5, .5]), ("Ball", [.6, .1, .2, .2])]
    assert skipped == {"group_of": 2, "depiction": 1}


def test_filter_ignores_other_classes_and_keeps_unflagged():
    dets = [("Person", [0, 0, 1, 1], {"IsGroupOf": True}),   # not ours: not counted
            ("Bottle", [0, 0, .5, .5], {})]
    kept, skipped = filter_detections(dets)
    assert kept == [("Bottle", [0, 0, .5, .5])]
    assert skipped == {"group_of": 0, "depiction": 0}


def test_filtered_detections_feed_yolo_lines():
    kept, _ = filter_detections([("Bottle", [0, 0, .5, .5], {"IsDepiction": True}),
                                 ("Bottle", [.5, .5, .5, .5], {})])
    assert yolo_lines(kept) == ["3 0.750000 0.750000 0.500000 0.500000"]
