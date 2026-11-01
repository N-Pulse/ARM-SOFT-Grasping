import pytest

from training.eval_d405 import iou, match_frame, report, session_of, to_xyxy

BOX = (0.1, 0.1, 0.5, 0.5)
FAR = (0.7, 0.7, 0.9, 0.9)


def test_iou_of_identical_and_disjoint_boxes():
    assert iou(BOX, BOX) == pytest.approx(1.0)
    assert iou(BOX, FAR) == 0.0


def test_yolo_box_to_corners():
    assert to_xyxy(.5, .5, .2, .4) == pytest.approx((.4, .3, .6, .7))


def test_right_class_on_the_box_is_found():
    counts = match_frame([(1, BOX)], [(1, BOX, .9)])
    assert counts["found"][1] == 1 and not counts["extra"] and not counts["wrong_class"]


def test_other_class_on_the_box_is_a_wrong_class_for_the_true_class():
    counts = match_frame([(1, BOX)], [(2, BOX, .9)])
    assert counts["wrong_class"][1] == 1 and not counts["found"] and not counts["extra"]


def test_prediction_elsewhere_is_extra_and_the_box_stays_missed():
    counts = match_frame([(1, BOX)], [(1, FAR, .9)])
    assert counts["extra"][1] == 1 and not counts["found"] and counts["boxes"][1] == 1


def test_duplicate_prediction_is_extra():
    counts = match_frame([(1, BOX)], [(1, BOX, .9), (1, BOX, .5)])
    assert counts["found"][1] == 1 and counts["extra"][1] == 1


def test_frame_without_truth_counts_every_prediction_as_extra():
    assert match_frame([], [(4, BOX, .9), (0, FAR, .3)])["extra"] == {4: 1, 0: 1}


def test_session_comes_from_the_file_stem():
    assert session_of("test-box-2__frame_000003") == "test-box-2"


def test_report_shows_classes_sessions_negatives_and_stays_apart():
    metrics = {"classes": {"test": {"boxes": {"can": 10}, "found": {"can": 7}, "ap50": {}},
                           "val": {"boxes": {"box": 4}, "found": {"box": 1},
                                   "wrong_class": {"box": 1}, "ap50": {}}},
               "sessions": {"can_7": {"split": "test", "frames": 10, "boxes": {"can": 10},
                                      "found": {"can": 7}, "extra": {"pen_pencil": 2}, "ap50": {}}},
               "negatives": {"mug_1": {"frames": 20, "frames_with_detection": 3, "detections": 4,
                                       "by_class": {"can": 4}}},
               "open_images": {"near": {"map50": .5, "ap50": {"ball": .6}}}}
    text = report({"run_a": metrics})
    assert "| run_a | can | 10 | 7 | 0 | 0.70 | – |" in text
    assert "| run_a | can_7 (test) | can | 10 | 10 | 7 | 0 | 0.70 | – | 2 |" in text
    assert "| run_a | mug_1 | 20 | 3 | 4 | can 4 |" in text
    assert "20-35 cm" in text and "Separate from `benchmark_val.md`" in text
