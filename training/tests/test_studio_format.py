import pytest

from training.data.studio_format import (SESSION_SPLITS, box_line, frame_lines, is_excluded,
                                         raw_session_name, sessions_are_disjoint)


def _frame(*objects):
    return {"width": 640, "height": 480, "objects": list(objects)}


def test_classes_are_matched_by_name_not_by_studio_id():
    frame = _frame({"class_id": 3, "class_name": "can", "bbox_xyxy": [0, 0, 64, 48]},
                   {"class_id": 2, "class_name": "ball-pen", "bbox_xyxy": [0, 0, 64, 48]},
                   {"class_id": 1, "class_name": "box", "bbox_xyxy": [0, 0, 64, 48]})
    assert [line.split()[0] for line in frame_lines(frame)] == ["1", "4", "2"]


def test_box_is_converted_to_relative_centre_and_size():
    assert box_line("can", [64, 48, 192, 144], 640, 480) == "1 0.200000 0.200000 0.200000 0.200000"


def test_box_is_clipped_to_the_image():
    assert box_line("box", [-10, 0, 64, 500], 640, 480) == "2 0.050000 0.500000 0.100000 1.000000"


def test_empty_box_is_an_error():
    with pytest.raises(ValueError):
        box_line("can", [10, 10, 10, 20], 640, 480)


def test_unknown_class_is_an_error_not_a_silent_gap():
    with pytest.raises(KeyError):
        frame_lines(_frame({"class_name": "human-hand", "bbox_xyxy": [0, 0, 10, 10]}))


def test_raw_session_comes_from_the_windows_path():
    assert raw_session_name("C:\\npulse_cv\\ARM-SOFT-data\\raw\\object\\can") == "can"


def test_rejected_frames_are_excluded_per_session():
    assert is_excluded("can_4", "frame_000007")
    assert is_excluded("can_5", "frame_000014")
    assert not is_excluded("can_4", "frame_000010")
    assert not is_excluded("can_5", "frame_000012")
    assert not is_excluded("can_2", "frame_000007")


def test_no_session_is_in_two_splits():
    assert sessions_are_disjoint(SESSION_SPLITS)
    assert not sessions_are_disjoint({"train": ["a"], "test": ["a"]})
