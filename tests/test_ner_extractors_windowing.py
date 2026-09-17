"""
Tests for _select_window, EntityRefiner's token-window fallback for when
a marked document exceeds max_length (src/ner_extractors.py).

Pure integer arithmetic, but importing src.ner_extractors still pulls in
torch/flair/helpers at module level, so run this inside the project's
Docker image (helpers isn't in requirements.txt). No pytest here - run
directly:

    python tests/test_ner_extractors_windowing.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ner_extractors import _select_window


def test_splits_80_20_when_room_on_both_sides():
    # marker spans tokens [1000, 1001] (length 2), budget 100: after
    # reserving the marker's own 2 tokens, 98 remain, split 78 before /
    # 20 after per the 80/20 ratio.
    start, end = _select_window(
        total_len=2002, marker_start=1000, marker_end=1001, budget=100)
    assert end - start <= 100
    assert start <= 1000 and end > 1001
    before = 1000 - start
    after = end - 1001 - 1
    assert before == 78, before
    assert after == 20, after


def test_redistributes_when_before_side_is_short():
    # Marker only 10 tokens into the document: "before" can supply at
    # most 10 of its 78-token share, the other 68 must go "after".
    start, end = _select_window(
        total_len=2000, marker_start=10, marker_end=11, budget=100)
    assert start == 0
    assert end - start <= 100
    after = end - 11 - 1
    assert after == 88, after


def test_redistributes_when_after_side_is_short():
    # Marker 5 tokens from the end: "after" can supply at most 5 of its
    # share, the rest must go "before".
    start, end = _select_window(
        total_len=2000, marker_start=1990, marker_end=1994, budget=100)
    assert end == 2000
    assert end - start <= 100
    before = 1990 - start
    assert before == 90, before


def test_window_never_exceeds_bounds_or_budget():
    for marker_start in (0, 1, 500, 1998, 1999):
        marker_end = min(marker_start + 1, 1999)
        start, end = _select_window(
            total_len=2000, marker_start=marker_start,
            marker_end=marker_end, budget=50)
        assert end - start <= 50
        assert 0 <= start and end <= 2000
        assert start <= marker_start and end > marker_end


def test_marker_wider_than_budget_is_anchored_at_its_start():
    # Pathological: marker span alone exceeds budget. Anchors at its
    # start instead of crashing - closing [/E] (index 160) falls outside
    # the window, a known limitation (see _select_window's docstring).
    start, end = _select_window(
        total_len=2000, marker_start=100, marker_end=160, budget=50)
    assert start == 100
    assert end == 150
    assert end <= 160  # marker_end not reached - [/E] excluded


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
