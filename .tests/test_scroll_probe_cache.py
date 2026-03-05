"""Unit tests for probe-scroll cache helpers."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scroll_detection import BoundingBox
from scroll_probe_cache import (
    ScrollProbeCacheEntry,
    full_frame_match_ratio,
    is_full_frame_mostly_unchanged,
    is_viewport_close,
    ratios_match,
    update_ratio_stability,
)


def _make_entry() -> ScrollProbeCacheEntry:
    return ScrollProbeCacheEntry(
        viewport_refined_img=BoundingBox(100, 150, 800, 600),
        viewport_unrefined_img=BoundingBox(90, 140, 820, 620),
        reference_before_full=np.zeros((720, 1280, 3), dtype=np.uint8),
    )


def test_full_frame_match_ratio_identical():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert full_frame_match_ratio(img, img.copy()) == 1.0


def test_full_frame_match_ratio_partial_change():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = a.copy()
    b[0:2, :, :] = 255
    assert full_frame_match_ratio(a, b) == 0.8


def test_is_full_frame_mostly_unchanged_threshold():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = a.copy()
    b[0, :, :] = 1  # 90% match
    assert is_full_frame_mostly_unchanged(a, b, threshold=0.9)
    assert not is_full_frame_mostly_unchanged(a, b, threshold=0.95)


def test_is_viewport_close_within_threshold():
    cached = BoundingBox(100, 200, 900, 500)
    estimated = BoundingBox(110, 210, 910, 505)
    assert is_viewport_close(cached, estimated, frame_w=2000, frame_h=1200)


def test_is_viewport_close_rejects_large_shift():
    cached = BoundingBox(100, 200, 900, 500)
    estimated = BoundingBox(400, 200, 900, 500)
    assert not is_viewport_close(cached, estimated, frame_w=2000, frame_h=1200)


def test_ratios_match_tolerates_float_noise():
    assert ratios_match(2.0 / 3.0, 0.6666666666666667)


def test_update_ratio_stability_requires_two_consecutive_matches():
    entry = _make_entry()
    update_ratio_stability(entry, 1.23, probe_amount=50)
    assert entry.stable_scroll_ratio is None
    assert entry.pending_probe_ratio == 1.23

    update_ratio_stability(entry, 1.23 + 1e-12, probe_amount=50)
    assert entry.stable_scroll_ratio is not None
    assert ratios_match(entry.stable_scroll_ratio, 1.23)


def test_update_ratio_stability_resets_pending_on_mismatch():
    entry = _make_entry()
    update_ratio_stability(entry, 1.0, probe_amount=50)
    update_ratio_stability(entry, 1.1, probe_amount=50)
    assert entry.stable_scroll_ratio is None
    assert entry.pending_probe_ratio == 1.1


def test_stable_ratio_remains_valid_after_mismatch():
    entry = _make_entry()
    update_ratio_stability(entry, 1.0, probe_amount=50)
    update_ratio_stability(entry, 1.0, probe_amount=50)
    assert entry.stable_scroll_ratio == 1.0

    update_ratio_stability(entry, 1.2, probe_amount=50)
    assert entry.stable_scroll_ratio == 1.0


def test_comparison_targets_unrefined_viewport_shape():
    entry = ScrollProbeCacheEntry(
        viewport_refined_img=BoundingBox(300, 300, 500, 300),
        viewport_unrefined_img=BoundingBox(90, 140, 820, 620),
        reference_before_full=np.zeros((720, 1280, 3), dtype=np.uint8),
    )
    estimated = BoundingBox(90, 140, 820, 620)
    assert is_viewport_close(
        entry.viewport_unrefined_img, estimated, frame_w=1280, frame_h=720
    )
    assert not is_viewport_close(
        entry.viewport_refined_img, estimated, frame_w=1280, frame_h=720
    )
