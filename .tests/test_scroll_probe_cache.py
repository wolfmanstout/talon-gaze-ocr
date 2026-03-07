"""Unit tests for probe-scroll cache helpers."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scroll_detection import BoundingBox
from scroll_probe_cache import (
    ScrollProbeCacheEntry,
    can_reuse_cached_viewport,
    is_outside_viewport_mostly_unchanged,
    outside_viewport_match_ratio,
    ratios_match,
    update_ratio_stability,
    viewport_contains_point,
)


def _make_entry() -> ScrollProbeCacheEntry:
    return ScrollProbeCacheEntry(
        viewport_refined_img=BoundingBox(100, 150, 800, 600),
        reference_before_full=np.zeros((720, 1280, 3), dtype=np.uint8),
    )


def test_outside_viewport_match_ratio_only_considers_outside():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = a.copy()
    viewport = BoundingBox(2, 2, 6, 6)

    # Change only inside viewport; outside should still match perfectly.
    b[2:8, 2:8, :] = 255
    assert outside_viewport_match_ratio(a, b, viewport) == 1.0

    # Change part of outside region too.
    b[0, :, :] = 255
    ratio = outside_viewport_match_ratio(a, b, viewport)
    assert 0.0 < ratio < 1.0


def test_is_outside_viewport_mostly_unchanged_threshold():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = a.copy()
    viewport = BoundingBox(2, 2, 6, 6)
    b[0, :, :] = 1
    assert is_outside_viewport_mostly_unchanged(a, b, viewport, threshold=0.6)
    assert not is_outside_viewport_mostly_unchanged(a, b, viewport, threshold=0.95)


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


def test_outside_viewport_check_uses_refined_viewport():
    entry = ScrollProbeCacheEntry(
        viewport_refined_img=BoundingBox(2, 2, 6, 6),
        reference_before_full=np.zeros((720, 1280, 3), dtype=np.uint8),
    )
    current = np.zeros((10, 10, 3), dtype=np.uint8)
    cached = current.copy()
    cached[2:8, 2:8, :] = 255
    assert is_outside_viewport_mostly_unchanged(
        current, cached, entry.viewport_refined_img, threshold=1.0
    )


def test_viewport_contains_point():
    viewport = BoundingBox(100, 150, 800, 600)
    assert viewport_contains_point(viewport, (100, 150))
    assert viewport_contains_point(viewport, (899, 749))
    assert not viewport_contains_point(viewport, (99, 150))
    assert not viewport_contains_point(viewport, (900, 750))


def test_can_reuse_cached_viewport_requires_cursor_inside_and_stable_outside():
    entry = ScrollProbeCacheEntry(
        viewport_refined_img=BoundingBox(2, 2, 6, 6),
        reference_before_full=np.zeros((10, 10, 3), dtype=np.uint8),
    )
    current = entry.reference_before_full.copy()

    assert can_reuse_cached_viewport(entry, current, (3, 3))
    assert not can_reuse_cached_viewport(entry, current, (0, 0))

    changed = current.copy()
    changed[0, :, :] = 255
    assert not can_reuse_cached_viewport(entry, changed, (3, 3))
