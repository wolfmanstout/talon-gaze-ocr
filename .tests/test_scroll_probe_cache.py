"""Unit tests for probe-scroll cache helpers."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scroll_detection import BoundingBox
from scroll_probe_cache import (
    AppScrollCache,
    ProbeSkipDecision,
    ScrollCacheEntry,
    outside_viewport_match_ratio,
    ratios_align,
    update_ratio_stability,
)


def _make_entry() -> ScrollCacheEntry:
    return ScrollCacheEntry(
        viewport=BoundingBox(100, 150, 800, 600),
        reference_before_array=np.zeros((720, 1280, 3), dtype=np.uint8),
    )


def test_cache_key_for_app():
    cache = AppScrollCache()
    assert cache.cache_key_for_app("Google Chrome") == "Google Chrome"
    assert cache.cache_key_for_app("Notification Center") is None
    assert cache.cache_key_for_app(None) is None


def test_invalidate_viewport_keeps_stable_scroll_ratio():
    cache = AppScrollCache(
        entries={
            "Google Chrome": ScrollCacheEntry(
                viewport=BoundingBox(2, 2, 6, 6),
                reference_before_array=np.zeros((10, 10, 3), dtype=np.uint8),
                stable_scroll_ratio=1.0,
            )
        }
    )

    cache.invalidate_viewport("Google Chrome")

    entry = cache.entries["Google Chrome"]
    assert entry.viewport is None
    assert entry.reference_before_array is None
    assert entry.stable_scroll_ratio == 1.0


def test_outside_viewport_match_ratio_only_considers_outside():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = a.copy()
    viewport = BoundingBox(2, 2, 6, 6)

    # Change only inside viewport; outside should still match perfectly.
    b[2:8, 2:8, :] = 255
    assert outside_viewport_match_ratio(a, b, viewport) == 1.0

    # Change part of outside region too.
    b[0, :, :] = 3
    ratio = outside_viewport_match_ratio(a, b, viewport)
    assert 0.0 < ratio < 1.0


def test_outside_viewport_match_ratio_threshold():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = a.copy()
    viewport = BoundingBox(2, 2, 6, 6)
    b[0, 0:4, :] = 2
    ratio = outside_viewport_match_ratio(a, b, viewport)
    assert ratio >= 0.90
    assert ratio >= 0.6
    assert ratio < 0.95


def test_outside_viewport_match_ratio_tolerates_tiny_rgb_drift():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = a.copy()
    viewport = BoundingBox(2, 2, 6, 6)
    b[0, :, :] = 1

    assert outside_viewport_match_ratio(a, b, viewport) == 1.0


def test_update_ratio_stability_requires_two_consecutive_matches():
    entry = _make_entry()
    entry = update_ratio_stability(entry, 1.23)
    assert entry.stable_scroll_ratio is None
    assert entry.pending_probe_ratio == 1.23

    entry = update_ratio_stability(entry, 1.23 + 1e-12)
    assert entry.stable_scroll_ratio is not None
    assert entry.stable_scroll_ratio == 1.23 + 1e-12


def test_update_ratio_stability_resets_pending_on_mismatch():
    entry = _make_entry()
    entry = update_ratio_stability(entry, 1.0)
    entry = update_ratio_stability(entry, 1.1)
    assert entry.stable_scroll_ratio is None
    assert entry.pending_probe_ratio == 1.1


def test_stable_ratio_remains_valid_after_mismatch():
    entry = _make_entry()
    entry = update_ratio_stability(entry, 1.0)
    entry = update_ratio_stability(entry, 1.0)
    assert entry.stable_scroll_ratio == 1.0

    entry = update_ratio_stability(entry, 1.2)
    assert entry.stable_scroll_ratio == 1.0


def test_ratios_align_tolerates_small_noise():
    assert ratios_align(4.0, 4.01)


def test_ratios_align_rejects_larger_drift():
    assert not ratios_align(4.0, 4.3)


def test_record_probe_invalidates_misaligned_stable_ratio():
    cache = AppScrollCache(
        entries={
            "Google Chrome": ScrollCacheEntry(
                viewport=BoundingBox(2, 2, 6, 6),
                reference_before_array=np.zeros((10, 10, 3), dtype=np.uint8),
                stable_scroll_ratio=1.0,
            )
        }
    )

    cache.record_probe(
        "Google Chrome",
        BoundingBox(3, 3, 5, 5),
        np.ones((10, 10, 3), dtype=np.uint8),
        scroll_ratio=1.4,
    )

    entry = cache.entries["Google Chrome"]
    assert entry.stable_scroll_ratio is None
    assert entry.pending_probe_ratio == 1.4


def test_record_calibrated_invalidates_misaligned_stable_ratio():
    cache = AppScrollCache(
        entries={
            "Google Chrome": ScrollCacheEntry(
                viewport=BoundingBox(2, 2, 6, 6),
                reference_before_array=np.zeros((10, 10, 3), dtype=np.uint8),
                stable_scroll_ratio=1.0,
            )
        }
    )

    cache.record_calibrated(
        "Google Chrome",
        BoundingBox(3, 3, 5, 5),
        np.ones((10, 10, 3), dtype=np.uint8),
        actual_scroll_ratio=1.2,
    )

    entry = cache.entries["Google Chrome"]
    assert entry.viewport == BoundingBox(3, 3, 5, 5)
    assert entry.stable_scroll_ratio is None
    assert entry.pending_probe_ratio is None


def test_cache_decision_keeps_entry_snapshot_after_cache_update():
    cache = AppScrollCache()
    before = np.zeros((10, 10, 3), dtype=np.uint8)
    after = np.ones((10, 10, 3), dtype=np.uint8)
    original_viewport = BoundingBox(2, 2, 6, 6)
    updated_viewport = BoundingBox(3, 3, 5, 5)

    cache.record_probe("Google Chrome", original_viewport, before, scroll_ratio=1.0)
    cache.record_probe("Google Chrome", original_viewport, before, scroll_ratio=1.0)

    decision = cache.evaluate_reuse(
        probe_skip_enabled=True,
        app_name="Google Chrome",
        current=after,
        cursor_local=(5, 5),
    )
    assert decision.cache_entry is not None

    cache.record_calibrated("Google Chrome", updated_viewport, after)

    assert decision.cache_entry.viewport == original_viewport
    assert cache.entries["Google Chrome"].viewport == updated_viewport


def test_outside_viewport_check_uses_refined_viewport():
    entry = ScrollCacheEntry(
        viewport=BoundingBox(2, 2, 6, 6),
        reference_before_array=np.zeros((720, 1280, 3), dtype=np.uint8),
    )
    assert entry.viewport is not None
    current = np.zeros((10, 10, 3), dtype=np.uint8)
    cached = current.copy()
    cached[2:8, 2:8, :] = 255
    assert outside_viewport_match_ratio(current, cached, entry.viewport) == 1.0


def test_evaluate_reuse_reports_cursor_outside_reason():
    cache = AppScrollCache(
        entries={
            "Google Chrome": ScrollCacheEntry(
                viewport=BoundingBox(2, 2, 400, 320),
                reference_before_array=np.zeros((400, 800, 3), dtype=np.uint8),
                stable_scroll_ratio=1.0,
            )
        }
    )

    decision = cache.evaluate_reuse(
        probe_skip_enabled=True,
        app_name="Google Chrome",
        current=np.zeros((400, 800, 3), dtype=np.uint8),
        cursor_local=(0, 0),
    )

    assert decision == ProbeSkipDecision(
        use_cached_probe=False,
        cache_key="Google Chrome",
        cache_entry=cache.entries["Google Chrome"],
        cache_debug_reason="cursor outside cached viewport",
    )


def test_evaluate_reuse_reports_small_viewport_reason():
    cache = AppScrollCache(
        entries={
            "Google Chrome": ScrollCacheEntry(
                viewport=BoundingBox(2, 2, 400, 250),
                reference_before_array=np.zeros((400, 800, 3), dtype=np.uint8),
                stable_scroll_ratio=1.0,
            )
        }
    )

    decision = cache.evaluate_reuse(
        probe_skip_enabled=True,
        app_name="Google Chrome",
        current=np.zeros((400, 800, 3), dtype=np.uint8),
        cursor_local=(10, 10),
    )

    assert decision == ProbeSkipDecision(
        use_cached_probe=False,
        cache_key="Google Chrome",
        cache_entry=cache.entries["Google Chrome"],
        cache_debug_reason="cached viewport too small",
    )


def test_evaluate_reuse_reports_missing_viewport_after_viewport_invalidation():
    cache = AppScrollCache(
        entries={
            "Google Chrome": ScrollCacheEntry(
                viewport=None,
                reference_before_array=None,
                stable_scroll_ratio=1.0,
            )
        }
    )

    decision = cache.evaluate_reuse(
        probe_skip_enabled=True,
        app_name="Google Chrome",
        current=np.zeros((10, 10, 3), dtype=np.uint8),
        cursor_local=(5, 5),
    )

    assert decision == ProbeSkipDecision(
        use_cached_probe=False,
        cache_key="Google Chrome",
        cache_entry=cache.entries["Google Chrome"],
        cache_debug_reason="no cached viewport",
    )


def test_evaluate_reuse_marks_outside_viewport_change():
    current = np.zeros((400, 800, 3), dtype=np.uint8)
    cached = current.copy()
    cached[:60, :, :] = 255
    cache = AppScrollCache(
        entries={
            "Google Chrome": ScrollCacheEntry(
                viewport=BoundingBox(2, 2, 400, 320),
                reference_before_array=cached,
                stable_scroll_ratio=1.0,
            )
        }
    )

    decision = cache.evaluate_reuse(
        probe_skip_enabled=True,
        app_name="Google Chrome",
        current=current,
        cursor_local=(5, 5),
    )

    assert not decision.use_cached_probe
    assert decision.outside_viewport_changed
    assert decision.cache_debug_reason is not None
    assert decision.cache_debug_reason.startswith("outside match ")
    assert " < 0.90" in decision.cache_debug_reason


def test_record_probe_and_calibrated_update_cache():
    cache = AppScrollCache()
    viewport = BoundingBox(2, 2, 6, 6)
    before = np.zeros((10, 10, 3), dtype=np.uint8)
    after = np.ones((10, 10, 3), dtype=np.uint8)

    cache.record_probe("Google Chrome", viewport, before, scroll_ratio=1.2)
    entry = cache.entries["Google Chrome"]
    assert entry.viewport == viewport
    assert np.array_equal(entry.reference_before_array, before)

    cache.record_calibrated("Google Chrome", BoundingBox(3, 3, 5, 5), after)
    assert cache.entries["Google Chrome"].viewport == BoundingBox(3, 3, 5, 5)
    assert np.array_equal(cache.entries["Google Chrome"].reference_before_array, after)

    cache.invalidate_viewport("Google Chrome")
    assert cache.entries["Google Chrome"].viewport is None
    assert cache.entries["Google Chrome"].reference_before_array is None
