"""Helpers for probe-scroll caching logic.

This module is Talon-free so behavior can be unit tested directly.
"""

import math
from dataclasses import dataclass, field, replace

import numpy as np

try:
    from .scroll_detection import BoundingBox
except ImportError:
    from scroll_detection import BoundingBox


DEFAULT_OUTSIDE_MATCH_THRESHOLD = 0.90
OUTSIDE_MATCH_PIXEL_TOLERANCE = 1
MIN_CACHED_VIEWPORT_HEIGHT = 300
SCROLL_RATIO_ALIGNMENT_REL_TOL = 0.05
VIEWPORT_VERTICAL_BOUNDS_TOLERANCE = 12
NOTIFICATION_CENTER_APP_NAME = "Notification Center"


@dataclass(frozen=True)
class ScrollCacheEntry:
    """Cache entry keyed by app name for enhanced scrolling."""

    viewport: BoundingBox | None
    reference_before_array: np.ndarray | None
    stable_scroll_ratio: float | None = None
    pending_probe_ratio: float | None = None


@dataclass(frozen=True)
class ProbeSkipDecision:
    """Decision for whether the cached viewport can replace the probe scroll."""

    use_cached_probe: bool
    cache_key: str | None
    cache_entry: ScrollCacheEntry | None = None
    cache_debug_reason: str | None = None
    outside_viewport_changed: bool = False


def outside_viewport_match_ratio(
    current: np.ndarray, cached: np.ndarray, viewport: BoundingBox
) -> float:
    """Return tolerant per-pixel match ratio outside the viewport rectangle."""
    # Reuse only applies when both captured arrays have identical dimensions.
    if current.shape != cached.shape:
        return 0.0
    if current.size == 0:
        return 0.0

    height, width = current.shape[:2]
    x1 = max(0, viewport.x)
    y1 = max(0, viewport.y)
    x2 = min(width, viewport.x + viewport.width)
    y2 = min(height, viewport.y + viewport.height)

    mask = np.ones((height, width), dtype=bool)
    if x1 < x2 and y1 < y2:
        mask[y1:y2, x1:x2] = False
    if not np.any(mask):
        return 1.0

    if current.ndim == 3:
        diff = np.abs(current.astype(np.int16) - cached.astype(np.int16))
        same = np.all(diff <= OUTSIDE_MATCH_PIXEL_TOLERANCE, axis=-1)
    else:
        diff = np.abs(current.astype(np.int16) - cached.astype(np.int16))
        same = diff <= OUTSIDE_MATCH_PIXEL_TOLERANCE
    return float(np.mean(same[mask]))


def update_ratio_stability(
    entry: ScrollCacheEntry, current_ratio: float
) -> ScrollCacheEntry:
    """Record a probe ratio, promoting it after a matching second measurement."""
    if entry.stable_scroll_ratio is not None:
        return entry

    if entry.pending_probe_ratio is not None and ratios_align(
        entry.pending_probe_ratio, current_ratio
    ):
        return replace(
            entry,
            stable_scroll_ratio=entry.pending_probe_ratio,
            pending_probe_ratio=None,
        )

    return replace(entry, pending_probe_ratio=current_ratio)


def ratios_align(expected_ratio: float, actual_ratio: float) -> bool:
    """Return True when two scroll ratios are close enough to be treated as aligned."""
    return math.isclose(
        expected_ratio,
        actual_ratio,
        rel_tol=SCROLL_RATIO_ALIGNMENT_REL_TOL,
        abs_tol=0.01,
    )


@dataclass
class AppScrollCache:
    """Encapsulates probe-scroll cache entries and reuse policy."""

    entries: dict[str, ScrollCacheEntry] = field(default_factory=dict)

    def cache_key_for_app(self, app_name: str | None) -> str | None:
        """Return cache key for an app name, or None when caching is bypassed."""
        if not app_name or app_name == NOTIFICATION_CENTER_APP_NAME:
            return None
        return app_name

    def evaluate_reuse(
        self,
        probe_skip_enabled: bool,
        app_name: str | None,
        current: np.ndarray,
        cursor_local: tuple[float, float],
        threshold: float = DEFAULT_OUTSIDE_MATCH_THRESHOLD,
    ) -> ProbeSkipDecision:
        """Decide whether the cached viewport can replace the probe scroll."""
        if not probe_skip_enabled:
            return ProbeSkipDecision(
                use_cached_probe=False,
                cache_key=None,
                cache_debug_reason="probe skip disabled",
            )
        if app_name == NOTIFICATION_CENTER_APP_NAME:
            return ProbeSkipDecision(
                use_cached_probe=False,
                cache_key=None,
                cache_debug_reason="cache bypassed for Notification Center",
            )

        cache_key = self.cache_key_for_app(app_name)
        if cache_key is None:
            return ProbeSkipDecision(
                use_cached_probe=False,
                cache_key=None,
                cache_debug_reason="app unavailable for cache",
            )

        cache_entry = self.entries.get(cache_key)
        if cache_entry is None:
            return ProbeSkipDecision(
                use_cached_probe=False,
                cache_key=cache_key,
                cache_debug_reason="no cached viewport",
            )
        if cache_entry.stable_scroll_ratio is None:
            return ProbeSkipDecision(
                use_cached_probe=False,
                cache_key=cache_key,
                cache_entry=cache_entry,
                cache_debug_reason="scroll ratio not stable yet",
            )
        if cache_entry.viewport is None or cache_entry.reference_before_array is None:
            return ProbeSkipDecision(
                use_cached_probe=False,
                cache_key=cache_key,
                cache_entry=cache_entry,
                cache_debug_reason="no cached viewport",
            )
        if cache_entry.viewport.height < MIN_CACHED_VIEWPORT_HEIGHT:
            return ProbeSkipDecision(
                use_cached_probe=False,
                cache_key=cache_key,
                cache_entry=cache_entry,
                cache_debug_reason="cached viewport too small",
            )
        if not cache_entry.viewport.contains_point(cursor_local):
            return ProbeSkipDecision(
                use_cached_probe=False,
                cache_key=cache_key,
                cache_entry=cache_entry,
                cache_debug_reason="cursor outside cached viewport",
            )

        match_ratio = outside_viewport_match_ratio(
            current,
            cache_entry.reference_before_array,
            cache_entry.viewport,
        )
        if match_ratio < threshold:
            return ProbeSkipDecision(
                use_cached_probe=False,
                cache_key=cache_key,
                cache_entry=cache_entry,
                cache_debug_reason=f"outside match {match_ratio:.2f} < {threshold:.2f}",
                outside_viewport_changed=True,
            )

        return ProbeSkipDecision(
            use_cached_probe=True,
            cache_key=cache_key,
            cache_entry=cache_entry,
            cache_debug_reason=f"outside match {match_ratio:.2f}",
        )

    def record_probe(
        self,
        cache_key: str | None,
        viewport: BoundingBox,
        reference_before_array: np.ndarray,
        scroll_ratio: float,
    ) -> None:
        """Store probe results for a cache key, creating the entry if needed."""
        if not cache_key:
            return

        entry = self.entries.get(cache_key)
        if entry is None:
            entry = ScrollCacheEntry(
                viewport=viewport,
                reference_before_array=reference_before_array.copy(),
            )
        else:
            if entry.stable_scroll_ratio is not None and not ratios_align(
                entry.stable_scroll_ratio, scroll_ratio
            ):
                entry = replace(
                    entry,
                    stable_scroll_ratio=None,
                    pending_probe_ratio=None,
                )
            entry = replace(
                entry,
                viewport=viewport,
                reference_before_array=reference_before_array.copy(),
            )
        self.entries[cache_key] = update_ratio_stability(entry, scroll_ratio)

    def record_calibrated(
        self,
        cache_key: str | None,
        viewport: BoundingBox,
        reference_before_array: np.ndarray,
        actual_scroll_ratio: float,
        used_cached_probe: bool = False,
    ) -> None:
        """Refresh an existing cache entry after successful calibrated detection."""
        if not cache_key:
            return

        entry = self.entries.get(cache_key)
        if entry is None:
            return
        assert entry.viewport is not None
        viewport_changed = not viewport.has_similar_vertical_bounds(
            entry.viewport, tolerance=VIEWPORT_VERTICAL_BOUNDS_TOLERANCE
        )
        if used_cached_probe:
            assert entry.stable_scroll_ratio is not None
            if viewport_changed and not ratios_align(
                entry.stable_scroll_ratio, actual_scroll_ratio
            ):
                # This calibrated-step ratio check is only used on cache
                # hits, where we skipped the smaller probe measurement. We
                # only invalidate the ratio when the viewport changed too;
                # otherwise a mismatched calibrated ratio is more likely to
                # mean we hit the top or bottom of the document than that
                # the cached ratio went stale.
                entry = replace(
                    entry,
                    stable_scroll_ratio=None,
                    pending_probe_ratio=None,
                )
        elif entry.pending_probe_ratio is not None:
            # A successful calibrated scroll can confirm the immediately
            # preceding probe measurement, so we do not need to wait for a
            # second probe scroll to trust the ratio.
            if ratios_align(entry.pending_probe_ratio, actual_scroll_ratio):
                entry = replace(
                    entry,
                    stable_scroll_ratio=entry.pending_probe_ratio,
                    pending_probe_ratio=None,
                )
        self.entries[cache_key] = replace(
            entry,
            viewport=viewport,
            reference_before_array=reference_before_array.copy(),
        )

    def invalidate_viewport(self, cache_key: str | None) -> None:
        """Clear cached viewport state while keeping any stable scroll ratio."""
        if not cache_key:
            return
        entry = self.entries.get(cache_key)
        if entry is None:
            return
        self.entries[cache_key] = replace(
            entry,
            viewport=None,
            reference_before_array=None,
        )
