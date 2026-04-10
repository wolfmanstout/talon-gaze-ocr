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
    scroll_ratio: float | None = None
    scroll_ratio_confirmed: bool = False

    def __post_init__(self):
        assert self.scroll_ratio is not None or not self.scroll_ratio_confirmed


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


def refresh_viewport_cache(
    entry: ScrollCacheEntry,
    viewport: BoundingBox,
    reference_before_array: np.ndarray,
) -> ScrollCacheEntry:
    """Return an entry with refreshed viewport/reference cache state."""
    return replace(
        entry,
        viewport=viewport,
        reference_before_array=reference_before_array.copy(),
    )


def update_ratio_state(
    entry: ScrollCacheEntry,
    measured_ratio: float,
    *,
    allow_confirmed_reset_on_mismatch: bool,
) -> ScrollCacheEntry:
    """Record a measured ratio, optionally demoting confirmed ratios on mismatch."""
    if entry.scroll_ratio is None:
        return replace(entry, scroll_ratio=measured_ratio, scroll_ratio_confirmed=False)

    if ratios_align(entry.scroll_ratio, measured_ratio):
        if entry.scroll_ratio_confirmed:
            return entry
        return replace(entry, scroll_ratio_confirmed=True)

    if entry.scroll_ratio_confirmed and not allow_confirmed_reset_on_mismatch:
        return entry

    return replace(
        entry,
        scroll_ratio=measured_ratio,
        scroll_ratio_confirmed=False,
    )


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
        if not cache_entry.scroll_ratio_confirmed:
            return ProbeSkipDecision(
                use_cached_probe=False,
                cache_key=cache_key,
                cache_entry=cache_entry,
                cache_debug_reason="scroll ratio not confirmed yet",
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

    def record_scroll_measurement(
        self,
        cache_key: str | None,
        viewport: BoundingBox,
        reference_before_array: np.ndarray,
        measured_ratio: float,
        *,
        is_probe: bool,
        used_cached_probe: bool = False,
    ) -> None:
        """Record a probe or calibrated measurement and refresh viewport cache."""
        if not cache_key:
            return

        entry = self.entries.get(cache_key) or ScrollCacheEntry(
            viewport=None,
            reference_before_array=None,
        )

        viewport_changed = (
            entry.viewport is not None
            and not viewport.has_similar_vertical_bounds(
                entry.viewport,
                tolerance=VIEWPORT_VERTICAL_BOUNDS_TOLERANCE,
            )
        )
        # Probe measurements are trusted enough to demote a confirmed ratio on
        # mismatch. Calibrated measurements normally are not, because they are
        # more likely to be clipped at the top or bottom of a document. The
        # one exception is a cache hit whose viewport changed, since that is a
        # stronger signal that the cached confirmed ratio went stale.
        entry = update_ratio_state(
            entry,
            measured_ratio,
            allow_confirmed_reset_on_mismatch=(
                is_probe or (used_cached_probe and viewport_changed)
            ),
        )
        self.entries[cache_key] = refresh_viewport_cache(
            entry,
            viewport,
            reference_before_array,
        )

    def invalidate_viewport(self, cache_key: str | None) -> None:
        """Clear cached viewport state while keeping any cached scroll ratio."""
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
