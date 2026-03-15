"""Helpers for probe-scroll caching logic.

This module is Talon-free so behavior can be unit tested directly.
"""

import math
from dataclasses import dataclass, field

import numpy as np

try:
    from .scroll_detection import BoundingBox
except ImportError:
    from scroll_detection import BoundingBox


DEFAULT_OUTSIDE_MATCH_THRESHOLD = 0.90
NOTIFICATION_CENTER_APP_NAME = "Notification Center"


@dataclass
class ScrollCacheEntry:
    """Cache entry keyed by app name for enhanced scrolling."""

    viewport_refined_img: BoundingBox
    reference_before_full: np.ndarray
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
    """Return exact per-pixel match ratio outside the viewport rectangle."""
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

    same = (
        np.all(current == cached, axis=-1) if current.ndim == 3 else current == cached
    )
    return float(np.mean(same[mask]))


def is_outside_viewport_mostly_unchanged(
    current: np.ndarray,
    cached: np.ndarray,
    viewport: BoundingBox,
    threshold: float = DEFAULT_OUTSIDE_MATCH_THRESHOLD,
) -> bool:
    """True when most pixels outside viewport are unchanged."""
    return outside_viewport_match_ratio(current, cached, viewport) >= threshold


def viewport_contains_point(viewport: BoundingBox, point: tuple[float, float]) -> bool:
    """True when the point lies inside the viewport rectangle."""
    px, py = point
    return (
        viewport.x <= px < viewport.x + viewport.width
        and viewport.y <= py < viewport.y + viewport.height
    )


def ratios_match(a: float, b: float) -> bool:
    """Float-tolerant ratio equality check."""
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def update_ratio_stability(entry: ScrollCacheEntry, current_ratio: float) -> None:
    """Promote ratio to stable after two consecutive matching probe readings."""
    if entry.stable_scroll_ratio is not None:
        return

    if entry.pending_probe_ratio is not None and ratios_match(
        entry.pending_probe_ratio, current_ratio
    ):
        entry.stable_scroll_ratio = current_ratio
        return

    entry.pending_probe_ratio = current_ratio


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
                cache_debug_reason="no cached viewport",
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
        if not viewport_contains_point(cache_entry.viewport_refined_img, cursor_local):
            return ProbeSkipDecision(
                use_cached_probe=False,
                cache_key=cache_key,
                cache_entry=cache_entry,
                cache_debug_reason="cursor outside cached viewport",
            )

        match_ratio = outside_viewport_match_ratio(
            current,
            cache_entry.reference_before_full,
            cache_entry.viewport_refined_img,
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
        viewport_refined_img: BoundingBox,
        reference_before_full: np.ndarray,
        scroll_ratio: float,
    ) -> None:
        """Store probe results for a cache key, creating the entry if needed."""
        if not cache_key:
            return

        entry = self.entries.get(cache_key)
        if entry is None:
            entry = ScrollCacheEntry(
                viewport_refined_img=viewport_refined_img,
                reference_before_full=reference_before_full.copy(),
            )
            self.entries[cache_key] = entry
        else:
            entry.viewport_refined_img = viewport_refined_img
            entry.reference_before_full = reference_before_full.copy()
        update_ratio_stability(entry, scroll_ratio)

    def record_calibrated(
        self,
        cache_key: str | None,
        viewport_refined_img: BoundingBox,
        reference_before_full: np.ndarray,
    ) -> None:
        """Refresh an existing cache entry after successful calibrated detection."""
        if not cache_key:
            return

        entry = self.entries.get(cache_key)
        if entry is None:
            return
        entry.viewport_refined_img = viewport_refined_img
        entry.reference_before_full = reference_before_full.copy()

    def invalidate(self, cache_key: str | None) -> None:
        """Remove a cache entry if it exists."""
        if cache_key and cache_key in self.entries:
            del self.entries[cache_key]
