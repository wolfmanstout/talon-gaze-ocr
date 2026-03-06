"""Helpers for probe-scroll caching logic.

This module is Talon-free so behavior can be unit tested directly.
"""

import math
from dataclasses import dataclass

import numpy as np

try:
    from .scroll_detection import BoundingBox
except ImportError:
    from scroll_detection import BoundingBox


@dataclass
class ScrollProbeCacheEntry:
    """Cache entry keyed by app name for enhanced scrolling."""

    viewport_refined_img: BoundingBox
    reference_before_full: np.ndarray
    stable_scroll_ratio: float | None = None
    pending_probe_ratio: float | None = None
    pending_probe_amount: int | None = None


def full_frame_match_ratio(current: np.ndarray, cached: np.ndarray) -> float:
    """Return exact per-pixel match ratio for two full-frame screenshots."""
    if current.shape != cached.shape:
        return 0.0
    if current.size == 0:
        return 0.0
    same = (
        np.all(current == cached, axis=-1) if current.ndim == 3 else current == cached
    )
    return float(np.mean(same))


def outside_viewport_match_ratio(
    current: np.ndarray, cached: np.ndarray, viewport: BoundingBox
) -> float:
    """Return exact per-pixel match ratio outside the viewport rectangle."""
    if current.shape != cached.shape:
        return 0.0
    if current.size == 0:
        return 0.0

    H, W = current.shape[:2]
    x1 = max(0, viewport.x)
    y1 = max(0, viewport.y)
    x2 = min(W, viewport.x + viewport.width)
    y2 = min(H, viewport.y + viewport.height)

    mask = np.ones((H, W), dtype=bool)
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
    threshold: float = 0.95,
) -> bool:
    """True when most pixels outside viewport are unchanged."""
    return outside_viewport_match_ratio(current, cached, viewport) >= threshold


def ratios_match(a: float, b: float) -> bool:
    """Float-tolerant ratio equality check."""
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def update_ratio_stability(
    entry: ScrollProbeCacheEntry, current_ratio: float, probe_amount: int
) -> None:
    """Promote ratio to stable after two consecutive matching probe readings."""
    if entry.stable_scroll_ratio is not None:
        return

    if (
        entry.pending_probe_ratio is not None
        and entry.pending_probe_amount == probe_amount
        and ratios_match(entry.pending_probe_ratio, current_ratio)
    ):
        entry.stable_scroll_ratio = current_ratio
        return

    entry.pending_probe_ratio = current_ratio
    entry.pending_probe_amount = probe_amount
