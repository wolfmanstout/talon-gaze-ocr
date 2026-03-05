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
    viewport_unrefined_img: BoundingBox
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


def is_full_frame_mostly_unchanged(
    current: np.ndarray, cached: np.ndarray, threshold: float = 0.95
) -> bool:
    """True when full-frame screenshots are mostly unchanged."""
    return full_frame_match_ratio(current, cached) >= threshold


def is_viewport_close(
    cached_unrefined: BoundingBox,
    estimated_unrefined: BoundingBox,
    frame_w: int,
    frame_h: int,
    threshold: float = 0.08,
) -> bool:
    """Compare viewport center/size deltas normalized by frame dimensions."""
    if frame_w <= 0 or frame_h <= 0:
        return False

    cached_cx = cached_unrefined.x + (cached_unrefined.width / 2.0)
    cached_cy = cached_unrefined.y + (cached_unrefined.height / 2.0)
    estimated_cx = estimated_unrefined.x + (estimated_unrefined.width / 2.0)
    estimated_cy = estimated_unrefined.y + (estimated_unrefined.height / 2.0)

    center_dx = abs(cached_cx - estimated_cx) / frame_w
    center_dy = abs(cached_cy - estimated_cy) / frame_h
    width_delta = abs(cached_unrefined.width - estimated_unrefined.width) / frame_w
    height_delta = abs(cached_unrefined.height - estimated_unrefined.height) / frame_h

    return (
        center_dx <= threshold
        and center_dy <= threshold
        and width_delta <= threshold
        and height_delta <= threshold
    )


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
