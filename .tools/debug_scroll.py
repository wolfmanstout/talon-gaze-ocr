#!/usr/bin/env python3
"""
Visual debugging tool for scroll detection algorithm (Three-Phase Approach).

Generates comprehensive debug visualizations showing:
- Phase 1: Initial viewport estimation (change map, row/col weights)
- Phase 2: Scroll distance calculation (profiles, cross-correlation)
- Phase 3: Viewport refinement (match map, refined weights)

Usage:
    uv run debug_scroll.py --from-json data/scroll_success_*.json
    uv run debug_scroll.py before.png after.png [--cursor-pos X Y]
"""

import json
import os
import sys
from dataclasses import dataclass, field

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from PIL import Image

# Import Kadane algorithms from parent module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scroll_detection import (
    get_best_1d_anchored,
    get_best_1d_range_constrained,
)

# Default parameters matching production scroll_detection.py
DEFAULT_MIN_SCROLL_DISTANCE = 20
DEFAULT_MIN_VIEWPORT_HEIGHT = 150
DEFAULT_MIN_VIEWPORT_WIDTH = 150
DEFAULT_PIXEL_MATCH_TOLERANCE = 20
DEFAULT_CHANGE_THRESHOLD_RATIO = 0.15
DEFAULT_DENSITY_THRESHOLD = 0.01
DEFAULT_NUM_STRIPS = 16


@dataclass
class DebugData:
    """Container for all debug information from three-phase detection."""

    # Original images
    img_b: np.ndarray
    img_a: np.ndarray

    # Grayscale images
    gray_b: np.ndarray | None = None
    gray_a: np.ndarray | None = None

    # Gradient images (for Phase 2 visualization)
    grad_b: np.ndarray | None = None
    grad_a: np.ndarray | None = None

    # Phase 1: Initial viewport estimation
    change_map: np.ndarray | None = None  # Binary diff > tolerance
    row_sums_phase1: np.ndarray | None = None  # Raw row sums of changed pixels
    row_weights_phase1: np.ndarray | None = None  # Mean-normalized row weights
    col_sums_phase1: np.ndarray | None = None  # Raw column sums
    col_weights_phase1: np.ndarray | None = None  # Mean-normalized col weights
    row_range_phase1: tuple | None = None  # (r1, r2) row range found
    col_range_phase1: tuple | None = None  # (c1, c2) column range found
    initial_viewport: tuple | None = None  # (x, y, w, h)

    # Phase 2: Scroll distance estimation (summed strip correlation)
    profile_before: np.ndarray | None = None  # Row-summed profile (full width, for viz)
    profile_after: np.ndarray | None = None
    correlation_curve: np.ndarray | None = (
        None  # Summed correlation normalized by overlap
    )
    strip_correlations: np.ndarray | None = (
        None  # Per-strip correlations (strips x distances)
    )
    peak_idx: int | None = None
    scroll_distance: int | None = None
    peak_value: float | None = None  # Correlation value at detected peak
    second_peak_idx: int | None = None  # Index of second highest peak
    second_peak_value: float | None = None  # Value of second highest peak

    # Phase 3: Viewport refinement
    match_map: np.ndarray | None = None  # Aligned regions match (overlap)
    static_match_map: np.ndarray | None = None  # Same-position regions match
    row_weights_phase3: np.ndarray | None = None
    col_weights_phase3: np.ndarray | None = None
    row_range_phase3: tuple | None = None  # (r1, r2)
    col_range_phase3: tuple | None = None  # (c1, c2)
    refined_viewport: tuple | None = None  # (x, y, w, h)
    after_bbox: tuple | None = None  # (x, y, w, h)

    # Parameters
    params: dict = field(default_factory=dict)

    # Failure info
    failure_phase: int | None = None  # Which phase failed (1, 2, or 3)
    failure_reason: str | None = None


def parse_from_json(json_path):
    """
    Parse a scroll_*.json file and derive before/after image paths, cursor position,
    existing_viewport, and scroll_direction if present.
    """
    if not os.path.exists(json_path):
        raise click.BadParameter(f"JSON file not found: {json_path}")

    base = json_path.rsplit(".json", 1)[0]
    before_path = base + "_before.png"
    after_path = base + "_after.png"

    if not os.path.exists(before_path):
        raise click.BadParameter(f"Before image not found: {before_path}")
    if not os.path.exists(after_path):
        raise click.BadParameter(f"After image not found: {after_path}")

    with open(json_path) as f:
        data = json.load(f)

    cursor_pos = None
    cp = data.get("cursor_position", {})
    if "x" in cp and "y" in cp:
        cursor_pos = (cp["x"], cp["y"])

    existing_viewport = None
    ev = data.get("existing_viewport")
    if ev and all(k in ev for k in ("x", "y", "width", "height")):
        existing_viewport = (ev["x"], ev["y"], ev["width"], ev["height"])

    # Default to "down" for backwards compatibility with old log files
    scroll_direction = data.get("scroll_direction", "down")

    return before_path, after_path, cursor_pos, existing_viewport, scroll_direction


def detect_scroll_debug(img_before, img_after, cursor_pos, params=None):
    """
    Instrumented version of detect_scroll that returns debug data.

    This mirrors the algorithm with instrumentation to collect
    intermediate values for visualization.
    """
    if params is None:
        params = {}

    # Unpack parameters with defaults from module constants
    MIN_SCROLL_DISTANCE = params.get("min_scroll_distance", DEFAULT_MIN_SCROLL_DISTANCE)
    MIN_VIEWPORT_HEIGHT = params.get("min_viewport_height", DEFAULT_MIN_VIEWPORT_HEIGHT)
    MIN_VIEWPORT_WIDTH = params.get("min_viewport_width", DEFAULT_MIN_VIEWPORT_WIDTH)
    PIXEL_MATCH_TOLERANCE = params.get(
        "pixel_match_tolerance", DEFAULT_PIXEL_MATCH_TOLERANCE
    )
    CHANGE_THRESHOLD_RATIO = params.get(
        "change_threshold_ratio", DEFAULT_CHANGE_THRESHOLD_RATIO
    )
    DENSITY_THRESHOLD = params.get("density_threshold", DEFAULT_DENSITY_THRESHOLD)
    scroll_direction = params.get("scroll_direction", "down")

    # Initialize debug data
    debug = DebugData(
        img_b=img_before,
        img_a=img_after,
        params=params,
    )

    cx, cy = cursor_pos if cursor_pos else (0, 0)
    cx, cy = int(cx), int(cy)

    # Preprocessing: Convert to grayscale
    if img_before.ndim == 3:
        gb = (
            img_before[..., 0] * 0.2126
            + img_before[..., 1] * 0.7152
            + img_before[..., 2] * 0.0722
        )
        ga = (
            img_after[..., 0] * 0.2126
            + img_after[..., 1] * 0.7152
            + img_after[..., 2] * 0.0722
        )
    else:
        gb, ga = img_before.astype(float), img_after.astype(float)

    debug.gray_b = gb
    debug.gray_a = ga

    # Compute and store gradients for visualization
    debug.grad_b = np.diff(gb, axis=0)
    debug.grad_a = np.diff(ga, axis=0)

    H, W = gb.shape

    # === Phase 1: Initial Viewport Estimation ===
    existing_viewport = params.get("existing_viewport")

    if existing_viewport is not None:
        # Skip Phase 1 - use provided viewport
        debug.initial_viewport = existing_viewport
        # Still compute change map for visualization
        diff = np.abs(ga.astype(float) - gb.astype(float))
        changed = diff > PIXEL_MATCH_TOLERANCE
        debug.change_map = changed
    else:
        diff = np.abs(ga.astype(float) - gb.astype(float))
        changed = diff > PIXEL_MATCH_TOLERANCE
        debug.change_map = changed

        row_sums = np.sum(changed, axis=1).astype(float)
        debug.row_sums_phase1 = row_sums
        active_rows = row_sums > 0

        if not np.any(active_rows):
            debug.failure_phase = 1
            debug.failure_reason = "No changed pixels detected"
            return debug

        mean_row_activity = np.mean(row_sums[active_rows])
        row_weights = row_sums - (CHANGE_THRESHOLD_RATIO * mean_row_activity)
        debug.row_weights_phase1 = row_weights

        r1, r2, r_score = get_best_1d_anchored(row_weights, cy)
        debug.row_range_phase1 = (r1, r2)

        if r_score <= 0 or r1 > r2:
            debug.failure_phase = 1
            debug.failure_reason = f"No valid row range (score={r_score:.2f})"
            return debug

        changed_cropped = changed[r1 : r2 + 1, :]
        col_sums = np.sum(changed_cropped, axis=0).astype(float)
        debug.col_sums_phase1 = col_sums
        active_cols = col_sums > 0

        if not np.any(active_cols):
            debug.failure_phase = 1
            debug.failure_reason = "No changed pixels in row range"
            return debug

        mean_col_activity = np.mean(col_sums[active_cols])
        col_weights = col_sums - (CHANGE_THRESHOLD_RATIO * mean_col_activity)
        debug.col_weights_phase1 = col_weights

        c1, c2, c_score = get_best_1d_anchored(col_weights, cx)
        debug.col_range_phase1 = (c1, c2)

        if c_score <= 0 or c1 > c2:
            debug.failure_phase = 1
            debug.failure_reason = f"No valid column range (score={c_score:.2f})"
            return debug

        width = c2 - c1 + 1
        height = r2 - r1 + 1
        debug.initial_viewport = (c1, r1, width, height)

        # Validate viewport size
        if height < MIN_VIEWPORT_HEIGHT or width < MIN_VIEWPORT_WIDTH:
            debug.failure_phase = 1
            debug.failure_reason = f"Viewport too small ({width}x{height})"
            return debug

    # === Phase 2: Scroll Distance Estimation (Summed Strip Correlation with NCC) ===
    NUM_STRIPS = params.get("num_strips", DEFAULT_NUM_STRIPS)
    MIN_OVERLAP_HEIGHT = 100  # Not configurable via CLI

    x, y, w, h = debug.initial_viewport

    before_crop = gb[y : y + h, x : x + w]
    after_crop = ga[y : y + h, x : x + w]

    # Compute vertical gradients (row-to-row differences) with abs applied immediately
    grad_before = np.abs(np.diff(before_crop, axis=0))
    grad_after = np.abs(np.diff(after_crop, axis=0))

    # Use gradient height for correlation (h - 1 due to diff)
    h_grad = grad_before.shape[0]

    # Store full-width profiles for visualization
    debug.profile_before = np.sum(grad_before, axis=1).astype(float)
    debug.profile_after = np.sum(grad_after, axis=1).astype(float)

    # Constraint: overlap = h_grad - d >= MIN_OVERLAP_HEIGHT
    max_d = h_grad - MIN_OVERLAP_HEIGHT
    if max_d <= MIN_SCROLL_DISTANCE:
        debug.failure_phase = 2
        debug.failure_reason = (
            "Viewport too small for minimum scroll distance with required overlap"
        )
        return debug

    # 1. Profile Construction - split into vertical strips
    strips_b = np.array_split(grad_before, NUM_STRIPS, axis=1)
    strips_a = np.array_split(grad_after, NUM_STRIPS, axis=1)

    # Sum horizontally within strips, then stack side-by-side
    # Shape: (h_grad, NUM_STRIPS)
    profs_b = np.stack([s.sum(axis=1) for s in strips_b], axis=1)
    profs_a = np.stack([s.sum(axis=1) for s in strips_a], axis=1)

    # 2. Batch Correlation - correlate each strip column-wise
    # Result Shape: (2*h_grad - 1, NUM_STRIPS)
    full_corrs = np.stack(
        [
            np.correlate(profs_b[:, i], profs_a[:, i], mode="full")
            for i in range(NUM_STRIPS)
        ],
        axis=1,
    )

    # Extract lags based on scroll direction
    # For scroll-down (content moves up): positive lags (profs_a shifts right)
    # For scroll-up (content moves down): negative lags (profs_a shifts left)
    # valid_corrs Shape: (h_grad - 1, NUM_STRIPS)
    if scroll_direction == "down":
        valid_corrs = full_corrs[h_grad:, :]  # positive lags
    else:
        # Negative lags: indices 0 to h_grad-2 represent lags -(h_grad-1) to -1
        # Reverse to get lags -1, -2, ..., -(h_grad-1) matching distance 1, 2, ...
        valid_corrs = full_corrs[: h_grad - 1, :][::-1, :]

    # Store per-strip raw correlations for heatmap visualization
    strip_correlations = np.zeros((NUM_STRIPS, h_grad))
    strip_correlations[:, 1:] = valid_corrs.T

    # Sum across all strips for the final raw correlation profile
    raw_corr = np.zeros(h_grad)
    raw_corr[1:] = valid_corrs.sum(axis=1)

    # 3. Vectorized Norm Calculation
    cs_b = np.cumsum(profs_b**2, axis=0)
    cs_a = np.cumsum(profs_a**2, axis=0)
    total_b = cs_b[-1, :]
    total_a = cs_a[-1, :]

    # Energy contributions depend on scroll direction:
    # For scroll-down: compare before[d:] with after[:h_grad-d]
    #   b_contribs = suffix of before, a_contribs = prefix of after
    # For scroll-up: compare before[:h_grad-d] with after[d:]
    #   b_contribs = prefix of before, a_contribs = suffix of after
    if scroll_direction == "down":
        # 'b' segment: Total energy - energy accumulated before the split index
        b_contribs = total_b - cs_b[:-1, :]
        # 'a' segment: Energy accumulated from 0 up to the overlap index (reversed)
        a_contribs = cs_a[:-1, :][::-1, :]
    else:
        # 'b' segment: Energy accumulated from 0 up to the overlap index (reversed)
        b_contribs = cs_b[:-1, :][::-1, :]
        # 'a' segment: Total energy - energy accumulated before the split index
        a_contribs = total_a - cs_a[:-1, :]

    # Sum squared norms across all strips
    norm_b_sq = np.concatenate(([0], b_contribs.sum(axis=1)))
    norm_a_sq = np.concatenate(([0], a_contribs.sum(axis=1)))

    # 4. Final NCC Calculation
    denom = np.sqrt(norm_b_sq * norm_a_sq)
    ncc = np.divide(raw_corr, denom, out=np.zeros_like(raw_corr), where=denom != 0)

    # Normalize per-strip correlations for visualization (divide by same denom)
    for i in range(NUM_STRIPS):
        strip_correlations[i, :] = np.divide(
            strip_correlations[i, :],
            denom,
            out=np.zeros(h_grad),
            where=denom != 0,
        )

    debug.correlation_curve = ncc
    debug.strip_correlations = strip_correlations

    # Find best distance (argmax of NCC)
    valid_range = ncc[MIN_SCROLL_DISTANCE : max_d + 1]
    if len(valid_range) == 0 or np.max(valid_range) <= 0:
        debug.failure_phase = 2
        debug.failure_reason = "No positive correlation found"
        return debug

    best_d = int(np.argmax(valid_range) + MIN_SCROLL_DISTANCE)
    debug.peak_idx = best_d
    debug.scroll_distance = best_d
    debug.peak_value = ncc[best_d]

    # Find second highest peak (at least 20px away from the best peak)
    min_peak_separation = 20
    valid_range_copy = valid_range.copy()
    # Zero out region around the best peak
    best_idx_in_range = best_d - MIN_SCROLL_DISTANCE
    left = max(0, best_idx_in_range - min_peak_separation)
    right = min(len(valid_range_copy), best_idx_in_range + min_peak_separation + 1)
    valid_range_copy[left:right] = 0

    if np.max(valid_range_copy) > 0:
        second_best_idx = int(np.argmax(valid_range_copy))
        debug.second_peak_idx = second_best_idx + MIN_SCROLL_DISTANCE
        debug.second_peak_value = valid_range_copy[second_best_idx]

    if best_d >= H - MIN_VIEWPORT_HEIGHT:
        debug.failure_phase = 2
        debug.failure_reason = f"Scroll distance too large ({best_d}px)"
        return debug

    # === Phase 3: Viewport Refinement ===
    d = best_d  # Use the detected scroll distance
    init_x, _, init_w, _ = debug.initial_viewport
    limit_h = H - d

    if limit_h <= 0:
        debug.failure_phase = 3
        debug.failure_reason = f"Invalid limit_h ({limit_h})"
        return debug

    c1_init = init_x
    c2_init = init_x + init_w

    # --- Row refinement: use initial viewport columns ---
    # Extract the four regions needed for match analysis:
    # - overlap_after/overlap_before: the aligned overlap regions (should match)
    # - dest_in_before: before image at destination position (for dest_static check)
    # - source_in_after: after image at source position (for source_static check)
    if scroll_direction == "down":
        overlap_after_row = ga[:limit_h, c1_init:c2_init]
        overlap_before_row = gb[d:, c1_init:c2_init]
        dest_in_before_row = gb[:limit_h, c1_init:c2_init]
        source_in_after_row = ga[d:, c1_init:c2_init]
    else:
        # Scroll-up: after[d:] matches before[:limit_h]
        overlap_after_row = ga[d:, c1_init:c2_init]
        overlap_before_row = gb[:limit_h, c1_init:c2_init]
        dest_in_before_row = gb[d:, c1_init:c2_init]
        source_in_after_row = ga[:limit_h, c1_init:c2_init]

    # Compute match categories for row refinement
    overlap_diff_row = np.abs(
        overlap_after_row.astype(float) - overlap_before_row.astype(float)
    )
    overlap_match_row = overlap_diff_row < PIXEL_MATCH_TOLERANCE

    dest_static_diff_row = np.abs(
        overlap_after_row.astype(float) - dest_in_before_row.astype(float)
    )
    dest_static_row = dest_static_diff_row < PIXEL_MATCH_TOLERANCE

    source_static_diff_row = np.abs(
        overlap_before_row.astype(float) - source_in_after_row.astype(float)
    )
    source_static_row = source_static_diff_row < PIXEL_MATCH_TOLERANCE

    # Three categories: dynamic (scrolled), static (unchanged), mismatch (outside viewport)
    static_match_row = dest_static_row | source_static_row
    dynamic_match_row = overlap_match_row & ~static_match_row
    static_in_overlap_row = overlap_match_row & static_match_row

    # Store for visualization
    debug.match_map = overlap_match_row
    debug.static_match_map = static_match_row

    # Build 2D weight map: positive for dynamic, near-zero for static, negative for mismatch
    STATIC_EPSILON = 1e-6
    weight_map_row = np.full_like(
        overlap_match_row, -DENSITY_THRESHOLD - 1e-5, dtype=float
    )
    weight_map_row[dynamic_match_row] = 1.0 - DENSITY_THRESHOLD
    weight_map_row[static_in_overlap_row] = STATIC_EPSILON

    # Cursor constraint: detected range must include part of cursor's scroll path [cy-d, cy]
    target_row_min = max(0, cy - d)
    target_row_max = min(cy, limit_h - 1)
    if target_row_min > target_row_max:
        target_row_min = target_row_max

    row_weights = np.sum(weight_map_row, axis=1)
    debug.row_weights_phase3 = row_weights

    r1, r2, row_score = get_best_1d_range_constrained(
        row_weights, target_row_min, target_row_max
    )
    debug.row_range_phase3 = (r1, r2)

    if row_score <= 0:
        debug.failure_phase = 3
        debug.failure_reason = f"No valid refined row range (score={row_score:.2f})"
        return debug

    # --- Column refinement: use full image width with refined rows ---
    # r1, r2 are slice indices; convert to screen rows based on direction
    if scroll_direction == "down":
        # Scroll-down: slice indices equal screen rows
        overlap_after_col = ga[r1 : r2 + 1, :]
        overlap_before_col = gb[r1 + d : r2 + 1 + d, :]
        dest_in_before_col = gb[r1 : r2 + 1, :]
        source_in_after_col = ga[r1 + d : r2 + 1 + d, :]
    else:
        # Scroll-up: screen row = slice index + d
        overlap_after_col = ga[r1 + d : r2 + 1 + d, :]
        overlap_before_col = gb[r1 : r2 + 1, :]
        dest_in_before_col = gb[r1 + d : r2 + 1 + d, :]
        source_in_after_col = ga[r1 : r2 + 1, :]

    # Compute match categories for column refinement
    overlap_diff_col = np.abs(
        overlap_after_col.astype(float) - overlap_before_col.astype(float)
    )
    overlap_match_col = overlap_diff_col < PIXEL_MATCH_TOLERANCE

    dest_static_diff_col = np.abs(
        overlap_after_col.astype(float) - dest_in_before_col.astype(float)
    )
    dest_static_col = dest_static_diff_col < PIXEL_MATCH_TOLERANCE

    source_static_diff_col = np.abs(
        overlap_before_col.astype(float) - source_in_after_col.astype(float)
    )
    source_static_col = source_static_diff_col < PIXEL_MATCH_TOLERANCE

    static_match_col = dest_static_col | source_static_col
    dynamic_match_col = overlap_match_col & ~static_match_col
    static_in_overlap_col = overlap_match_col & static_match_col

    # Build 2D weight map for column refinement
    weight_map_col = np.full_like(
        overlap_match_col, -DENSITY_THRESHOLD - 1e-5, dtype=float
    )
    weight_map_col[dynamic_match_col] = 1.0 - DENSITY_THRESHOLD
    weight_map_col[static_in_overlap_col] = STATIC_EPSILON

    col_weights = np.sum(weight_map_col, axis=0)
    debug.col_weights_phase3 = col_weights

    c1, c2, col_score = get_best_1d_anchored(col_weights, cx)
    debug.col_range_phase3 = (c1, c2)

    if col_score <= 0:
        debug.failure_phase = 3
        debug.failure_reason = f"No valid refined column range (score={col_score:.2f})"
        return debug

    h_overlap = r2 - r1 + 1
    w_refined = c2 - c1 + 1
    h_viewport = h_overlap + d

    # Convert slice-relative r1 to screen coordinates
    # after_bbox: where the overlap region is in the after image
    # refined_viewport: the full viewport including new content
    if scroll_direction == "down":
        # Scroll-down: r1 is already a screen row (overlap was ga[:limit_h])
        # Viewport starts at overlap and extends DOWN by d
        after_bbox_y = r1
        viewport_y = r1
    else:
        # Scroll-up: r1 is slice index, overlap is at screen row r1 + d
        # Viewport starts d pixels ABOVE the overlap (to include new content at top)
        after_bbox_y = r1 + d
        viewport_y = r1  # Same as overlap_y - d

    debug.after_bbox = (c1, after_bbox_y, w_refined, h_overlap)
    debug.refined_viewport = (c1, viewport_y, w_refined, h_viewport)

    # Final size check
    if h_viewport < MIN_VIEWPORT_HEIGHT or w_refined < MIN_VIEWPORT_WIDTH:
        debug.failure_phase = 3
        debug.failure_reason = f"Refined viewport too small ({w_refined}x{h_viewport})"
        return debug

    return debug


# --- Visualization Functions ---


def render_original_images(ax_before, ax_after, debug_data, cursor_pos):
    """Render the original before/after images with cursor and viewport overlays."""
    ax_before.imshow(debug_data.img_b)
    ax_before.set_title("Before")
    ax_before.axis("off")

    ax_after.imshow(debug_data.img_a)
    ax_after.set_title("After")
    ax_after.axis("off")

    # Draw cursor position
    if cursor_pos:
        cx, cy = cursor_pos
        marker_size = 15
        for ax in [ax_before, ax_after]:
            ax.axhline(y=cy, color="red", linewidth=0.5, alpha=0.7)
            ax.axvline(x=cx, color="red", linewidth=0.5, alpha=0.7)
            ax.plot(cx, cy, "r+", markersize=marker_size, markeredgewidth=2)

    # Draw initial viewport (cyan, dashed)
    if debug_data.initial_viewport:
        x, y, w, h = debug_data.initial_viewport
        for ax in [ax_before, ax_after]:
            rect = Rectangle(
                (x, y),
                w,
                h,
                fill=False,
                edgecolor="cyan",
                linewidth=1.5,
                linestyle="--",
                label="Initial",
            )
            ax.add_patch(rect)

    # Draw refined viewport / overlap region (lime, solid)
    if debug_data.after_bbox:
        x, y, w, h_overlap = debug_data.after_bbox
        d = debug_data.scroll_distance or 0
        scroll_direction = debug_data.params.get("scroll_direction", "down")

        # In After image: overlap is at (x, y)
        rect_after = Rectangle(
            (x, y), w, h_overlap, fill=False, edgecolor="lime", linewidth=2
        )
        ax_after.add_patch(rect_after)

        # In Before image: same content is offset by d
        # For scroll-down: before content is at y + d (content moved up)
        # For scroll-up: before content is at y - d (content moved down)
        before_y = y + d if scroll_direction == "down" else y - d
        rect_before = Rectangle(
            (x, before_y), w, h_overlap, fill=False, edgecolor="lime", linewidth=2
        )
        ax_before.add_patch(rect_before)


def render_change_map(ax, debug_data, cursor_pos):
    """Render the binary change map from Phase 1."""
    if debug_data.change_map is None:
        ax.text(
            0.5, 0.5, "No change map", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Phase 1: Change Map")
        return

    ax.imshow(debug_data.change_map, cmap="RdYlGn", aspect="auto")
    ax.set_title("Phase 1: Change Map\n(Green=changed, Red=unchanged)")

    # Overlay initial viewport
    if debug_data.initial_viewport:
        x, y, w, h = debug_data.initial_viewport
        rect = Rectangle((x, y), w, h, fill=False, edgecolor="cyan", linewidth=2)
        ax.add_patch(rect)

    # Cursor marker
    if cursor_pos:
        ax.plot(cursor_pos[0], cursor_pos[1], "w+", markersize=10, markeredgewidth=2)


def render_row_weights_phase1(ax, debug_data, cursor_pos):
    """Render row weights from Phase 1 with detected range."""
    if debug_data.row_weights_phase1 is None:
        ax.text(
            0.5, 0.5, "No row weights", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Phase 1: Row Weights")
        return

    weights = debug_data.row_weights_phase1
    rows = np.arange(len(weights))

    ax.plot(weights, rows, "b-", linewidth=1)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax.fill_betweenx(rows, 0, weights, where=weights > 0, alpha=0.3, color="green")
    ax.fill_betweenx(rows, 0, weights, where=weights < 0, alpha=0.3, color="red")

    # Highlight detected range
    if debug_data.row_range_phase1:
        r1, r2 = debug_data.row_range_phase1
        ax.axhline(y=r1, color="cyan", linewidth=2, label=f"Range [{r1}, {r2}]")
        ax.axhline(y=r2, color="cyan", linewidth=2)
        ax.axhspan(r1, r2, alpha=0.2, color="cyan")

    # Cursor row
    if cursor_pos:
        ax.axhline(
            y=cursor_pos[1], color="red", linestyle=":", linewidth=1.5, label="Cursor"
        )

    ax.invert_yaxis()
    ax.margins(y=0)
    ax.set_xlabel("Weight")
    ax.set_ylabel("Row")
    ax.set_title("Phase 1: Row Weights")
    ax.legend(loc="upper right", fontsize=7)


def render_col_weights_phase1(ax, debug_data, cursor_pos):
    """Render column weights from Phase 1 with detected range."""
    if debug_data.col_weights_phase1 is None:
        ax.text(
            0.5, 0.5, "No col weights", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Phase 1: Column Weights")
        return

    weights = debug_data.col_weights_phase1
    cols = np.arange(len(weights))

    ax.plot(cols, weights, "b-", linewidth=1)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.fill_between(cols, 0, weights, where=weights > 0, alpha=0.3, color="green")
    ax.fill_between(cols, 0, weights, where=weights < 0, alpha=0.3, color="red")

    # Highlight detected range
    if debug_data.col_range_phase1:
        c1, c2 = debug_data.col_range_phase1
        ax.axvline(x=c1, color="cyan", linewidth=2, label=f"Range [{c1}, {c2}]")
        ax.axvline(x=c2, color="cyan", linewidth=2)
        ax.axvspan(c1, c2, alpha=0.2, color="cyan")

    # Cursor column
    if cursor_pos:
        ax.axvline(
            x=cursor_pos[0], color="red", linestyle=":", linewidth=1.5, label="Cursor"
        )

    ax.margins(x=0)
    ax.set_xlabel("Column")
    ax.set_ylabel("Weight")
    ax.set_title("Phase 1: Column Weights")
    ax.legend(loc="upper right", fontsize=7)


def render_correlation_curve(ax, debug_data):
    """Render the correlation curve from Phase 2."""
    corr = debug_data.correlation_curve

    if corr is None:
        ax.text(
            0.5,
            0.5,
            "No correlation data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Phase 2: Correlation Curve")
        return

    # Plot NCC curve
    x_vals = np.arange(len(corr))
    ax.plot(x_vals, corr, "b-", linewidth=1, label="NCC")
    ax.fill_between(x_vals, 0, corr, alpha=0.3, color="blue")
    ax.set_ylabel("NCC (Normalized Cross-Correlation)")

    # Mark winning distance with triangle at top
    if debug_data.scroll_distance is not None:
        peak_d = debug_data.scroll_distance
        peak_val = corr[peak_d] if peak_d < len(corr) else 0
        # Triangle marker at top of plot area
        ax.plot(peak_d, peak_val, "rv", markersize=10, label=f"d={peak_d}px")

    ax.set_xlabel("Scroll Distance (pixels)")

    # Build title with confidence ratio if available
    title = f"Phase 2: NCC\nDetected: {debug_data.scroll_distance}px"
    if (
        debug_data.peak_value is not None
        and debug_data.second_peak_value is not None
        and debug_data.second_peak_value > 0
    ):
        confidence = debug_data.peak_value / debug_data.second_peak_value
        title += f" (confidence: {confidence:.2f}x)"
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def render_strip_correlation_heatmap(ax, debug_data):
    """Render heatmap of per-strip correlations (strip index vs scroll distance)."""
    strip_corr = debug_data.strip_correlations

    if strip_corr is None:
        ax.text(
            0.5,
            0.5,
            "No strip correlation data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Phase 2: Strip Correlations")
        return

    # Transpose so x-axis is scroll distance, y-axis is strip index
    # Only show the valid range (skip first 20 pixels which are below MIN_SCROLL_DISTANCE)
    min_d = 20
    heatmap_data = strip_corr[:, min_d:]

    im = ax.imshow(
        heatmap_data,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=[min_d, min_d + heatmap_data.shape[1], 0, heatmap_data.shape[0]],
    )

    # Mark detected scroll distance with a small triangle at top
    if debug_data.scroll_distance is not None:
        ax.plot(
            debug_data.scroll_distance,
            heatmap_data.shape[0],
            "rv",
            markersize=8,
            clip_on=False,
        )

    ax.set_xlabel("Scroll Distance (pixels)")
    ax.set_ylabel("Strip Index")
    title = "Phase 2: Per-Strip Correlation Heatmap"
    if debug_data.scroll_distance is not None:
        title += f" (detected: {debug_data.scroll_distance}px)"
    ax.set_title(title)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Correlation / Overlap")


def render_profiles(ax, debug_data):
    """Render the row-summed profiles from Phase 2."""
    if debug_data.profile_before is None or debug_data.profile_after is None:
        ax.text(
            0.5,
            0.5,
            "No profile data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Phase 2: Row Profiles")
        return

    rows = np.arange(len(debug_data.profile_before))

    ax.plot(
        debug_data.profile_before, rows, "b-", linewidth=1, label="Before", alpha=0.7
    )
    ax.plot(debug_data.profile_after, rows, "r-", linewidth=1, label="After", alpha=0.7)

    ax.invert_yaxis()
    ax.set_xlabel("Sum")
    ax.set_ylabel("Row")
    ax.set_title("Phase 2: Row-Summed Profiles")
    ax.legend(loc="upper right")


def render_match_map(ax, debug_data, cursor_pos):
    """Render the match map from Phase 3 with 3 colors for weight categories."""
    if debug_data.match_map is None:
        ax.text(
            0.5, 0.5, "No match map", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Phase 3: Match Map")
        return

    overlap_match = debug_data.match_map
    static_match = debug_data.static_match_map

    # Compute three categories (same logic as weight calculation)
    # dynamic_match: matches when shifted but NOT statically (scrolled content)
    # static_in_overlap: matches both shifted AND statically (static content)
    # mismatch: doesn't match when shifted
    if static_match is not None:
        dynamic_match = overlap_match & ~static_match
        static_in_overlap = overlap_match & static_match
    else:
        # Fallback if static_match not available
        dynamic_match = overlap_match
        static_in_overlap = np.zeros_like(overlap_match)

    mismatch = ~overlap_match

    # Three colors corresponding to the three weights:
    # Green (0, 200, 0) for dynamic_match (positive weight - scrolled content)
    # Yellow (200, 200, 0) for static_in_overlap (near-zero weight - static content)
    # Red (200, 0, 0) for mismatch (negative weight - non-viewport)
    h, w = overlap_match.shape
    match_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    match_rgb[dynamic_match] = [0, 200, 0]
    match_rgb[static_in_overlap] = [200, 200, 0]
    match_rgb[mismatch] = [200, 0, 0]

    ax.imshow(match_rgb, aspect="auto")

    # Overlay refined viewport (relative to match_map coordinates)
    if debug_data.after_bbox:
        x, y, w, h = debug_data.after_bbox
        init_x = debug_data.initial_viewport[0] if debug_data.initial_viewport else 0
        scroll_direction = debug_data.params.get("scroll_direction", "down")
        scroll_distance = debug_data.scroll_distance or 0

        # Convert to match_map coordinates
        # x: relative to c1_init (initial viewport x)
        x_rel = x - init_x
        # y: match_map starts at row 0 (scroll-down) or row d (scroll-up) of after image
        y_rel = y if scroll_direction == "down" else y - scroll_distance

        rect = Rectangle(
            (x_rel, y_rel), w, h, fill=False, edgecolor="cyan", linewidth=2
        )
        ax.add_patch(rect)

    # Calculate percentages for each category
    total = overlap_match.size
    dynamic_pct = 100 * np.sum(dynamic_match) / total if total > 0 else 0
    static_pct = 100 * np.sum(static_in_overlap) / total if total > 0 else 0
    mismatch_pct = 100 * np.sum(mismatch) / total if total > 0 else 0

    ax.set_title(
        f"Phase 3: Match Map\n"
        f"Dynamic(+): {dynamic_pct:.0f}%  Static(0): {static_pct:.0f}%  "
        f"Mismatch(-): {mismatch_pct:.0f}%"
    )
    ax.axis("off")


def render_row_weights_phase3(ax, debug_data, cursor_pos):
    """Render row weights from Phase 3 with detected range."""
    if debug_data.row_weights_phase3 is None:
        ax.text(
            0.5, 0.5, "No row weights", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Phase 3: Row Weights")
        return

    weights = debug_data.row_weights_phase3
    rows = np.arange(len(weights))

    ax.plot(weights, rows, "b-", linewidth=1)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax.fill_betweenx(rows, 0, weights, where=weights > 0, alpha=0.3, color="green")
    ax.fill_betweenx(rows, 0, weights, where=weights < 0, alpha=0.3, color="red")

    # Highlight detected range
    if debug_data.row_range_phase3:
        r1, r2 = debug_data.row_range_phase3
        ax.axhline(y=r1, color="lime", linewidth=2, label=f"Range [{r1}, {r2}]")
        ax.axhline(y=r2, color="lime", linewidth=2)
        ax.axhspan(r1, r2, alpha=0.2, color="lime")

    ax.invert_yaxis()
    ax.margins(y=0)
    ax.set_xlabel("Weight")
    ax.set_ylabel("Row")
    ax.set_title("Phase 3: Refined Row Weights")
    ax.legend(loc="upper right", fontsize=7)


def render_col_weights_phase3(ax, debug_data, cursor_pos):
    """Render column weights from Phase 3 with detected range."""
    if debug_data.col_weights_phase3 is None:
        ax.text(
            0.5, 0.5, "No col weights", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Phase 3: Column Weights")
        return

    weights = debug_data.col_weights_phase3
    cols = np.arange(len(weights))

    ax.plot(cols, weights, "b-", linewidth=1)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.fill_between(cols, 0, weights, where=weights > 0, alpha=0.3, color="green")
    ax.fill_between(cols, 0, weights, where=weights < 0, alpha=0.3, color="red")

    # Highlight detected range
    if debug_data.col_range_phase3:
        c1, c2 = debug_data.col_range_phase3
        ax.axvline(x=c1, color="lime", linewidth=2, label=f"Range [{c1}, {c2}]")
        ax.axvline(x=c2, color="lime", linewidth=2)
        ax.axvspan(c1, c2, alpha=0.2, color="lime")

    ax.margins(x=0)
    ax.set_xlabel("Column (relative)")
    ax.set_ylabel("Weight")
    ax.set_title("Phase 3: Refined Column Weights")
    ax.legend(loc="upper right", fontsize=7)


# --- Candidate Visualization Functions ---


def compute_correlation_2d(debug_data, candidate_distance):
    """Compute per-pixel gradient correlation for a candidate distance."""
    d = candidate_distance
    grad_b = debug_data.grad_b
    grad_a = debug_data.grad_a
    scroll_direction = debug_data.params.get("scroll_direction", "down")

    if grad_b is None or grad_a is None:
        return None

    # Use viewport bounds if available
    if debug_data.initial_viewport:
        vx, vy, vw, vh = debug_data.initial_viewport
        grad_b = grad_b[vy : vy + vh - 1, vx : vx + vw]
        grad_a = grad_a[vy : vy + vh - 1, vx : vx + vw]

    H = grad_b.shape[0]

    if d <= 0 or d >= H:
        return None

    limit_h = H - d
    # Scroll-down: compare before[d:] with after[:limit_h]
    # Scroll-up: compare before[:limit_h] with after[d:]
    if scroll_direction == "down":
        b_region = grad_b[d:, :]
        a_region = grad_a[:limit_h, :]
    else:
        b_region = grad_b[:limit_h, :]
        a_region = grad_a[d:, :]

    return np.abs(b_region) * np.abs(a_region)


def compute_pixel_match(debug_data, candidate_distance):
    """Compute per-pixel match/mismatch for a candidate distance."""
    d = candidate_distance
    gray_b = debug_data.gray_b
    gray_a = debug_data.gray_a
    tolerance = debug_data.params["pixel_match_tolerance"]
    scroll_direction = debug_data.params.get("scroll_direction", "down")

    if debug_data.initial_viewport:
        vx, vy, vw, vh = debug_data.initial_viewport
        gray_b = gray_b[vy : vy + vh, vx : vx + vw]
        gray_a = gray_a[vy : vy + vh, vx : vx + vw]

    H = gray_b.shape[0]

    if d <= 0 or d >= H:
        return None, 0, 0

    limit_h = H - d
    # Scroll-down: compare before[d:] with after[:limit_h]
    # Scroll-up: compare before[:limit_h] with after[d:]
    if scroll_direction == "down":
        b_region = gray_b[d:, :]
        a_region = gray_a[:limit_h, :]
    else:
        b_region = gray_b[:limit_h, :]
        a_region = gray_a[d:, :]

    diff = np.abs(b_region - a_region)
    match_mask = diff < tolerance

    h, w = match_mask.shape
    match_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    match_rgb[match_mask] = [0, 200, 0]
    match_rgb[~match_mask] = [200, 0, 0]

    match_count = np.sum(match_mask)
    mismatch_count = np.sum(~match_mask)

    return match_rgb, match_count, mismatch_count


def render_candidate_correlation(ax, debug_data, candidate_distance, is_detected=False):
    """Render per-pixel gradient correlation for a candidate distance."""
    d = candidate_distance

    corr_2d = compute_correlation_2d(debug_data, d)
    if corr_2d is None:
        ax.text(
            0.5,
            0.5,
            f"Invalid distance: {d}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(f"d={d}px (invalid)")
        ax.axis("off")
        return

    corr_log = np.log1p(corr_2d)
    vmax = np.percentile(corr_log, 99)
    ax.imshow(corr_log, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)

    total_corr = np.sum(corr_2d)

    title = f"Gradient Corr d={d}px"
    if is_detected:
        title += " (DETECTED)"
    title += f"\nTotal: {total_corr:.0f}"
    ax.set_title(title)
    ax.axis("off")


def render_candidate_match(ax, debug_data, candidate_distance, is_detected=False):
    """Render per-pixel match/mismatch for a candidate distance."""
    d = candidate_distance

    match_rgb, match_count, mismatch_count = compute_pixel_match(debug_data, d)
    if match_rgb is None:
        ax.text(
            0.5,
            0.5,
            f"Invalid distance: {d}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(f"d={d}px (invalid)")
        ax.axis("off")
        return

    ax.imshow(match_rgb, aspect="auto")

    total = match_count + mismatch_count
    pct = 100 * match_count / total if total > 0 else 0
    title = f"Pixel Match d={d}px"
    if is_detected:
        title += " (DETECTED)"
    title += f"\nMatch: {pct:.1f}%"
    ax.set_title(title)
    ax.axis("off")


def print_summary(debug_data, cursor_pos, output_path, params):
    """Print algorithm results to stdout."""
    print("Scroll Detection Results (Three-Phase Algorithm)")
    print("=" * 50)

    if debug_data.failure_phase:
        print(f"FAILED at Phase {debug_data.failure_phase}")
        print(f"Reason: {debug_data.failure_reason}")
        print()
    else:
        print(f"Detected scroll distance: {debug_data.scroll_distance} pixels")
        print()

    # Phase 1 results
    print("Phase 1: Initial Viewport Estimation")
    print("-" * 40)
    if debug_data.initial_viewport:
        x, y, w, h = debug_data.initial_viewport
        print(f"  Initial viewport: ({x}, {y}) {w}x{h}")
        print(f"  Row range: {debug_data.row_range_phase1}")
        print(f"  Col range: {debug_data.col_range_phase1}")
    else:
        print("  Failed to estimate initial viewport")
    print()

    # Phase 2 results
    print("Phase 2: Scroll Distance Estimation")
    print("-" * 40)
    if debug_data.scroll_distance is not None:
        print(f"  Scroll distance: {debug_data.scroll_distance}px")
        print(f"  Peak correlation: {debug_data.peak_value:.2f}")
        if debug_data.second_peak_value is not None:
            ratio = debug_data.peak_value / debug_data.second_peak_value
            print(
                f"  Second peak: {debug_data.second_peak_idx}px ({debug_data.second_peak_value:.2f})"
            )
            print(f"  Confidence ratio: {ratio:.2f}x")
    else:
        print("  Failed to estimate scroll distance")
    print()

    # Phase 3 results
    print("Phase 3: Viewport Refinement")
    print("-" * 40)
    if debug_data.refined_viewport:
        x, y, w, h = debug_data.refined_viewport
        print(f"  Refined viewport: ({x}, {y}) {w}x{h}")
        print(f"  Row range: {debug_data.row_range_phase3}")
        print(f"  Col range: {debug_data.col_range_phase3}")
        if debug_data.after_bbox:
            ax, ay, aw, ah = debug_data.after_bbox
            print(f"  After bbox: ({ax}, {ay}) {aw}x{ah}")
    else:
        print("  Failed to refine viewport")
    print()

    if cursor_pos:
        print(f"Cursor position: ({cursor_pos[0]}, {cursor_pos[1]})")

    print()
    print("Parameters used:")
    for key, val in params.items():
        print(f"  {key}: {val}")

    print(f"\nDebug image saved to: {output_path}")


@click.command()
@click.argument("before_image", type=click.Path(exists=True), required=False)
@click.argument("after_image", type=click.Path(exists=True), required=False)
@click.option(
    "--from-json",
    type=click.Path(exists=True),
    default=None,
    help="Load from scroll_*.json file (derives before/after images and cursor)",
)
@click.option("--output", "-o", default=None, help="Output debug image path")
@click.option(
    "--cursor-pos",
    nargs=2,
    type=int,
    default=None,
    help="Cursor position as X Y coordinates",
)
@click.option(
    "--check-offset",
    "-c",
    multiple=True,
    type=int,
    default=(),
    help="Additional offset(s) to visualize (can be specified multiple times)",
)
# Algorithm tuning parameters
@click.option(
    "--min-scroll-distance",
    default=DEFAULT_MIN_SCROLL_DISTANCE,
    help="Minimum scroll distance to consider",
)
@click.option(
    "--min-viewport-height",
    default=DEFAULT_MIN_VIEWPORT_HEIGHT,
    help="Minimum viewport height",
)
@click.option(
    "--min-viewport-width",
    default=DEFAULT_MIN_VIEWPORT_WIDTH,
    help="Minimum viewport width",
)
@click.option(
    "--pixel-match-tolerance",
    default=DEFAULT_PIXEL_MATCH_TOLERANCE,
    help="Pixel difference tolerance for match",
)
@click.option(
    "--change-threshold-ratio",
    default=DEFAULT_CHANGE_THRESHOLD_RATIO,
    help="Threshold ratio for Phase 1 weights (lower = bridge more gaps)",
)
@click.option(
    "--density-threshold",
    default=DEFAULT_DENSITY_THRESHOLD,
    help="Density threshold for Phase 3 weights",
)
@click.option(
    "--num-strips",
    default=DEFAULT_NUM_STRIPS,
    help="Number of vertical strips for correlation",
)
@click.option(
    "--viewport",
    nargs=4,
    type=int,
    default=None,
    help="Override viewport as X Y WIDTH HEIGHT (skips Phase 1)",
)
@click.option(
    "--direction",
    type=click.Choice(["down", "up"]),
    default=None,
    help="Scroll direction (default: from JSON or 'down')",
)
def main(
    before_image,
    after_image,
    from_json,
    output,
    cursor_pos,
    check_offset,
    min_scroll_distance,
    min_viewport_height,
    min_viewport_width,
    pixel_match_tolerance,
    change_threshold_ratio,
    density_threshold,
    num_strips,
    viewport,
    direction,
):
    """Generate debug visualization for scroll detection (Three-Phase Algorithm).

    Usage:
        uv run debug_scroll.py --from-json data/scroll_*.json
        uv run debug_scroll.py before.png after.png [--cursor-pos X Y]
        uv run debug_scroll.py --from-json data/scroll_*.json --check-offset 880
    """

    # Determine input source
    json_existing_viewport = None
    json_scroll_direction = "down"
    if from_json:
        (
            before_image,
            after_image,
            cursor,
            json_existing_viewport,
            json_scroll_direction,
        ) = parse_from_json(from_json)
        if cursor_pos:
            cursor = tuple(cursor_pos)
        if output is None:
            base = from_json.rsplit(".json", 1)[0]
            output = base + "_debug.png"
    elif before_image and after_image:
        cursor = tuple(cursor_pos) if cursor_pos else None
        if output is None:
            output = "debug_output.png"
    else:
        raise click.UsageError(
            "Either provide --from-json or both BEFORE_IMAGE and AFTER_IMAGE"
        )

    # CLI --direction overrides JSON, which defaults to "down"
    effective_direction = direction if direction else json_scroll_direction

    # Load images
    img_b = np.array(Image.open(before_image))
    img_a = np.array(Image.open(after_image))

    # Build params dict (CLI --viewport overrides JSON existing_viewport)
    effective_viewport = tuple(viewport) if viewport else json_existing_viewport
    params = {
        "min_scroll_distance": min_scroll_distance,
        "min_viewport_height": min_viewport_height,
        "min_viewport_width": min_viewport_width,
        "pixel_match_tolerance": pixel_match_tolerance,
        "change_threshold_ratio": change_threshold_ratio,
        "density_threshold": density_threshold,
        "num_strips": num_strips,
        "existing_viewport": effective_viewport,
        "scroll_direction": effective_direction,
    }

    # Run instrumented detection
    debug_data = detect_scroll_debug(img_b, img_a, cursor, params)

    # Build list of candidate distances to visualize
    candidates = []
    detected_d = debug_data.scroll_distance
    if detected_d is not None and detected_d > 0:
        candidates.append(detected_d)

    for offset in check_offset:
        if offset not in candidates and offset > 0:
            candidates.append(offset)

    if not candidates and check_offset:
        candidates = list(check_offset)

    has_candidates = len(candidates) > 0
    n_cols = max(3, len(candidates)) if has_candidates else 3

    fig_height = 24 if has_candidates else 16
    fig_width = max(18, 6 * n_cols)

    fig = plt.figure(figsize=(fig_width, fig_height))

    if has_candidates:
        gs_main = GridSpec(
            2,
            1,
            figure=fig,
            height_ratios=[2, 1],
            hspace=0.1,
            top=0.95,
            bottom=0.04,
            left=0.05,
            right=0.97,
        )
        gs_top = GridSpecFromSubplotSpec(
            4, 3, subplot_spec=gs_main[0], hspace=0.35, wspace=0.2
        )
        gs_bottom = GridSpecFromSubplotSpec(
            2, len(candidates), subplot_spec=gs_main[1], hspace=0.15, wspace=0.2
        )
    else:
        gs_top = GridSpec(
            4,
            3,
            figure=fig,
            hspace=0.35,
            wspace=0.2,
            top=0.95,
            bottom=0.04,
            left=0.05,
            right=0.97,
        )
        gs_bottom = None

    # Row 0: Summary - Original images
    ax_before = fig.add_subplot(gs_top[0, 0])
    ax_after = fig.add_subplot(gs_top[0, 1])

    render_original_images(ax_before, ax_after, debug_data, cursor)

    # Empty third column - could add summary text later
    ax_summary = fig.add_subplot(gs_top[0, 2])
    ax_summary.axis("off")

    # === Phase 1: Initial Viewport Estimation ===
    # Row 1: Change map + Phase 1 weights
    ax_change = fig.add_subplot(gs_top[1, 0])
    ax_row_p1 = fig.add_subplot(gs_top[1, 1])
    ax_col_p1 = fig.add_subplot(gs_top[1, 2])

    render_change_map(ax_change, debug_data, cursor)
    render_row_weights_phase1(ax_row_p1, debug_data, cursor)
    render_col_weights_phase1(ax_col_p1, debug_data, cursor)

    # === Phase 2: Scroll Distance Estimation ===
    # Row 2: Correlation curve + Heatmap
    ax_corr = fig.add_subplot(gs_top[2, 0])
    ax_heatmap = fig.add_subplot(gs_top[2, 1:])

    render_correlation_curve(ax_corr, debug_data)
    render_strip_correlation_heatmap(ax_heatmap, debug_data)

    # === Phase 3: Viewport Refinement ===
    # Row 3: Match map + Phase 3 weights
    ax_match = fig.add_subplot(gs_top[3, 0])
    ax_row_p3 = fig.add_subplot(gs_top[3, 1])
    ax_col_p3 = fig.add_subplot(gs_top[3, 2])

    render_match_map(ax_match, debug_data, cursor)
    render_row_weights_phase3(ax_row_p3, debug_data, cursor)
    render_col_weights_phase3(ax_col_p3, debug_data, cursor)

    # Rows 4-5: Per-candidate gradient correlation and pixel match
    if has_candidates and gs_bottom is not None:
        for i, cand_d in enumerate(candidates):
            is_detected = cand_d == detected_d

            ax_grad = fig.add_subplot(gs_bottom[0, i])
            render_candidate_correlation(
                ax_grad, debug_data, cand_d, is_detected=is_detected
            )

            ax_pix = fig.add_subplot(gs_bottom[1, i])
            render_candidate_match(ax_pix, debug_data, cand_d, is_detected=is_detected)

    # Title with status
    status = (
        "SUCCESS"
        if debug_data.scroll_distance is not None and debug_data.failure_phase is None
        else "FAILED"
    )
    if debug_data.failure_phase:
        status += f" (Phase {debug_data.failure_phase}: {debug_data.failure_reason})"

    title = f"Scroll Detection Debug: {status}\n"
    title += f"Before: {before_image} | After: {after_image}"
    if cursor:
        title += f" | Cursor: ({cursor[0]}, {cursor[1]})"
    if debug_data.scroll_distance is not None:
        title += f" | Distance: {debug_data.scroll_distance}px"
    fig.suptitle(title, fontsize=12)

    plt.savefig(output, dpi=200)
    plt.close()

    print_summary(debug_data, cursor, output, params)

    if len(candidates) > 1:
        print("\nCandidate Comparison:")
        print("-" * 40)
        for cand_d in candidates:
            _, match_count, mismatch_count = compute_pixel_match(debug_data, cand_d)
            total = match_count + mismatch_count
            pct = 100 * match_count / total if total > 0 else 0
            marker = " (DETECTED)" if cand_d == detected_d else ""
            print(f"  d={cand_d}px: {pct:.1f}% pixel match{marker}")


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
