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

    # Phase 2: Scroll distance estimation (strip voting)
    profile_before: np.ndarray | None = None  # Row-summed profile (full width, for viz)
    profile_after: np.ndarray | None = None
    correlation: np.ndarray | None = None  # NCC curve (for viz)
    votes: dict = field(default_factory=dict)  # Vote counts per distance
    strip_results: list = field(default_factory=list)  # Per-strip voting info
    peak_idx: int | None = None
    scroll_distance: int | None = None

    # Phase 3: Viewport refinement
    match_map: np.ndarray | None = None  # Aligned regions match
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
    Parse a scroll_*.json file and derive before/after image paths and cursor position.
    """
    if not os.path.exists(json_path):
        raise click.BadParameter(f"JSON file not found: {json_path}")

    base = json_path.rsplit(".json", 1)[0]
    before_path = base + ".png"
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

    return before_path, after_path, cursor_pos


def detect_scroll_debug(img_before, img_after, cursor_pos, params):
    """
    Instrumented version of detect_scroll that returns debug data.

    This mirrors the algorithm with instrumentation to collect
    intermediate values for visualization.
    """
    # Unpack parameters
    MIN_SCROLL_DISTANCE = params.get("min_scroll_distance", 20)
    MIN_VIEWPORT_HEIGHT = params.get("min_viewport_height", 150)
    MIN_VIEWPORT_WIDTH = params.get("min_viewport_width", 300)
    PIXEL_MATCH_TOLERANCE = params.get("pixel_match_tolerance", 15)
    CHANGE_THRESHOLD_RATIO = params.get("change_threshold_ratio", 0.15)
    DENSITY_THRESHOLD = params.get("density_threshold", 0.85)
    ROW_SEARCH_SLICE_WIDTH = params.get("row_search_slice_width", 100)

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

    # === Phase 2: Scroll Distance Estimation (Strip-based NCC Voting) ===
    NUM_STRIPS = params.get("num_strips", 8)
    MIN_STRIP_GRADIENT = params.get("min_strip_gradient", 5.0)
    MIN_NCC_SCORE = params.get("min_ncc_score", 0.3)
    NCC_EXPONENT = params.get("ncc_exponent", 2)

    x, y, w, h = debug.initial_viewport

    before_crop = gb[y : y + h, x : x + w]
    after_crop = ga[y : y + h, x : x + w]

    # Compute vertical gradients (row-to-row differences)
    grad_before = np.diff(before_crop, axis=0)
    grad_after = np.diff(after_crop, axis=0)

    # Store full-width profiles for visualization
    profile_before = np.sum(np.abs(grad_before), axis=1).astype(float)
    profile_after = np.sum(np.abs(grad_after), axis=1).astype(float)
    debug.profile_before = profile_before
    debug.profile_after = profile_after

    # Use gradient height for correlation (h - 1 due to diff)
    h_grad = grad_before.shape[0]

    # Constraint: overlap = h_grad - d >= MIN_VIEWPORT_HEIGHT
    max_d = h_grad - MIN_VIEWPORT_HEIGHT
    if max_d <= MIN_SCROLL_DISTANCE:
        debug.failure_phase = 2
        debug.failure_reason = (
            "Viewport too small for minimum scroll distance with required overlap"
        )
        return debug

    # Strip-based voting
    votes = {}
    strip_results = []
    eps = 1e-9
    strip_width = w // NUM_STRIPS

    for strip_idx in range(NUM_STRIPS):
        x_start = strip_idx * strip_width
        x_end = x_start + strip_width if strip_idx < NUM_STRIPS - 1 else w

        # Extract strip gradients
        strip_grad_b = grad_before[:, x_start:x_end]
        strip_grad_a = grad_after[:, x_start:x_end]

        mean_grad = np.mean(np.abs(strip_grad_b))
        strip_info = {
            "idx": strip_idx,
            "x_start": x_start,
            "x_end": x_end,
            "mean_grad": mean_grad,
            "voted": False,
            "best_d": None,
            "best_ncc": None,
        }

        # Skip strips with low gradient activity (uniform regions)
        if mean_grad < MIN_STRIP_GRADIENT:
            strip_results.append(strip_info)
            continue

        # Create 1D profiles for this strip
        profile_b = np.sum(np.abs(strip_grad_b), axis=1).astype(float)
        profile_a = np.sum(np.abs(strip_grad_a), axis=1).astype(float)

        # Find best NCC for this strip
        best_ncc = -1
        best_d = MIN_SCROLL_DISTANCE

        for d_val in range(MIN_SCROLL_DISTANCE, max_d + 1):
            overlap_h = h_grad - d_val
            b_seg = profile_b[d_val:]
            a_seg = profile_a[:overlap_h]

            dot_product = np.dot(b_seg, a_seg)
            norm_b = np.linalg.norm(b_seg)
            norm_a = np.linalg.norm(a_seg)

            ncc = dot_product / (norm_b * norm_a + eps)

            if ncc > best_ncc:
                best_ncc = ncc
                best_d = d_val

        strip_info["best_d"] = best_d
        strip_info["best_ncc"] = best_ncc

        # Vote for best distance (weighted by NCC score)
        if best_ncc >= MIN_NCC_SCORE:
            strip_info["voted"] = True
            vote_weight = best_ncc**NCC_EXPONENT
            votes[best_d] = votes.get(best_d, 0) + vote_weight

        strip_results.append(strip_info)

    debug.votes = votes
    debug.strip_results = strip_results

    # Find winning scroll distance
    if not votes:
        debug.failure_phase = 2
        debug.failure_reason = "No strips voted for any scroll distance"
        return debug

    # Get best voted distance
    best_d = max(votes.keys(), key=lambda d: votes[d])
    debug.peak_idx = best_d
    debug.scroll_distance = best_d

    # Also compute full-width NCC for visualization
    ncc_full = np.zeros(max_d + 1)
    for d_val in range(MIN_SCROLL_DISTANCE, max_d + 1):
        overlap_h = h_grad - d_val
        b_seg = profile_before[d_val:]
        a_seg = profile_after[:overlap_h]
        dot_product = np.dot(b_seg, a_seg)
        norm_b = np.linalg.norm(b_seg)
        norm_a = np.linalg.norm(a_seg)
        ncc_full[d_val] = dot_product / (norm_b * norm_a + eps)
    debug.correlation = ncc_full

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

    after_region = ga[:limit_h, c1_init:c2_init]
    before_region = gb[d:, c1_init:c2_init]

    diff_aligned = np.abs(after_region.astype(float) - before_region.astype(float))
    match_map = diff_aligned < PIXEL_MATCH_TOLERANCE
    debug.match_map = match_map

    _, region_w = match_map.shape

    target_y_max = min(cy, limit_h - 1)
    target_y_min = max(0, cy - d)
    if target_y_min > target_y_max:
        target_y_min = target_y_max

    weights = np.where(match_map, 1.0 - DENSITY_THRESHOLD, -DENSITY_THRESHOLD - 1e-5)

    half_slice = ROW_SEARCH_SLICE_WIDTH // 2
    cursor_in_region = cx - c1_init
    slice_left = max(0, cursor_in_region - half_slice)
    slice_right = min(region_w, cursor_in_region + half_slice)

    row_weights_p3 = np.sum(weights[:, slice_left:slice_right], axis=1)
    debug.row_weights_phase3 = row_weights_p3

    r1_p3, r2_p3, r_score_p3 = get_best_1d_range_constrained(
        row_weights_p3, target_y_min, target_y_max
    )
    debug.row_range_phase3 = (r1_p3, r2_p3)

    if r_score_p3 <= 0:
        debug.failure_phase = 3
        debug.failure_reason = f"No valid refined row range (score={r_score_p3:.2f})"
        return debug

    col_weights_p3 = np.sum(weights[r1_p3 : r2_p3 + 1, :], axis=0)
    debug.col_weights_phase3 = col_weights_p3

    c1_rel, c2_rel, c_score_p3 = get_best_1d_anchored(col_weights_p3, cursor_in_region)
    debug.col_range_phase3 = (c1_rel, c2_rel)

    if c_score_p3 <= 0:
        debug.failure_phase = 3
        debug.failure_reason = f"No valid refined column range (score={c_score_p3:.2f})"
        return debug

    c1_final = c1_init + c1_rel
    c2_final = c1_init + c2_rel

    h_overlap = r2_p3 - r1_p3 + 1
    w_refined = c2_final - c1_final + 1
    h_viewport = h_overlap + d

    debug.after_bbox = (c1_final, r1_p3, w_refined, h_overlap)
    debug.refined_viewport = (c1_final, r1_p3, w_refined, h_viewport)

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

        # In After image: overlap is at (x, y)
        rect_after = Rectangle(
            (x, y), w, h_overlap, fill=False, edgecolor="lime", linewidth=2
        )
        ax_after.add_patch(rect_after)

        # In Before image: same content is at (x, y + d)
        rect_before = Rectangle(
            (x, y + d), w, h_overlap, fill=False, edgecolor="lime", linewidth=2
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

    ax.set_xlabel("Column")
    ax.set_ylabel("Weight")
    ax.set_title("Phase 1: Column Weights")
    ax.legend(loc="upper right", fontsize=7)


def render_correlation_curve(ax, debug_data):
    """Render the vote histogram and NCC curve from Phase 2."""
    votes = debug_data.votes
    ncc = debug_data.correlation

    if not votes and ncc is None:
        ax.text(
            0.5,
            0.5,
            "No correlation data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Phase 2: Strip Voting")
        return

    # Create secondary axis for NCC curve
    ax2 = ax.twinx()

    # Plot vote histogram
    if votes:
        distances = sorted(votes.keys())
        vote_counts = [votes[d] for d in distances]
        ax.bar(distances, vote_counts, width=5, alpha=0.6, color="green", label="Votes")
        ax.set_ylabel("Vote Weight", color="green")
        ax.tick_params(axis="y", labelcolor="green")

    # Plot NCC curve on secondary axis
    if ncc is not None:
        x_vals = np.arange(len(ncc))
        ax2.plot(x_vals, ncc, "b-", linewidth=1, alpha=0.5, label="Full-width NCC")
        ax2.set_ylabel("NCC", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")

    # Mark winning distance
    if debug_data.scroll_distance is not None:
        peak_d = debug_data.scroll_distance
        ax.axvline(
            x=peak_d, color="red", linestyle="--", linewidth=2, label=f"d={peak_d}px"
        )

    ax.set_xlabel("Scroll Distance (pixels)")

    # Count voting strips
    n_voted = sum(1 for s in debug_data.strip_results if s.get("voted", False))
    n_total = len(debug_data.strip_results)
    ax.set_title(
        f"Phase 2: Strip Voting ({n_voted}/{n_total} strips voted)\nDetected: {debug_data.scroll_distance}px"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


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
    """Render the match map from Phase 3."""
    if debug_data.match_map is None:
        ax.text(
            0.5, 0.5, "No match map", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title("Phase 3: Match Map")
        return

    # Green for match, red for mismatch
    h, w = debug_data.match_map.shape
    match_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    match_rgb[debug_data.match_map] = [0, 200, 0]
    match_rgb[~debug_data.match_map] = [200, 0, 0]

    ax.imshow(match_rgb, aspect="auto")

    # Overlay refined viewport (relative to match_map coordinates)
    if debug_data.after_bbox:
        x, y, w, h = debug_data.after_bbox
        init_x = debug_data.initial_viewport[0] if debug_data.initial_viewport else 0
        # Convert to match_map coordinates (relative to c1_init)
        x_rel = x - init_x
        rect = Rectangle((x_rel, y), w, h, fill=False, edgecolor="lime", linewidth=2)
        ax.add_patch(rect)

    # Calculate match percentage
    total = debug_data.match_map.size
    match_count = np.sum(debug_data.match_map)
    pct = 100 * match_count / total if total > 0 else 0

    ax.set_title(f"Phase 3: Match Map\nMatch: {pct:.1f}%")
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
    b_region = grad_b[d:, :]
    a_region = grad_a[:limit_h, :]

    return np.abs(b_region) * np.abs(a_region)


def compute_pixel_match(debug_data, candidate_distance):
    """Compute per-pixel match/mismatch for a candidate distance."""
    d = candidate_distance
    gray_b = debug_data.gray_b
    gray_a = debug_data.gray_a
    tolerance = debug_data.params.get("pixel_match_tolerance", 15)

    if debug_data.initial_viewport:
        vx, vy, vw, vh = debug_data.initial_viewport
        gray_b = gray_b[vy : vy + vh, vx : vx + vw]
        gray_a = gray_a[vy : vy + vh, vx : vx + vw]

    H = gray_b.shape[0]

    if d <= 0 or d >= H:
        return None, 0, 0

    limit_h = H - d
    b_region = gray_b[d:, :]
    a_region = gray_a[:limit_h, :]

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
        print(f"  Correlation peak index: {debug_data.peak_idx}")
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
    "--min-scroll-distance", default=20, help="Minimum scroll distance to consider"
)
@click.option("--min-viewport-height", default=150, help="Minimum viewport height")
@click.option("--min-viewport-width", default=300, help="Minimum viewport width")
@click.option(
    "--pixel-match-tolerance", default=15, help="Pixel difference tolerance for match"
)
@click.option(
    "--change-threshold-ratio",
    default=0.15,
    help="Threshold ratio for Phase 1 weights (lower = bridge more gaps)",
)
@click.option(
    "--density-threshold", default=0.85, help="Density threshold for Phase 3 weights"
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
):
    """Generate debug visualization for scroll detection (Three-Phase Algorithm).

    Usage:
        uv run debug_scroll.py --from-json data/scroll_*.json
        uv run debug_scroll.py before.png after.png [--cursor-pos X Y]
        uv run debug_scroll.py --from-json data/scroll_*.json --check-offset 880
    """

    # Determine input source
    if from_json:
        before_image, after_image, cursor = parse_from_json(from_json)
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

    # Load images
    img_b = np.array(Image.open(before_image))
    img_a = np.array(Image.open(after_image))

    # Build params dict
    params = {
        "min_scroll_distance": min_scroll_distance,
        "min_viewport_height": min_viewport_height,
        "min_viewport_width": min_viewport_width,
        "pixel_match_tolerance": pixel_match_tolerance,
        "change_threshold_ratio": change_threshold_ratio,
        "density_threshold": density_threshold,
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

    fig_height = 30 if has_candidates else 18
    fig_width = max(18, 6 * n_cols)

    fig = plt.figure(figsize=(fig_width, fig_height))

    if has_candidates:
        gs_main = GridSpec(2, 1, figure=fig, height_ratios=[3, 2], hspace=0.3)
        gs_top = GridSpecFromSubplotSpec(
            3, 3, subplot_spec=gs_main[0], hspace=0.3, wspace=0.3
        )
        gs_bottom = GridSpecFromSubplotSpec(
            2, len(candidates), subplot_spec=gs_main[1], hspace=0.3, wspace=0.3
        )
    else:
        gs_top = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        gs_bottom = None

    # Row 0: Original images + Change map
    ax_before = fig.add_subplot(gs_top[0, 0])
    ax_after = fig.add_subplot(gs_top[0, 1])
    ax_change = fig.add_subplot(gs_top[0, 2])

    render_original_images(ax_before, ax_after, debug_data, cursor)
    render_change_map(ax_change, debug_data, cursor)

    # Row 1: Phase 1 weights + Phase 2 correlation
    ax_row_p1 = fig.add_subplot(gs_top[1, 0])
    ax_col_p1 = fig.add_subplot(gs_top[1, 1])
    ax_corr = fig.add_subplot(gs_top[1, 2])

    render_row_weights_phase1(ax_row_p1, debug_data, cursor)
    render_col_weights_phase1(ax_col_p1, debug_data, cursor)
    render_correlation_curve(ax_corr, debug_data)

    # Row 2: Phase 3 match map + Phase 3 weights
    ax_match = fig.add_subplot(gs_top[2, 0])
    ax_row_p3 = fig.add_subplot(gs_top[2, 1])
    ax_col_p3 = fig.add_subplot(gs_top[2, 2])

    render_match_map(ax_match, debug_data, cursor)
    render_row_weights_phase3(ax_row_p3, debug_data, cursor)
    render_col_weights_phase3(ax_col_p3, debug_data, cursor)

    # Rows 3-4: Per-candidate gradient correlation and pixel match
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

    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight")
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
