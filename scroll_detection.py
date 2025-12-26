"""Scroll detection algorithm for detecting vertical scrolling in screenshots.

Uses a three-phase approach:
1. Initial viewport estimation (find changed pixels via binary diff + Kadane)
2. Scroll distance calculation (strip-based NCC voting on gradient profiles)
3. Viewport refinement (find matching pixels to refine bounds)

This module contains only the pure algorithmic functions with no Talon dependencies.
"""

import logging
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Immutable bounding box with x, y, width, height."""

    x: int
    y: int
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return (x, y, width, height) tuple for compatibility."""
        return (self.x, self.y, self.width, self.height)

    @classmethod
    def from_tuple(cls, t: tuple[int, int, int, int]) -> "BoundingBox":
        """Create from (x, y, width, height) tuple."""
        return cls(x=int(t[0]), y=int(t[1]), width=int(t[2]), height=int(t[3]))


@dataclass(frozen=True)
class ScrollResult:
    """Result from successful scroll detection."""

    scroll_distance: int
    after_bbox: BoundingBox
    viewport: BoundingBox


# --- Kadane's Algorithm Variants (O(N) 1D Solvers) ---


def get_best_1d_anchored(arr: np.ndarray, anchor_idx: int) -> tuple[int, int, float]:
    """
    Finds the max-sum subarray that MUST include arr[anchor_idx].
    Complexity: O(N)
    """
    N = len(arr)
    if anchor_idx < 0 or anchor_idx >= N:
        return 0, 0, -np.inf

    # Scan Left: Find max prefix sum ending at anchor
    left_scan = np.cumsum(arr[anchor_idx::-1])
    best_left = np.argmax(left_scan)
    max_left = left_scan[best_left]
    start = anchor_idx - best_left

    # Scan Right: Find max suffix sum starting after anchor
    max_right = 0
    end = anchor_idx
    if anchor_idx + 1 < N:
        right_scan = np.cumsum(arr[anchor_idx + 1 :])
        best_right = np.argmax(right_scan)
        if right_scan[best_right] > 0:
            max_right = right_scan[best_right]
            end = (anchor_idx + 1) + best_right

    return start, end, max_left + max_right


def get_best_1d_unanchored(arr: np.ndarray, offset: int = 0) -> tuple[int, int, float]:
    """
    Standard Kadane's Algorithm to find max-sum subarray anywhere.
    Complexity: O(N)
    """
    if len(arr) == 0:
        return 0, 0, -np.inf

    max_so_far = -np.inf
    max_ending_here = 0
    start_idx = 0
    best_start, best_end = 0, 0

    for i, x in enumerate(arr):
        max_ending_here += x
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            best_start = start_idx
            best_end = i
        if max_ending_here < 0:
            max_ending_here = 0
            start_idx = i + 1

    return best_start + offset, best_end + offset, max_so_far


def get_best_1d_range_constrained(
    arr: np.ndarray, r_min: int, r_max: int
) -> tuple[int, int, float]:
    """
    Finds max-sum subarray that overlaps the interval [r_min, r_max].
    Checks 3 topological cases: Inside, Crossing Top, Crossing Bottom.
    """
    candidates = []

    # Case 1: Fully Inside [r_min, r_max]
    if r_max >= r_min:
        s, e, score = get_best_1d_unanchored(arr[r_min : r_max + 1], offset=r_min)
        candidates.append((s, e, score))

    # Case 2: Anchored to Top Boundary (r_min)
    s, e, score = get_best_1d_anchored(arr, r_min)
    candidates.append((s, e, score))

    # Case 3: Anchored to Bottom Boundary (r_max)
    if r_max != r_min:
        s, e, score = get_best_1d_anchored(arr, r_max)
        candidates.append((s, e, score))

    return max(candidates, key=lambda x: x[2])


# --- Phase 1: Initial Viewport Estimation ---


def estimate_initial_viewport(
    before: np.ndarray,
    after: np.ndarray,
    cursor_pos: tuple[int, int],
    pixel_tolerance: int = 15,
) -> tuple[int, int, int, int] | None:
    """
    Estimates the initial viewport by finding the largest region of changed pixels,
    anchored at the cursor position.

    Uses binary thresholding of pixel differences, then mean-normalized weights
    to handle varying viewport sizes robustly.

    Args:
        before: Grayscale image before scroll (H, W)
        after: Grayscale image after scroll (H, W)
        cursor_pos: (x, y) cursor position
        pixel_tolerance: Maximum difference to consider pixels unchanged

    Returns:
        (x, y, width, height) viewport bounds or None if detection fails
    """
    # Threshold ratio for mean-normalized weights
    # Lower value = less penalty for unchanged rows, allowing viewport to bridge gaps
    CHANGE_THRESHOLD_RATIO = 0.15

    H, W = before.shape
    cx, cy = cursor_pos

    # Validate cursor position
    if not (0 <= cx < W and 0 <= cy < H):
        logging.debug(f"Initial viewport: cursor out of bounds ({cx}, {cy})")
        return None

    # Step 1-2: Compute diff and threshold to binary change map
    diff = np.abs(after.astype(float) - before.astype(float))
    changed = diff > pixel_tolerance

    # Step 3-4: Sum changed pixels per row and normalize
    row_sums = np.sum(changed, axis=1).astype(float)
    active_rows = row_sums > 0

    if not np.any(active_rows):
        logging.debug("Initial viewport: no changed pixels detected")
        return None

    mean_row_activity = np.mean(row_sums[active_rows])
    row_weights = row_sums - (CHANGE_THRESHOLD_RATIO * mean_row_activity)

    # Step 5: Run anchored Kadane to find row range
    r1, r2, r_score = get_best_1d_anchored(row_weights, cy)

    if r_score <= 0 or r1 > r2:
        logging.debug(f"Initial viewport: no valid row range (score={r_score:.2f})")
        return None

    # Step 6: Repeat for columns using found row range
    changed_cropped = changed[r1 : r2 + 1, :]
    col_sums = np.sum(changed_cropped, axis=0).astype(float)
    active_cols = col_sums > 0

    if not np.any(active_cols):
        logging.debug("Initial viewport: no changed pixels in row range")
        return None

    mean_col_activity = np.mean(col_sums[active_cols])
    col_weights = col_sums - (CHANGE_THRESHOLD_RATIO * mean_col_activity)

    # Step 7: Run anchored Kadane to find column range
    c1, c2, c_score = get_best_1d_anchored(col_weights, cx)

    if c_score <= 0 or c1 > c2:
        logging.debug(f"Initial viewport: no valid column range (score={c_score:.2f})")
        return None

    width = c2 - c1 + 1
    height = r2 - r1 + 1

    logging.debug(
        f"Initial viewport: ({c1}, {r1}) {width}x{height}, "
        f"row_score={r_score:.2f}, col_score={c_score:.2f}"
    )

    return (c1, r1, width, height)


# --- Phase 2: Scroll Distance Estimation ---


def estimate_scroll_distance(
    before: np.ndarray,
    after: np.ndarray,
    viewport: tuple[int, int, int, int],
) -> int | None:
    """
    Estimates scroll distance using summed strip correlations with NCC normalization.

    Splits the viewport into vertical strips, computes correlation for each strip
    at each candidate distance, sums across strips, and normalizes using NCC
    (Normalized Cross-Correlation). The distance with maximum NCC is returned.

    Args:
        before: Grayscale image before scroll (H, W)
        after: Grayscale image after scroll (H, W)
        viewport: (x, y, width, height) viewport bounds

    Returns:
        Scroll distance in pixels (positive = content scrolled up), or None
    """
    MIN_SCROLL_DISTANCE = 20  # Minimum scroll distance to consider (pixels)
    MIN_OVERLAP_HEIGHT = 100  # Minimum overlap height required (prevents edge effects)
    NUM_STRIPS = 16  # Number of vertical strips for correlation

    x, y, w, h = viewport

    # Crop both images to viewport bounds
    before_crop = before[y : y + h, x : x + w]
    after_crop = after[y : y + h, x : x + w]

    # Compute vertical gradients (row-to-row differences) with abs applied immediately
    grad_before = np.abs(np.diff(before_crop, axis=0))
    grad_after = np.abs(np.diff(after_crop, axis=0))

    # Use gradient height for correlation (h - 1 due to diff)
    h_grad = grad_before.shape[0]

    # Constraint: overlap = h_grad - d >= MIN_OVERLAP_HEIGHT
    max_d = h_grad - MIN_OVERLAP_HEIGHT
    if max_d <= MIN_SCROLL_DISTANCE:
        logging.debug(
            "Viewport too small for minimum scroll distance with required overlap"
        )
        return None

    # 1. Profile Construction
    # Split images into vertical strips
    strips_b = np.array_split(grad_before, NUM_STRIPS, axis=1)
    strips_a = np.array_split(grad_after, NUM_STRIPS, axis=1)

    # Sum horizontally within strips, then stack side-by-side
    # Shape: (h_grad, NUM_STRIPS)
    profs_b = np.stack([s.sum(axis=1) for s in strips_b], axis=1)
    profs_a = np.stack([s.sum(axis=1) for s in strips_a], axis=1)

    # 2. Batch Correlation
    # Correlate each strip column-wise
    # Result Shape: (2*h_grad - 1, NUM_STRIPS)
    full_corrs = np.stack(
        [
            np.correlate(profs_b[:, i], profs_a[:, i], mode="full")
            for i in range(NUM_STRIPS)
        ],
        axis=1,
    )

    # Extract positive lags (rows from h_grad onwards)
    # valid_corrs Shape: (h_grad - 1, NUM_STRIPS)
    valid_corrs = full_corrs[h_grad:, :]

    # Sum across all strips (axis 1) to get the final raw correlation profile
    raw_corr = np.zeros(h_grad)
    raw_corr[1:] = valid_corrs.sum(axis=1)

    # 3. Vectorized Norm Calculation
    # Compute cumulative sums down the columns (axis 0)
    cs_b = np.cumsum(profs_b**2, axis=0)
    cs_a = np.cumsum(profs_a**2, axis=0)
    total_b = cs_b[-1, :]  # Shape: (NUM_STRIPS,)

    # 'b' segment: Total energy - energy accumulated before the split index
    # We look at all indices up to the last one (0 to h_grad-2)
    b_contribs = total_b - cs_b[:-1, :]

    # 'a' segment: Energy accumulated from 0 up to the overlap index
    # We take the prefix sums and reverse the order of rows
    a_contribs = cs_a[:-1, :][::-1, :]

    # Sum squared norms across all strips (axis 1)
    # Prepend a 0 to match the raw_corr indexing (d=0 is unused/zero)
    norm_b_sq = np.concatenate(([0], b_contribs.sum(axis=1)))
    norm_a_sq = np.concatenate(([0], a_contribs.sum(axis=1)))

    # 4. Final NCC Calculation
    denom = np.sqrt(norm_b_sq * norm_a_sq)
    ncc = np.divide(raw_corr, denom, out=np.zeros_like(raw_corr), where=denom != 0)

    # Find best distance (argmax of NCC)
    # Only consider valid range [MIN_SCROLL_DISTANCE, max_d]
    valid_range = ncc[MIN_SCROLL_DISTANCE : max_d + 1]
    if len(valid_range) == 0 or np.max(valid_range) <= 0:
        logging.debug("No positive correlation found")
        return None

    best_d = int(np.argmax(valid_range) + MIN_SCROLL_DISTANCE)
    best_ncc = ncc[best_d]

    logging.debug(f"Scroll distance: {best_d}px (NCC={best_ncc:.4f})")
    return best_d


# --- Phase 3: Viewport Refinement ---


def refine_viewport(
    before: np.ndarray,
    after: np.ndarray,
    initial_viewport: tuple[int, int, int, int],
    scroll_distance: int,
    cursor_pos: tuple[int, int],
    pixel_tolerance: int = 15,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """
    Refines viewport bounds by finding matching pixels in aligned regions.

    Uses three-category weighting to handle static content:
    - Dynamic match: pixels that match in overlap but changed position (full positive weight)
    - Static match: pixels that match but were already identical before scroll (epsilon weight)
    - Non-match: pixels that don't align in overlap (negative weight)

    Args:
        before: Grayscale image before scroll (H, W)
        after: Grayscale image after scroll (H, W)
        initial_viewport: (x, y, width, height) initial viewport estimate
        scroll_distance: Detected scroll distance in pixels
        cursor_pos: (x, y) cursor position
        pixel_tolerance: Maximum difference to consider pixels matching

    Returns:
        Tuple of (refined_viewport, after_bbox) or None if refinement fails.
        refined_viewport: (x, y, width, height) refined viewport bounds
        after_bbox: (x, y, width, height) overlap region in after image
    """
    DENSITY_THRESHOLD = 0.5  # Density threshold for matching pixel detection
    STATIC_EPSILON = 1e-6  # Small positive weight for static pixels

    H, _ = before.shape
    cx, cy = cursor_pos
    init_x, _, init_w, _ = initial_viewport
    d = scroll_distance

    # Calculate aligned regions
    # After content at [0 : H-d] corresponds to Before content at [d : H]
    limit_h = H - d
    if limit_h <= 0:
        logging.debug(f"Viewport refinement: invalid limit_h={limit_h}")
        return None

    # Initial viewport column range (used for row refinement only)
    c1_init = init_x
    c2_init = init_x + init_w

    # --- Row refinement: use initial viewport columns ---
    after_region_init = after[:limit_h, c1_init:c2_init]
    before_shifted_init = before[d:, c1_init:c2_init]
    before_same_pos_init = before[:limit_h, c1_init:c2_init]
    after_shifted_init = after[d:, c1_init:c2_init]  # For source_static check

    # Compute match maps for initial column range
    overlap_diff_init = np.abs(
        after_region_init.astype(float) - before_shifted_init.astype(float)
    )
    overlap_match_init = overlap_diff_init < pixel_tolerance

    # Check if destination position is static (after[y] == before[y])
    dest_static_diff_init = np.abs(
        after_region_init.astype(float) - before_same_pos_init.astype(float)
    )
    dest_static_init = dest_static_diff_init < pixel_tolerance

    # Check if source position is static (before[y+d] == after[y+d])
    source_static_diff_init = np.abs(
        before_shifted_init.astype(float) - after_shifted_init.astype(float)
    )
    source_static_init = source_static_diff_init < pixel_tolerance

    # Three categories for row refinement
    # Static if EITHER destination or source position is static
    static_match_init = dest_static_init | source_static_init
    dynamic_match_init = overlap_match_init & ~static_match_init
    static_in_overlap_init = overlap_match_init & static_match_init

    # Build weights for row refinement
    weights_init = np.full_like(
        overlap_match_init, -DENSITY_THRESHOLD - 1e-5, dtype=float
    )
    weights_init[dynamic_match_init] = 1.0 - DENSITY_THRESHOLD
    weights_init[static_in_overlap_init] = STATIC_EPSILON

    # Compute cursor constraints (must overlap cursor's scroll path)
    target_y_max = min(cy, limit_h - 1)
    target_y_min = max(0, cy - d)
    if target_y_min > target_y_max:
        target_y_min = target_y_max

    row_weights = np.sum(weights_init, axis=1)
    r1, r2, r_score = get_best_1d_range_constrained(
        row_weights, target_y_min, target_y_max
    )

    if r_score <= 0:
        logging.debug(f"Viewport refinement: no valid row range (score={r_score:.2f})")
        return None

    # --- Column refinement: use full image width with refined rows ---
    after_region_full = after[r1 : r2 + 1, :]
    before_shifted_full = before[r1 + d : r2 + 1 + d, :]
    before_same_pos_full = before[r1 : r2 + 1, :]
    after_shifted_full = after[r1 + d : r2 + 1 + d, :]  # For source_static check

    # Compute match maps for full width
    overlap_diff_full = np.abs(
        after_region_full.astype(float) - before_shifted_full.astype(float)
    )
    overlap_match_full = overlap_diff_full < pixel_tolerance

    # Check if destination position is static (after[y] == before[y])
    dest_static_diff_full = np.abs(
        after_region_full.astype(float) - before_same_pos_full.astype(float)
    )
    dest_static_full = dest_static_diff_full < pixel_tolerance

    # Check if source position is static (before[y+d] == after[y+d])
    source_static_diff_full = np.abs(
        before_shifted_full.astype(float) - after_shifted_full.astype(float)
    )
    source_static_full = source_static_diff_full < pixel_tolerance

    # Three categories for column refinement
    # Static if EITHER destination or source position is static
    static_match_full = dest_static_full | source_static_full
    dynamic_match_full = overlap_match_full & ~static_match_full
    static_in_overlap_full = overlap_match_full & static_match_full

    # Build weights for column refinement
    weights_full = np.full_like(
        overlap_match_full, -DENSITY_THRESHOLD - 1e-5, dtype=float
    )
    weights_full[dynamic_match_full] = 1.0 - DENSITY_THRESHOLD
    weights_full[static_in_overlap_full] = STATIC_EPSILON

    col_weights = np.sum(weights_full, axis=0)
    c1, c2, c_score = get_best_1d_anchored(col_weights, cx)

    if c_score <= 0:
        logging.debug(
            f"Viewport refinement: no valid column range (score={c_score:.2f})"
        )
        return None

    # Compute final viewport
    h_overlap = r2 - r1 + 1
    w_refined = c2 - c1 + 1
    h_viewport = h_overlap + d

    after_bbox = (c1, r1, w_refined, h_overlap)
    refined_viewport = (c1, r1, w_refined, h_viewport)

    logging.debug(
        f"Viewport refined: ({c1}, {r1}) {w_refined}x{h_viewport}, "
        f"overlap_h={h_overlap}, r_score={r_score:.2f}, c_score={c_score:.2f}"
    )

    return (refined_viewport, after_bbox)


# --- Main Detection Function ---


def detect_scroll(
    img_before: np.ndarray,
    img_after: np.ndarray,
    cursor_pos: tuple[float, float],
    existing_viewport: BoundingBox | None = None,
) -> ScrollResult | None:
    """
    Detects vertical scrolling using a three-phase approach:
    1. Initial viewport estimation (find changed region)
    2. Scroll distance calculation (cross-correlation)
    3. Viewport refinement (find matching pixels)

    When existing_viewport is provided, skips Phase 1 and 3 (for second scroll detection).

    Args:
        img_before: Numpy array (H, W, 3) or (H, W) - image before scroll
        img_after: Numpy array (H, W, 3) or (H, W) - image after scroll
        cursor_pos: (x, y) tuple - cursor position
        existing_viewport: Optional BoundingBox from previous detection

    Returns:
        ScrollResult with scroll_distance, after_bbox, and viewport.
        Returns None if no valid scroll is found.
    """
    # Algorithm configuration constants
    MIN_VIEWPORT_HEIGHT = 150  # Detected viewport must be at least this tall
    MIN_VIEWPORT_WIDTH = 150  # Detected viewport must be at least this wide
    PIXEL_MATCH_TOLERANCE = 15  # Max pixel difference to consider a match

    # Convert cursor position to integers for array indexing
    cx, cy = int(cursor_pos[0]), int(cursor_pos[1])

    # Preprocessing: Convert to grayscale
    if img_before.ndim == 3:
        # Grayscale conversion using ITU-R BT.709 coefficients
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

    H, _ = gb.shape

    # --- Phase 1: Initial Viewport Estimation ---
    # Internal functions use tuples; convert at API boundary
    if existing_viewport is not None:
        # Use provided viewport (skip Phase 1)
        viewport = existing_viewport.as_tuple()
        logging.debug(f"Using existing viewport: {viewport}")
    else:
        # Estimate initial viewport by finding changed pixels
        viewport = estimate_initial_viewport(
            gb, ga, (cx, cy), pixel_tolerance=PIXEL_MATCH_TOLERANCE
        )
        if viewport is None:
            logging.info("Scroll detection failed: Could not estimate initial viewport")
            return None

    # Validate viewport size
    vp_x, vp_y, vp_w, vp_h = viewport
    if vp_h < MIN_VIEWPORT_HEIGHT or vp_w < MIN_VIEWPORT_WIDTH:
        logging.info(
            f"Scroll detection failed: Initial viewport too small "
            f"(detected: {vp_w}x{vp_h}, required: {MIN_VIEWPORT_WIDTH}x{MIN_VIEWPORT_HEIGHT})"
        )
        return None

    # --- Phase 2: Scroll Distance Estimation ---
    scroll_distance = estimate_scroll_distance(gb, ga, viewport)
    if scroll_distance is None:
        logging.info("Scroll detection failed: Could not estimate scroll distance")
        return None

    # Validate scroll distance against image bounds
    if scroll_distance >= H - MIN_VIEWPORT_HEIGHT:
        logging.info(
            f"Scroll detection failed: Scroll distance too large "
            f"(d={scroll_distance}, max={H - MIN_VIEWPORT_HEIGHT})"
        )
        return None

    # --- Phase 3: Viewport Refinement ---
    if existing_viewport is not None:
        # Skip refinement when using existing viewport (second scroll)
        # Use existing viewport as refined viewport
        refined_viewport = viewport  # Already a tuple from conversion above
        # Compute after_bbox from viewport and scroll distance
        x, y, w, h = viewport
        h_overlap = h - scroll_distance
        if h_overlap <= 0:
            logging.info(
                f"Scroll detection failed: Invalid overlap height ({h_overlap})"
            )
            return None
        after_bbox = (x, y, w, h_overlap)
    else:
        # Refine viewport by finding matching pixels
        result = refine_viewport(
            gb,
            ga,
            viewport,
            scroll_distance,
            (cx, cy),
            pixel_tolerance=PIXEL_MATCH_TOLERANCE,
        )
        if result is None:
            logging.info("Scroll detection failed: Could not refine viewport")
            return None

        refined_viewport, after_bbox = result

    # Final Feasibility Check
    _, _, rv_w, rv_h = refined_viewport
    if rv_h < MIN_VIEWPORT_HEIGHT or rv_w < MIN_VIEWPORT_WIDTH:
        logging.info(
            f"Scroll detection failed: Refined viewport too small "
            f"(detected: {rv_w}x{rv_h}, required: {MIN_VIEWPORT_WIDTH}x{MIN_VIEWPORT_HEIGHT})"
        )
        return None

    return ScrollResult(
        scroll_distance=int(scroll_distance),
        after_bbox=BoundingBox.from_tuple(after_bbox),
        viewport=BoundingBox.from_tuple(refined_viewport),
    )
