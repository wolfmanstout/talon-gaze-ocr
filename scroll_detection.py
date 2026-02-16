"""Scroll detection algorithm for detecting vertical scrolling in screenshots.

Uses a three-phase approach:
1. Initial viewport estimation (find changed pixels via binary diff + Kadane)
2. Scroll distance calculation (strip-based NCC voting on gradient profiles)
3. Viewport refinement (find matching pixels to refine bounds)

This module contains only the pure algorithmic functions with no Talon dependencies.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Module-level constants
PIXEL_TOLERANCE = 20  # Max pixel difference to consider a match
MIN_VIEWPORT_HEIGHT = 150  # Detected viewport must be at least this tall
MIN_VIEWPORT_WIDTH = 150  # Detected viewport must be at least this wide


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
    cursor_pos: tuple[int, int],
    same_pos_diff: NDArray[np.floating[Any]],
) -> tuple[int, int, int, int] | None:
    """
    Estimates the initial viewport by finding the largest region of changed pixels,
    anchored at the cursor position.

    Uses binary thresholding of pixel differences, then mean-normalized weights
    to handle varying viewport sizes robustly.

    Args:
        cursor_pos: (x, y) cursor position
        same_pos_diff: Precomputed |after - before| diff

    Returns:
        (x, y, width, height) viewport bounds or None if detection fails
    """
    # Threshold ratio for mean-normalized weights
    # Lower value = less penalty for unchanged rows, allowing viewport to bridge gaps
    CHANGE_THRESHOLD_RATIO = 0.15

    H, W = same_pos_diff.shape
    cx, cy = cursor_pos

    # Validate cursor position
    if not (0 <= cx < W and 0 <= cy < H):
        logging.debug(f"Initial viewport: cursor out of bounds ({cx}, {cy})")
        return None

    # Step 1-2: Threshold precomputed diff to binary change map
    changed = same_pos_diff > PIXEL_TOLERANCE

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

    return (c1, r1, width, height)


# --- Phase 2: Scroll Distance Estimation ---


def estimate_scroll_distance(
    before: NDArray[np.floating[Any]],
    after: NDArray[np.floating[Any]],
    viewport: tuple[int, int, int, int],
    scroll_direction: str = "down",
) -> int | None:
    """
    Estimates scroll distance using summed strip correlations with NCC normalization.

    Splits the viewport into vertical strips, computes correlation for each strip
    at each candidate distance, sums across strips, and normalizes using NCC
    (Normalized Cross-Correlation). The distance with maximum NCC is returned.

    Args:
        before: Grayscale float image before scroll (H, W)
        after: Grayscale float image after scroll (H, W)
        viewport: (x, y, width, height) viewport bounds
        scroll_direction: "down" (content moves up) or "up" (content moves down)

    Returns:
        Scroll distance in pixels (always positive magnitude), or None
    """
    MIN_SCROLL_DISTANCE = 1  # Minimum scroll distance to consider (pixels)
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

    # Sum across all strips (axis 1) to get the final raw correlation profile
    raw_corr = np.zeros(h_grad)
    raw_corr[1:] = valid_corrs.sum(axis=1)

    # 3. Vectorized Norm Calculation
    # Compute cumulative sums down the columns (axis 0)
    cs_b = np.cumsum(profs_b**2, axis=0)
    cs_a = np.cumsum(profs_a**2, axis=0)
    total_b = cs_b[-1, :]  # Shape: (NUM_STRIPS,)
    total_a = cs_a[-1, :]  # Shape: (NUM_STRIPS,)

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
    return best_d


# --- Phase 3: Viewport Refinement ---


def _build_weight_map(
    overlap_diff: NDArray[np.floating[Any]],
    dest_static: NDArray[np.bool_],
    source_static: NDArray[np.bool_],
) -> NDArray[np.floating[Any]]:
    """Build weight map with three categories: dynamic, static, and mismatch."""
    DENSITY_THRESHOLD = 0.01  # Density threshold for matching pixel detection
    STATIC_EPSILON = 1e-6  # Small positive weight for static pixels

    overlap_match = overlap_diff < PIXEL_TOLERANCE
    static_match = dest_static | source_static
    dynamic_match = overlap_match & ~static_match
    static_in_overlap = overlap_match & static_match

    weight_map = np.full_like(overlap_match, -DENSITY_THRESHOLD - 1e-5, dtype=float)
    weight_map[dynamic_match] = 1.0 - DENSITY_THRESHOLD
    weight_map[static_in_overlap] = STATIC_EPSILON
    return weight_map


def refine_viewport(
    before: NDArray[np.floating[Any]],
    after: NDArray[np.floating[Any]],
    initial_viewport: tuple[int, int, int, int],
    scroll_distance: int,
    cursor_pos: tuple[int, int],
    same_pos_diff: NDArray[np.floating[Any]],
    scroll_direction: str = "down",
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """
    Refines viewport bounds by finding matching pixels in aligned regions.

    Uses three-category weighting to handle static content:
    - Dynamic match: pixels that match in overlap but changed position (full positive weight)
    - Static match: pixels that match but were already identical before scroll (epsilon weight)
    - Non-match: pixels that don't align in overlap (negative weight)

    Args:
        before: Grayscale float image before scroll (H, W)
        after: Grayscale float image after scroll (H, W)
        initial_viewport: (x, y, width, height) initial viewport estimate
        scroll_distance: Detected scroll distance in pixels
        cursor_pos: (x, y) cursor position
        same_pos_diff: Precomputed |after - before| diff for static pixel detection
        scroll_direction: "down" (content moves up) or "up" (content moves down)

    Returns:
        Tuple of (refined_viewport, after_bbox) or None if refinement fails.
        refined_viewport: (x, y, width, height) refined viewport bounds
        after_bbox: (x, y, width, height) overlap region in after image
    """
    H, _ = before.shape
    cx, cy = cursor_pos
    init_x, _, init_w, _ = initial_viewport
    d = scroll_distance

    # Calculate aligned regions
    # For scroll-down: After content at [0 : H-d] corresponds to Before content at [d : H]
    # For scroll-up: After content at [d : H] corresponds to Before content at [0 : H-d]
    limit_h = H - d
    if limit_h <= 0:
        logging.debug(f"Viewport refinement: invalid limit_h={limit_h}")
        return None

    # Initial viewport column range (used for row refinement only)
    c1_init = init_x
    c2_init = init_x + init_w

    # Precompute full-width shifted diff once for reuse in row and column refinement
    # This avoids redundant np.abs() calls over overlapping regions
    if scroll_direction == "down":
        shifted_diff = np.abs(after[:limit_h, :] - before[d:, :])
    else:
        shifted_diff = np.abs(after[d:, :] - before[:limit_h, :])

    # --- Row refinement: use initial viewport columns ---
    # Extract the overlap regions for match analysis
    if scroll_direction == "down":
        # Static detection uses precomputed same-position diff
        dest_static_row = same_pos_diff[:limit_h, c1_init:c2_init] < PIXEL_TOLERANCE
        source_static_row = same_pos_diff[d:, c1_init:c2_init] < PIXEL_TOLERANCE
    else:
        # Scroll-up: after[d:] matches before[:limit_h]
        dest_static_row = same_pos_diff[d:, c1_init:c2_init] < PIXEL_TOLERANCE
        source_static_row = same_pos_diff[:limit_h, c1_init:c2_init] < PIXEL_TOLERANCE

    weight_map_row = _build_weight_map(
        shifted_diff[:, c1_init:c2_init], dest_static_row, source_static_row
    )

    # Cursor constraint: detected range must include part of cursor's scroll path [cy-d, cy]
    target_row_min = max(0, cy - d)
    target_row_max = min(cy, limit_h - 1)
    if target_row_min > target_row_max:
        target_row_min = target_row_max

    row_weights = np.sum(weight_map_row, axis=1)
    r1, r2, row_score = get_best_1d_range_constrained(
        row_weights, target_row_min, target_row_max
    )

    if row_score <= 0:
        logging.debug(
            f"Viewport refinement: no valid row range (score={row_score:.2f})"
        )
        return None

    # --- Column refinement: use full image width with refined rows ---
    # r1, r2 are slice indices; convert to screen rows based on direction
    if scroll_direction == "down":
        # Static detection uses precomputed same-position diff
        dest_static_col = same_pos_diff[r1 : r2 + 1, :] < PIXEL_TOLERANCE
        source_static_col = same_pos_diff[r1 + d : r2 + 1 + d, :] < PIXEL_TOLERANCE
    else:
        # Scroll-up: screen row = slice index + d
        dest_static_col = same_pos_diff[r1 + d : r2 + 1 + d, :] < PIXEL_TOLERANCE
        source_static_col = same_pos_diff[r1 : r2 + 1, :] < PIXEL_TOLERANCE

    weight_map_col = _build_weight_map(
        shifted_diff[r1 : r2 + 1, :], dest_static_col, source_static_col
    )

    col_weights = np.sum(weight_map_col, axis=0)
    c1, c2, col_score = get_best_1d_anchored(col_weights, cx)

    if col_score <= 0:
        logging.debug(
            f"Viewport refinement: no valid column range (score={col_score:.2f})"
        )
        return None

    # Compute final viewport
    h_overlap = r2 - r1 + 1
    w_refined = c2 - c1 + 1
    h_viewport = h_overlap + d

    # Convert r1 from slice index to screen row for the final bounding boxes
    # after_bbox: where the overlap region is in the after image
    # refined_viewport: the full viewport including new content
    if scroll_direction == "down":
        # Scroll-down: r1 is already a screen row (overlap was after[:limit_h])
        # Viewport starts at overlap and extends DOWN by d
        after_bbox_y = r1
        viewport_y = r1
    else:
        # Scroll-up: r1 is slice index, overlap is at screen row r1 + d
        # Viewport starts d pixels ABOVE the overlap (to include new content at top)
        after_bbox_y = r1 + d
        viewport_y = r1  # Same as overlap_y - d

    after_bbox = (c1, after_bbox_y, w_refined, h_overlap)
    refined_viewport = (c1, viewport_y, w_refined, h_viewport)

    return (refined_viewport, after_bbox)


def _to_grayscale(img: np.ndarray) -> NDArray[np.floating[Any]]:
    """Convert RGB image to grayscale using ITU-R BT.709 coefficients."""
    if img.ndim == 3:
        return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
    return img.astype(float)


def _validate_viewport_size(viewport: tuple[int, int, int, int], context: str) -> bool:
    """Check if viewport meets minimum size requirements."""
    _, _, w, h = viewport
    if h < MIN_VIEWPORT_HEIGHT or w < MIN_VIEWPORT_WIDTH:
        logging.info(
            f"Scroll detection failed: {context} viewport too small "
            f"(detected: {w}x{h}, required: {MIN_VIEWPORT_WIDTH}x{MIN_VIEWPORT_HEIGHT})"
        )
        return False
    return True


# --- Main Detection Function ---


def detect_scroll(
    img_before: np.ndarray,
    img_after: np.ndarray,
    cursor_pos: tuple[float, float],
    existing_viewport: BoundingBox | None = None,
    scroll_direction: str = "down",
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
        scroll_direction: "down" (content moves up) or "up" (content moves down)

    Returns:
        ScrollResult with scroll_distance, after_bbox, and viewport.
        Returns None if no valid scroll is found.
    """
    # Convert cursor position to integers for array indexing
    cx, cy = int(cursor_pos[0]), int(cursor_pos[1])

    # Preprocessing: Convert to grayscale
    gb = _to_grayscale(img_before)
    ga = _to_grayscale(img_after)
    H, _ = gb.shape

    # Precompute same-position diff once for reuse in Phase 1 and Phase 3
    # This avoids redundant computation of |after - before| in multiple places
    same_pos_diff: NDArray[np.floating[Any]] | None = None
    if existing_viewport is None:
        same_pos_diff = np.abs(ga - gb)

    # --- Phase 1: Initial Viewport Estimation ---
    # Internal functions use tuples; convert at API boundary
    if existing_viewport is not None:
        # Use provided viewport (skip Phase 1)
        viewport = existing_viewport.as_tuple()
    else:
        # Estimate initial viewport by finding changed pixels
        assert same_pos_diff is not None
        viewport = estimate_initial_viewport((cx, cy), same_pos_diff)
        if viewport is None:
            logging.info("Scroll detection failed: Could not estimate initial viewport")
            return None

    # Validate viewport size
    if not _validate_viewport_size(viewport, "Initial"):
        return None

    # --- Phase 2: Scroll Distance Estimation ---
    scroll_distance = estimate_scroll_distance(gb, ga, viewport, scroll_direction)
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
        refined_viewport = viewport
        # Compute after_bbox from viewport and scroll distance
        x, y, w, h = viewport
        h_overlap = h - scroll_distance
        if h_overlap <= 0:
            logging.info(
                f"Scroll detection failed: Invalid overlap height ({h_overlap})"
            )
            return None
        # For scroll-down: overlap is at top (y unchanged)
        # For scroll-up: overlap is at bottom (y + scroll_distance)
        overlap_y = y if scroll_direction == "down" else y + scroll_distance
        after_bbox = (x, overlap_y, w, h_overlap)
    else:
        # Refine viewport by finding matching pixels
        assert same_pos_diff is not None
        result = refine_viewport(
            gb,
            ga,
            viewport,
            scroll_distance,
            (cx, cy),
            same_pos_diff,
            scroll_direction=scroll_direction,
        )
        if result is None:
            logging.info("Scroll detection failed: Could not refine viewport")
            return None
        refined_viewport, after_bbox = result

    # Final Feasibility Check
    if not _validate_viewport_size(refined_viewport, "Refined"):
        return None

    return ScrollResult(
        scroll_distance=int(scroll_distance),
        after_bbox=BoundingBox.from_tuple(after_bbox),
        viewport=BoundingBox.from_tuple(refined_viewport),
    )
