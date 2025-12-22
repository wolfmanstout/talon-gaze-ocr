import glob
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from talon import Context, Module, actions, app, cron, fs, screen, settings, ui
from talon.canvas import Canvas, MouseEvent
from talon.skia.typeface import Fontstyle, Typeface
from talon.types import rect

from .timestamped_captures import TextRange, TimestampedText

try:
    from talon.experimental import ocr
except ImportError:
    ocr = None

# Adjust path to search adjacent package directories. Prefixed with dot to avoid
# Talon running them itself. Append to search path so that faster binary
# packages can be used instead if available.
subtree_dir = Path(__file__).parent / ".subtrees"
package_paths = [
    str(subtree_dir / "gaze-ocr"),
    str(subtree_dir / "screen-ocr"),
    str(subtree_dir / "rapidfuzz/src"),
]
saved_path = sys.path.copy()
try:
    sys.path.extend([path for path in package_paths if path not in sys.path])
    import gaze_ocr
    import screen_ocr  # dependency of gaze-ocr
    from gaze_ocr import talon_adapter
finally:
    # Restore the unmodified path.
    sys.path = saved_path.copy()

mod = Module()
ctx = Context()

mod.setting(
    "ocr_use_talon_backend",
    type=bool,
    default=True,
    desc="If true, use Talon backend, otherwise use default fast backend from screen_ocr.",
)
mod.setting(
    "ocr_connect_tracker",
    type=bool,
    default=True,
    desc="If true, automatically connect the eye tracker at startup.",
)
mod.setting(
    "ocr_logging_dir",
    type=str,
    default=None,
    desc="If specified, log OCR'ed images to this directory.",
)
mod.setting(
    "ocr_click_offset_right",
    type=int,
    default=0,
    desc="Adjust the X-coordinate when clicking around OCR text.",
)
mod.setting(
    "ocr_select_pause_seconds",
    type=float,
    default=0.5,
    desc="Adjust the pause between clicks when performing a selection.",
)
mod.setting(
    "ocr_use_window_at_api",
    type=bool,
    default=False,
    desc="Use ui.window_at() API for focusing windows (requires beta Talon). Falls back to accessibility API if disabled or unavailable.",
)
mod.setting(
    "ocr_debug_display_seconds",
    type=float,
    default=3,
    desc="Adjust how long debugging display is shown.",
)
mod.setting(
    "ocr_disambiguation_display_seconds",
    type=float,
    default=5,
    desc="Adjust how long disambiguation display is shown. Use 0 to remove timeout.",
)
mod.setting(
    "ocr_gaze_box_padding",
    type=int,
    default=100,
    desc="How much padding is applied to gaze bounding box when searching for text.",
)
mod.setting(
    "ocr_gaze_point_padding",
    type=int,
    default=200,
    desc="How much padding is applied to gaze point when taking screenshots for debug overlay commands.",
)
mod.setting(
    "ocr_light_background_debug_color",
    type=str,
    default="000000",
    desc="Debug color to use on a light background",
)
mod.setting(
    "ocr_dark_background_debug_color",
    type=str,
    default="FFFFFF",
    desc="Debug color to use on a dark background",
)
mod.setting(
    "ocr_behavior_when_no_eye_tracker",
    type=Literal["MAIN_SCREEN", "ACTIVE_WINDOW"],
    default="MAIN_SCREEN",
    desc="Region to OCR when no data from the eye tracker",
)
mod.setting(
    "ocr_scroll_indicator_enabled",
    type=bool,
    default=True,
    desc="Enable visual indicator when scrolling.",
)
mod.setting(
    "ocr_scroll_indicator_fade_seconds",
    type=float,
    default=1.5,
    desc="Duration for scroll indicator line to fade out.",
)
mod.setting(
    "ocr_scroll_indicator_color",
    type=str,
    default="ADD8E6",
    desc="Color for scroll indicator line (hex RGB).",
)
mod.setting(
    "ocr_scroll_wait_ms",
    type=int,
    default=50,
    desc="Milliseconds to wait after scroll before capturing 'after' screenshot.",
)
mod.setting(
    "ocr_scroll_debug_mode",
    type=bool,
    default=True,
    desc="Show full debug visualization (red/green boxes) instead of just the line.",
)
mod.setting(
    "ocr_scroll_viewport_fraction",
    type=float,
    default=0.8,
    desc="Fraction of viewport height to scroll (0.0-1.0).",
)

mod.tag(
    "gaze_ocr_disambiguation",
    desc="Tag for disambiguating between different onscreen matches.",
)
mod.list("ocr_actions", desc="Actions to perform on selected text.")
mod.list(
    "ocr_common_actions", desc="Common actions that can be used without 'seen'/'scene'."
)
mod.list("ocr_modifiers", desc="Modifiers to perform on selected text.")
mod.list("onscreen_ocr_text", desc="Selection list for onscreen text.")


def paste_link() -> None:
    actions.user.hyperlink()
    actions.sleep("100ms")
    actions.edit.paste()


def capitalize() -> None:
    text = actions.edit.selected_text()
    actions.insert(text[0].capitalize() + text[1:] if text else "")


def uncapitalize() -> None:
    text = actions.edit.selected_text()
    actions.insert(text[0].lower() + text[1:] if text else "")


_OCR_ACTIONS: dict[str, Callable[[], None]] = {
    "": lambda: None,
    "select": lambda: None,
    "copy": lambda: actions.edit.copy(),
    "cut": lambda: actions.edit.cut(),
    "paste": lambda: actions.edit.paste(),
    "paste_link": paste_link,
    "delete": lambda: actions.key("backspace"),
    "delete_with_whitespace": lambda: actions.key("backspace"),
    "capitalize": capitalize,
    "uncapitalize": uncapitalize,
    "lowercase": lambda: actions.insert(actions.edit.selected_text().lower()),
    "uppercase": lambda: actions.insert(actions.edit.selected_text().upper()),
    "bold": lambda: actions.user.bold(),
    "italic": lambda: actions.user.italic(),
    "strikethrough": lambda: actions.user.strikethrough(),
    "number_list": lambda: actions.user.number_list(),
    "bullet_list": lambda: actions.user.bullet_list(),
    "link": lambda: actions.user.hyperlink(),
}

_OCR_MODIFIERS: dict[str, Callable[[], None]] = {
    "": lambda: None,
    "selectAll": lambda: actions.edit.select_all(),
}


@ctx.dynamic_list("user.onscreen_ocr_text")
def onscreen_ocr_text(phrase) -> str | list[str] | dict[str, str]:
    global gaze_ocr_controller, punctuation_table
    reset_disambiguation()
    gaze_ocr_controller.read_nearby((phrase[0].start, phrase[-1].end))
    selection_list = gaze_ocr_controller.latest_screen_contents().as_string()
    # Split camel-casing.
    selection_list = re.sub(r"([a-z])([A-Z])", r"\1 \2", selection_list)
    # Make punctuation speakable.
    selection_list = selection_list.translate(punctuation_table)
    return selection_list


def add_homophones(
    homophones: dict[str, Sequence[str]], to_add: Iterable[Iterable[str]]
):
    for words in to_add:
        merged_words = set(words)
        for word in words:
            old_words = homophones.get(word.lower(), [])
            merged_words.update(old_words)
        merged_words = sorted(merged_words)
        for word in merged_words:
            homophones[word.lower()] = merged_words


digits = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]
default_digits_map = {n: i for i, n in enumerate(digits)}

# Inline punctuation words in case people are using vanilla knausj, where these are not
# exposed. Listed in order of preference.
default_punctuation_words = {
    "back tick": "`",
    "grave": "`",
    "comma": ",",
    "period": ".",
    "full stop": ".",
    "semicolon": ";",
    "colon": ":",
    "forward slash": "/",
    "question mark": "?",
    "exclamation mark": "!",
    "exclamation point": "!",
    "asterisk": "*",
    "hash sign": "#",
    "number sign": "#",
    "percent sign": "%",
    "at sign": "@",
    "and sign": "&",
    "ampersand": "&",
    # Currencies
    "dollar sign": "$",
    "pound sign": "£",
    "hyphen": "-",
    "underscore": "_",
}


user_dir = Path(__file__).parents[1]
# Search user_dir to find homophones.csv
homophones_file = None
for path in glob.glob(str(user_dir / "**/homophones.csv"), recursive=True):
    homophones_file = path
    break
if homophones_file:
    logging.info(f"Found homophones file: {homophones_file}")
else:
    logging.warning("Could not find homophones.csv. Is knausj_talon installed?")


def get_knausj_homophones():
    phones = {}
    if not homophones_file:
        return phones
    with open(homophones_file) as f:
        for line in f:
            words = line.rstrip().split(",")
            merged_words = set(words)
            for word in words:
                old_words = phones.get(word.lower(), [])
                merged_words.update(old_words)
            merged_words = sorted(merged_words)
            for word in merged_words:
                phones[word.lower()] = merged_words
    return phones


def reload_backend(name, flags):
    # Initialize eye tracking and OCR.
    global tracker, ocr_reader, gaze_ocr_controller, punctuation_table
    tracker = talon_adapter.TalonEyeTracker()
    # Note: tracker is connected automatically in the constructor.
    if not settings.get("user.ocr_connect_tracker"):
        tracker.disconnect()
    homophones = get_knausj_homophones()
    # TODO: Get this through an action to support customization.
    add_homophones(
        homophones, [(str(num), spoken) for spoken, num in default_digits_map.items()]
    )
    # Attempt to use overridable action to get punctuation. This is available in
    # wolfmanstout_talon, but not yet in knausj_talon, so fallback if needed.
    try:
        punctuation_words = actions.user.get_punctuation_words()
    except KeyError:
        punctuation_words = default_punctuation_words
    add_homophones(
        homophones,
        [
            (punctuation, spoken.replace(" ", ""))
            for spoken, punctuation in punctuation_words.items()
        ],
    )
    # Add common OCR errors to homophones.
    add_homophones(
        homophones,
        [
            ("ok", "okay", "0k"),
            ("ally", "a11y"),
            ("AI", "Al"),
        ],
    )
    punctuation_table = str.maketrans(
        {
            punctuation: f" {spoken.replace(' ', '')} "
            for spoken, punctuation in reversed(default_punctuation_words.items())
            if len(punctuation) == 1
        }
    )
    setting_ocr_use_talon_backend = settings.get("user.ocr_use_talon_backend")
    if setting_ocr_use_talon_backend and ocr:
        ocr_reader = screen_ocr.Reader.create_reader(
            backend="talon",
            radius=settings.get("user.ocr_gaze_point_padding"),
            homophones=homophones,
        )
    else:
        if setting_ocr_use_talon_backend and not ocr:
            logging.info("Talon OCR not available, will rely on external support.")
        ocr_reader = screen_ocr.Reader.create_fast_reader(
            radius=settings.get("user.ocr_gaze_point_padding"),
            homophones=homophones,
        )
    gaze_ocr_controller = gaze_ocr.Controller(
        ocr_reader,
        tracker,
        mouse=talon_adapter.Mouse(),
        keyboard=talon_adapter.Keyboard(),
        app_actions=talon_adapter.AppActions(),
        save_data_directory=settings.get("user.ocr_logging_dir"),
        gaze_box_padding=settings.get("user.ocr_gaze_box_padding"),
        fallback_when_no_eye_tracker=gaze_ocr.EyeTrackerFallback[
            settings.get("user.ocr_behavior_when_no_eye_tracker").upper()
        ],
    )


def on_ready():
    reload_backend(None, None)
    if homophones_file:
        fs.watch(str(homophones_file), reload_backend)


app.register("ready", on_ready)


def has_light_background(screenshot):
    array = np.array(screenshot)
    # From https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
    grayscale = 0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]
    return np.mean(grayscale) > 128


# --- Scroll Detection Algorithm ---
# Vertical Scrolling Detection using Global Projection
# 1. Global vertical projection with correlation
# 2. Constraint-aware density search (Kadane's algorithm variants)


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


# Width of vertical slice around cursor for row-finding pass
ROW_SEARCH_SLICE_WIDTH = 100


def solve_constrained_box(
    binary_map: np.ndarray,
    target_col: int,
    target_y_range: tuple[int, int],
    threshold_vertical: float,
    threshold_horizontal: float,
) -> tuple[int, int, int, int] | None:
    """
    Solves 2D max-density box using 2-pass projection.

    Args:
        binary_map: 2D boolean array (True = match, False = mismatch)
        target_col: Column that must be included
        target_y_range: (min, max) row range that must be overlapped
        threshold_vertical: Minimum match density for vertical pass (row finding)
        threshold_horizontal: Minimum match density for horizontal pass (column finding)

    Returns:
        (r1, c1, r2, c2) bounding box or None
    """
    H, W = binary_map.shape

    r_min, r_max = target_y_range
    r_min, r_max = max(0, r_min), min(H - 1, r_max)

    # Pass 1: Vertical (Project Rows) -> Best Height
    # Use a vertical slice around target_col to reduce noise from non-scrolled regions
    half_slice = ROW_SEARCH_SLICE_WIDTH // 2
    slice_left = max(0, target_col - half_slice)
    slice_right = min(W, target_col + half_slice)

    # Weight Transformation: Match=+ve, Mismatch=-ve
    # Matches need to outweigh noise based on vertical threshold
    weights_vertical = np.where(
        binary_map[:, slice_left:slice_right],
        1.0 - threshold_vertical,
        -threshold_vertical - 1e-5,
    )
    row_prof = np.sum(weights_vertical, axis=1)
    r1, r2, r_score = get_best_1d_range_constrained(row_prof, r_min, r_max)

    if r_score <= 0:
        return None

    # Pass 2: Horizontal (Project Cols) -> Best Width
    # Crop to the best rows found and apply horizontal threshold
    weights_horizontal = np.where(
        binary_map, 1.0 - threshold_horizontal, -threshold_horizontal - 1e-5
    )
    col_prof = np.sum(weights_horizontal[r1 : r2 + 1, :], axis=0)

    if target_col < 0 or target_col >= W:
        return None

    # Anchored search (cursor must be included)
    c1, c2, c_score = get_best_1d_anchored(col_prof, target_col)

    if c_score <= 0:
        return None

    return (r1, c1, r2, c2)


def detect_scroll(
    img_before: np.ndarray, img_after: np.ndarray, cursor_pos: tuple[float, float]
) -> dict | None:
    """
    Detects vertical scrolling using strip-based voting and constraint-aware search.

    Args:
        img_before: Numpy array (H, W, 3) or (H, W) - image before scroll
        img_after: Numpy array (H, W, 3) or (H, W) - image after scroll
        cursor_pos: (x, y) tuple - cursor position

    Returns:
        Dictionary with:
            - scroll_distance: Integer pixels scrolled (content moved up by this amount)
            - after_bbox: (x, y, w, h) overlap region in the after image
            - consensus_strength: Ratio of best vote to second-best (quality metric)
        Returns None if no valid scroll is found.
    """
    # Algorithm configuration constants
    MIN_SCROLL_DISTANCE = 20  # Minimum scroll distance to consider (pixels)
    MIN_REGION_HEIGHT = 200  # Minimum overlap region after scroll
    MIN_VIEWPORT_HEIGHT = 150  # Detected viewport must be at least this tall
    MIN_VIEWPORT_WIDTH = 300  # Detected viewport must be at least this wide
    PIXEL_MATCH_TOLERANCE = 15  # Max pixel difference to consider a match
    DENSITY_THRESHOLD_VERTICAL = (
        0.85  # Minimum match density for vertical pass (row finding)
    )
    DENSITY_THRESHOLD_HORIZONTAL = (
        0.85  # Minimum match density for horizontal pass (column finding)
    )

    # Strip-based voting configuration
    NUM_STRIPS = 16  # Number of vertical strips to divide image into

    # Vertical tiling configuration (grid probes)
    NUM_TILES = 16  # Number of vertical tiles per strip (determines tile size)
    TILE_OVERLAP = 0.5  # Overlap between tiles (0.5 = 50%)

    # Correlation and filtering thresholds
    MIN_CORRELATION_SCORE = 0.35  # Minimum normalized correlation to consider
    MIN_STRIP_GRADIENT = (
        2.0  # Minimum mean gradient to consider strip/tile (skip flat regions)
    )
    PROBE_ENERGY_EPSILON = 1e-9  # Minimum probe energy to avoid division by zero
    NCC_EXPONENT = 3  # Exponent to apply to NCC scores (amplifies strong correlations)

    # Convert cursor position to integers for array indexing
    cx, cy = int(cursor_pos[0]), int(cursor_pos[1])

    # 1. Preprocessing (Vertical Gradients)
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

    # Absolute vertical gradients (row differences)
    grad_b = np.abs(gb[1:] - gb[:-1])
    grad_a = np.abs(ga[1:] - ga[:-1])

    # 2. Grid Probes: Find Distance 'd' using Strip-Based Voting with Vertical Tiling
    H_grad, W = grad_b.shape
    H = gb.shape[0]
    strip_width = W // NUM_STRIPS

    # Calculate tile parameters
    tile_size = H_grad // NUM_TILES
    step = int(tile_size * (1 - TILE_OVERLAP))

    # Initialize voting dictionary
    votes = defaultdict(float)

    # Process each strip
    for strip_idx in range(NUM_STRIPS):
        x_start = strip_idx * strip_width
        x_end = min(x_start + strip_width, W)

        # Extract full vertical strips
        strip_b = grad_b[:, x_start:x_end]
        strip_a = grad_a[:, x_start:x_end]

        # Skip flat strips (no content)
        if np.mean(strip_b) < MIN_STRIP_GRADIENT:
            continue

        # Project full 'After' strip (target)
        proj_a = np.sum(strip_a, axis=1)

        # Precompute window energy outside inner loop (tile_size is constant)
        target_sq = proj_a**2
        window_energy = np.convolve(target_sq, np.ones(tile_size), mode="valid")
        window_energy = np.sqrt(window_energy)

        # --- INNER LOOP: Vertical Tiles (Probes) ---
        # Slide a window down the 'Before' strip to create probes
        for y in range(0, H_grad - tile_size + 1, step):
            # Extract probe (chunk of Before)
            tile_b = strip_b[y : y + tile_size, :]

            # Skip empty tiles
            if np.mean(tile_b) < MIN_STRIP_GRADIENT:
                continue

            # Project probe
            proj_b_tile = np.sum(tile_b, axis=1)

            # Correlate probe against full target (mode='valid')
            raw_corr = np.correlate(proj_a, proj_b_tile, mode="valid")

            # Normalized Cross-Correlation (NCC)
            # Probe energy (scalar, varies per tile)
            probe_energy = np.sqrt(np.sum(proj_b_tile**2))

            if probe_energy < PROBE_ENERGY_EPSILON:
                continue

            # Normalized correlation
            norm_corr = raw_corr / (probe_energy * window_energy + PROBE_ENERGY_EPSILON)

            # Find best match
            best_idx = np.argmax(norm_corr)
            score = norm_corr[best_idx]

            # Apply exponent to amplify strong correlations
            score_weighted = score**NCC_EXPONENT

            # Calculate scroll distance
            # The probe started at 'y' in Before and matches at 'best_idx' in After
            # Scroll distance d = y - best_idx
            d = y - best_idx

            # Filter by score and minimum scroll distance
            if score > MIN_CORRELATION_SCORE and abs(d) >= MIN_SCROLL_DISTANCE:
                votes[d] += score_weighted

    # 3. Find winning scroll distance from votes
    if not votes:
        logging.info("Scroll detection failed: No votes from strip-based correlation")
        return None

    # Filter by minimum scroll distance and maximum scroll
    max_scroll = H - MIN_REGION_HEIGHT
    if max_scroll < MIN_SCROLL_DISTANCE:
        logging.info(
            f"Scroll detection failed: Screen height ({H}) too small for minimum scroll distance "
            f"(max_scroll={max_scroll}, required={MIN_SCROLL_DISTANCE})"
        )
        return None

    valid_votes = {
        d: v for d, v in votes.items() if MIN_SCROLL_DISTANCE <= d <= max_scroll
    }

    if not valid_votes:
        logging.info(
            f"Scroll detection failed: No valid votes in range "
            f"(min_scroll={MIN_SCROLL_DISTANCE}, max_scroll={max_scroll}, total_votes={len(votes)})"
        )
        return None

    # Get best and second-best
    sorted_votes = sorted(valid_votes.items(), key=lambda x: x[1], reverse=True)
    d, best_vote = sorted_votes[0]
    second_best_vote = sorted_votes[1][1] if len(sorted_votes) > 1 else 0

    # Calculate consensus strength
    if second_best_vote > 0:
        consensus_ratio = best_vote / second_best_vote
    else:
        consensus_ratio = np.inf if best_vote > 0 else 1.0

    # Log voting details for debugging (debug level - internal algorithm detail)
    logging.debug(
        f"Scroll voting: best_distance={d}px, "
        f"vote_strength={best_vote:.2f}, "
        f"consensus_ratio={consensus_ratio:.2f}"
    )

    # 4. Viewport Extraction (Shift & Diff)
    limit_h = H - d

    # Align After[0:limit] with Before[d:H]
    rect_a = ga[:limit_h, :]
    rect_b = gb[d:, :]

    # Create Binary Map (Matches vs Mismatches)
    diff = np.abs(rect_a - rect_b)
    binary_map = diff < PIXEL_MATCH_TOLERANCE

    # 5. Apply Geometric Constraints
    # The viewport must overlap the path [y_cursor - d, y_cursor]
    target_y_max = min(cy, limit_h - 1)
    target_y_min = max(0, cy - d)

    # Handle deep scrolling cursor edge case
    if target_y_min > target_y_max:
        target_y_min = target_y_max

    bbox = solve_constrained_box(
        binary_map,
        target_col=cx,
        target_y_range=(target_y_min, target_y_max),
        threshold_vertical=DENSITY_THRESHOLD_VERTICAL,
        threshold_horizontal=DENSITY_THRESHOLD_HORIZONTAL,
    )

    if not bbox:
        logging.info(
            f"Scroll detection failed: No viewport found meeting density thresholds "
            f"(scroll_distance={d}, cursor={cursor_pos}, y_range=({target_y_min}, {target_y_max}), "
            f"v_thresh={DENSITY_THRESHOLD_VERTICAL}, h_thresh={DENSITY_THRESHOLD_HORIZONTAL})"
        )
        return None

    r1, c1, r2, c2 = bbox
    h_box, w_box = r2 - r1, c2 - c1

    # Final Feasibility Check
    if h_box < MIN_VIEWPORT_HEIGHT or w_box < MIN_VIEWPORT_WIDTH:
        logging.info(
            f"Scroll detection failed: Viewport too small "
            f"(detected: {w_box}x{h_box}, required: {MIN_VIEWPORT_WIDTH}x{MIN_VIEWPORT_HEIGHT}, "
            f"scroll_distance={d})"
        )
        return None

    return {
        "scroll_distance": int(d),
        "after_bbox": (int(c1), int(r1), int(w_box), int(h_box)),
        "consensus_strength": float(consensus_ratio),
    }


@dataclass
class ScrollPhaseResult:
    """Result from a single scroll phase (probe or completion)."""

    detection: dict | None  # Result from detect_scroll(), None if detection failed
    img_before_pil: Any  # PIL Image before scroll
    img_after_pil: Any  # PIL Image after scroll
    img_after: np.ndarray  # Numpy array of after image (for chaining phases)

    @property
    def succeeded(self) -> bool:
        return self.detection is not None

    @property
    def scroll_distance(self) -> int:
        return self.detection["scroll_distance"] if self.detection else 0

    @property
    def viewport(self) -> tuple[int, int, int, int]:
        """Returns (x, y, width, height) of the viewport.

        The viewport is the full scrollable region, computed as the union of the
        overlap regions in the before and after images. Since content scrolls by
        scroll_distance pixels, viewport height = overlap height + scroll_distance.
        """
        if self.detection:
            x, y, w, h = self.detection["after_bbox"]
            return (x, y, w, h + self.detection["scroll_distance"])
        return (0, 0, 0, 0)


def perform_scroll_and_detect(
    img_before_pil,
    img_before: np.ndarray,
    scroll_amount: float,
    cursor_pos: tuple[float, float],
    screen_rect,
    wait_ms: int,
    phase_name: str = "",
) -> ScrollPhaseResult:
    """Perform a scroll, capture the result, and run scroll detection.

    Args:
        img_before_pil: PIL Image from before the scroll
        img_before: Numpy array of the before image
        scroll_amount: Wheel units to scroll
        cursor_pos: (x, y) cursor position
        screen_rect: Screen rectangle for capture
        wait_ms: Milliseconds to wait after scroll
        phase_name: Label for logging (e.g., "probe", "phase2")

    Returns:
        ScrollPhaseResult with detection results and images
    """
    actions.mouse_scroll(scroll_amount)
    actions.sleep(f"{wait_ms}ms")

    img_after_pil = screen.capture_rect(screen_rect, retina=False)
    img_after = np.array(img_after_pil)

    detection = detect_scroll(img_before, img_after, cursor_pos)

    # Log result with phase label
    if detection and phase_name:
        _, _, w, h = detection["after_bbox"]
        viewport_h = h + detection["scroll_distance"]
        logging.info(
            f"Scroll {phase_name}: {detection['scroll_distance']}px, "
            f"viewport={w}x{viewport_h}"
        )
    elif phase_name:
        logging.info(f"Scroll {phase_name}: detection failed")

    return ScrollPhaseResult(
        detection=detection,
        img_before_pil=img_before_pil,
        img_after_pil=img_after_pil,
        img_after=img_after,
    )


def save_scroll_screenshots(
    img_before_pil,
    img_after_pil,
    result: dict | None,
    cursor_pos: tuple[float, float],
    logging_dir: str,
) -> None:
    """Save before/after screenshots and metadata for scroll detection.

    Args:
        img_before_pil: PIL Image from screen.capture_rect() before scroll
        img_after_pil: PIL Image from screen.capture_rect() after scroll
        result: Dictionary from detect_scroll() or None if detection failed
        cursor_pos: Tuple (x, y) of cursor position during scroll
        logging_dir: Directory path from ocr_logging_dir setting
    """
    timestamp = time.time()

    # Determine file naming based on success/failure
    if result is None:
        file_prefix = f"scroll_failure_{timestamp:.2f}"
        metadata = {
            "status": "failure",
            "timestamp": timestamp,
            "cursor_position": {"x": cursor_pos[0], "y": cursor_pos[1]},
        }
    else:
        distance = result["scroll_distance"]
        x, y, w, h = result["after_bbox"]
        file_prefix = f"scroll_success_{distance}px_{timestamp:.2f}"
        metadata = {
            "status": "success",
            "timestamp": timestamp,
            "scroll_distance_px": distance,
            "cursor_position": {"x": cursor_pos[0], "y": cursor_pos[1]},
            "after_bbox": {"x": x, "y": y, "width": w, "height": h},
            # before_bbox is after_bbox shifted down by scroll_distance
            "before_bbox": {"x": x, "y": y + distance, "width": w, "height": h},
            "consensus_strength": result["consensus_strength"],
        }

    # Build file paths
    before_path = os.path.join(logging_dir, f"{file_prefix}.png")
    after_path = os.path.join(logging_dir, f"{file_prefix}_after.png")
    json_path = os.path.join(logging_dir, f"{file_prefix}.json")

    # Save screenshots (pattern from gaze_ocr._write_data)
    if hasattr(img_before_pil, "save"):
        img_before_pil.save(before_path)
        img_after_pil.save(after_path)
    else:
        img_before_pil.write_file(before_path)
        img_after_pil.write_file(after_path)

    # Save metadata as JSON
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Saved scroll screenshots to {file_prefix}*")


def get_debug_color(has_light_background: bool):
    return (
        settings.get("user.ocr_light_background_debug_color")
        if has_light_background
        else settings.get("user.ocr_dark_background_debug_color")
    )


def calculate_optimal_text_size(
    paint, text: str, bbox_width: float, bbox_height: float
) -> float:
    """Calculate optimal font size to fit text within bounding box using Paint.measure_text().

    Uses scaling approach with existing Paint object to avoid Font creation overhead.
    Maximizes font size while ensuring text fits within both width and height constraints.

    Args:
        paint: Skia Paint object to use for measurements
        text: Text string to size
        bbox_width: Target bounding box width
        bbox_height: Target bounding box height

    Returns:
        Optimal font size as float
    """
    # Store original text size to restore later
    original_size = paint.textsize

    # Use a reasonable base size for measurement
    base_size = 16
    paint.textsize = base_size

    try:
        width, bounds = paint.measure_text(text)

        if width <= 0 or bounds.height <= 0:
            return bbox_height  # Fallback to using bbox height as font size

        # Calculate scaling factors with small safety margin (95% of available space)
        safety_factor = 0.95
        width_scale = (bbox_width * safety_factor) / width
        height_scale = (bbox_height * safety_factor) / bounds.height

        # Use the more restrictive constraint
        scale = min(width_scale, height_scale)
        optimal_size = base_size * scale

        # Ensure minimum readable size
        optimal_size = max(optimal_size, 8)

        return optimal_size

    finally:
        # Always restore original text size
        paint.textsize = original_size


disambiguation_canvas = None
debug_canvas = None
scroll_indicator_canvas = None
ambiguous_matches: Optional[Sequence[gaze_ocr.CursorLocation]] = None
disambiguation_generator = None


def reset_disambiguation():
    global \
        ambiguous_matches, \
        disambiguation_generator, \
        disambiguation_canvas, \
        debug_canvas
    ctx.tags = []
    ambiguous_matches = None
    disambiguation_generator = None
    hide_canvas = disambiguation_canvas or debug_canvas
    if disambiguation_canvas:
        disambiguation_canvas.close()
    disambiguation_canvas = None
    if debug_canvas:
        debug_canvas.close()
    debug_canvas = None
    if hide_canvas:
        # Ensure that the canvas doesn't interfere with subsequent screenshots.
        actions.sleep("10ms")


def show_disambiguation():
    global ambiguous_matches, disambiguation_canvas

    contents = gaze_ocr_controller.latest_screen_contents()

    def on_draw(c):
        assert ambiguous_matches
        debug_color = get_debug_color(has_light_background(contents.screenshot))
        nearest = gaze_ocr_controller.find_nearest_cursor_location(ambiguous_matches)
        used_locations = set()
        for i, match in enumerate(ambiguous_matches):
            if nearest == match:
                c.paint.typeface = Typeface.from_name(
                    "", Fontstyle.new(weight=700, width=5)
                )
            else:
                c.paint.typeface = ""
            c.paint.textsize = max(round(match.text_height * 2), 15)
            c.paint.style = c.paint.Style.FILL
            c.paint.color = debug_color
            location = (match.visual_coordinates[0], match.visual_coordinates[1])
            # TODO: Check for nearby used locations, not just identical.
            while location in used_locations:
                # Shift right.
                location = (location[0] + match.text_height, location[1])
            used_locations.add(location)
            c.draw_text(str(i + 1), *location)
        setting_ocr_disambiguation_display_seconds = settings.get(
            "user.ocr_disambiguation_display_seconds"
        )
        if setting_ocr_disambiguation_display_seconds and disambiguation_canvas:
            current_canvas = disambiguation_canvas

            def timeout_disambiguation():
                global disambiguation_canvas
                if disambiguation_canvas and disambiguation_canvas == current_canvas:
                    reset_disambiguation()

            cron.after(
                f"{setting_ocr_disambiguation_display_seconds}s",
                timeout_disambiguation,
            )

    ctx.tags = ["user.gaze_ocr_disambiguation"]
    if disambiguation_canvas:
        disambiguation_canvas.close()
    rect = screen_ocr.to_rect(contents.bounding_box)
    screen_rect = screen.main().rect
    # If rect is approximately equal to screen.main().rect, use Canvas.from_screen to
    # avoid Windows bug where the screen is blacked out.
    # https://github.com/wolfmanstout/talon-gaze-ocr/issues/47
    if (
        abs(rect.x - screen_rect.x) < 1
        and abs(rect.y - screen_rect.y) < 1
        and abs(rect.width - screen_rect.width) < 1
        and abs(rect.height - screen_rect.height) < 1
    ):
        disambiguation_canvas = Canvas.from_screen(screen.main())
    else:
        disambiguation_canvas = Canvas.from_rect(rect)
    disambiguation_canvas.register("draw", on_draw)
    disambiguation_canvas.freeze()


def begin_generator(generator):
    global ambiguous_matches, disambiguation_generator, disambiguation_canvas
    reset_disambiguation()
    try:
        ambiguous_matches = next(generator)
        disambiguation_generator = generator
        show_disambiguation()
    except StopIteration:
        # Execution completed without need for disambiguation.
        pass


def move_cursor_to_word_generator(text: TimestampedText, disambiguate: bool = True):
    result = yield from gaze_ocr_controller.move_cursor_to_words_generator(
        text.text,
        disambiguate=disambiguate,
        time_range=(text.start, text.end),
        click_offset_right=lambda: settings.get("user.ocr_click_offset_right"),
    )
    if not result:
        actions.user.show_ocr_overlay_for_query("text", f"{text.text}")
        raise RuntimeError(f'Unable to find: "{text}"')


def move_text_cursor_to_word_generator(
    text: TimestampedText,
    position: str,
    hold_shift: bool = False,
):
    result = yield from gaze_ocr_controller.move_text_cursor_to_words_generator(
        text.text,
        disambiguate=True,
        cursor_position=position,
        time_range=(text.start, text.end),
        click_offset_right=lambda: settings.get("user.ocr_click_offset_right"),
        hold_shift=hold_shift,
    )
    if not result:
        actions.user.show_ocr_overlay_for_query("text", f"{text.text}")
        raise RuntimeError(f'Unable to find: "{text}"')


def move_text_cursor_to_longest_prefix_generator(
    text: TimestampedText, position: str, hold_shift: bool = False
):
    (
        locations,
        prefix_length,
    ) = yield from gaze_ocr_controller.move_text_cursor_to_longest_prefix_generator(
        text.text,
        disambiguate=True,
        cursor_position=position,
        time_range=(text.start, text.end),
        click_offset_right=lambda: settings.get("user.ocr_click_offset_right"),
        hold_shift=hold_shift,
    )
    if not locations:
        actions.user.show_ocr_overlay_for_query("text", f"{text.text}")
        raise RuntimeError(f'Unable to find: "{text}"')
    return prefix_length


def move_text_cursor_to_longest_suffix_generator(
    text: TimestampedText, position: str, hold_shift: bool = False
):
    (
        locations,
        prefix_length,
    ) = yield from gaze_ocr_controller.move_text_cursor_to_longest_suffix_generator(
        text.text,
        disambiguate=True,
        cursor_position=position,
        time_range=(text.start, text.end),
        click_offset_right=lambda: settings.get("user.ocr_click_offset_right"),
        hold_shift=hold_shift,
    )
    if not locations:
        actions.user.show_ocr_overlay_for_query("text", f"{text.text}")
        raise RuntimeError(f'Unable to find: "{text}"')
    return prefix_length


def move_text_cursor_to_difference(text: TimestampedText):
    result = yield from gaze_ocr_controller.move_text_cursor_to_difference_generator(
        text.text,
        disambiguate=True,
        time_range=(text.start, text.end),
        click_offset_right=lambda: settings.get("user.ocr_click_offset_right"),
    )
    if not result:
        actions.user.show_ocr_overlay_for_query("text", f"{text.text}")
        raise RuntimeError(f'Unable to find: "{text}"')
    return result


def select_text_generator(
    start: TimestampedText,
    end: Optional[TimestampedText] = None,
    for_deletion: bool = False,
    after_start: bool = False,
    before_end: bool = False,
):
    start_text = start.text
    end_text = end.text if end else None
    result = yield from gaze_ocr_controller.select_text_generator(
        start_text,
        disambiguate=True,
        end_words=end_text,
        for_deletion=for_deletion,
        start_time_range=(start.start, start.end),
        end_time_range=(end.start, end.end) if end else None,
        click_offset_right=lambda: settings.get("user.ocr_click_offset_right"),
        after_start=after_start,
        before_end=before_end,
        select_pause_seconds=lambda: settings.get("user.ocr_select_pause_seconds"),
    )
    if not result:
        actions.user.show_ocr_overlay_for_query(
            "text", f"{start.text}...{end.text if end else None}"
        )
        raise RuntimeError(f'Unable to select "{start}" to "{end}"')


def select_matching_text_generator(text: TimestampedText):
    result = yield from gaze_ocr_controller.select_matching_text_generator(
        text.text,
        disambiguate=True,
        time_range=(text.start, text.end),
        click_offset_right=lambda: settings.get("user.ocr_click_offset_right"),
        select_pause_seconds=lambda: settings.get("user.ocr_select_pause_seconds"),
    )
    if not result:
        actions.user.show_ocr_overlay_for_query("text", f"{text.text}")
        raise RuntimeError(f'Unable to find: "{text}"')


def select_text_range_generator(
    text_range: TextRange,
    for_deletion: bool,
):
    if not text_range.start:
        assert text_range.end
        yield from move_text_cursor_to_word_generator(
            text_range.end,
            position="before" if text_range.before_end else "after",
            hold_shift=True,
        )
    else:
        yield from select_text_generator(
            text_range.start,
            text_range.end,
            for_deletion,
            after_start=text_range.after_start,
            before_end=text_range.before_end,
        )


def context_sensitive_insert(text: str):
    if settings.get("user.context_sensitive_dictation"):
        actions.user.dictation_insert(text)
    else:
        # Use the default insert because the dictation context is likely wrong.
        actions.insert(text)


@mod.action_class
class GazeOcrActions:
    def focus_at(x: int, y: int):
        """Focus the window at the given coordinates."""
        # Default implementation is a no-op Mac has a specific implementation that uses
        # either ui.window_at() or ui.element_at()
        # TODO: Implement for Windows/Linux once ui.window_at() is fixed on those
        # platforms Need to have at least one action for Talon to recognize this as
        # implemented
        actions.sleep("0ms")

    #
    # Actions related to the eye tracker.
    #

    def connect_ocr_eye_tracker():
        """Connects eye tracker to OCR."""
        tracker.connect()

    def disconnect_ocr_eye_tracker():
        """Disconnects eye tracker from OCR."""
        tracker.disconnect()

    #
    # Actions related to the UI.
    #

    def show_ocr_overlay(
        type: str, near: Optional[TimestampedText] = None, refresh: bool = True
    ):
        """Displays OCR debug overlay over primary screen, refreshing the OCR nearby
        where the user is looking by default.

        If the near parameter is provided, refreshes OCR nearby where the user is
        looking when they spoke the near parameter."""
        reset_disambiguation()
        if refresh:
            if near:
                gaze_ocr_controller.read_nearby((near.start, near.end))
            else:
                gaze_ocr_controller.read_nearby()
        actions.user.show_ocr_overlay_for_query(type, "", True)

    def show_scroll_indicator(
        line_y: int,
        line_x_start: int = 0,
        line_x_end: Optional[int] = None,
        viewport_bbox: Optional[tuple[int, int, int, int]] = None,
        scroll_distance: Optional[int] = None,
    ):
        """Display fading horizontal line at scroll boundary.

        Args:
            line_y: Y-coordinate for the line
            line_x_start: X-coordinate where line starts (default: 0)
            line_x_end: X-coordinate where line ends (default: screen width)
            viewport_bbox: Optional (x, y, w, h) to show viewport outline in debug mode
            scroll_distance: Optional scroll distance to show in debug label
        """
        global scroll_indicator_canvas

        # Close existing canvas if any (handles rapid scrolls)
        if scroll_indicator_canvas:
            scroll_indicator_canvas.close()
            scroll_indicator_canvas = None

        screen_rect = screen.main().rect
        if line_x_end is None:
            line_x_end = screen_rect.width

        # Type narrowing: line_x_end is guaranteed to be int here
        assert line_x_end is not None
        line_x_end_val = line_x_end

        start_time = time.time()
        # Use longer fade for debug mode
        fade_duration = (
            5.0
            if viewport_bbox
            else settings.get("user.ocr_scroll_indicator_fade_seconds")
        )

        def on_draw(c):
            elapsed = time.time() - start_time
            alpha = max(0.0, 1.0 - (elapsed / fade_duration))

            if alpha <= 0:
                return

            alpha_byte = int(alpha * 255)

            # Draw viewport box in green if in debug mode
            if viewport_bbox:
                vx, vy, vw, vh = viewport_bbox
                c.paint.style = c.paint.Style.STROKE
                c.paint.stroke_width = 3.0
                c.paint.color = f"00FF00{alpha_byte:02X}"  # Green with alpha
                c.draw_rect(rect.Rect(x=vx, y=vy, width=vw, height=vh))

                # Draw label
                c.paint.style = c.paint.Style.FILL
                c.paint.textsize = 20
                label = (
                    f"VIEWPORT (scroll={scroll_distance}px)"
                    if scroll_distance
                    else "VIEWPORT"
                )
                c.draw_text(label, vx, vy - 5)

            # Draw scroll seam line in light blue
            color = settings.get("user.ocr_scroll_indicator_color")
            c.paint.color = f"{color}{alpha_byte:02X}"
            c.paint.style = c.paint.Style.STROKE
            c.paint.stroke_width = 2.0

            try:
                c.draw_line(line_x_start, line_y, line_x_end_val, line_y)
            except AttributeError:
                line_width = line_x_end_val - line_x_start
                c.draw_rect(
                    rect.Rect(x=line_x_start, y=line_y, width=line_width, height=2)
                )

        scroll_indicator_canvas = Canvas.from_screen(screen.main())
        scroll_indicator_canvas.blocks_mouse = False
        scroll_indicator_canvas.allows_capture = (
            False  # Prevent canvas from appearing in screenshots
        )
        scroll_indicator_canvas.register("draw", on_draw)

        # Schedule cleanup - capture canvas reference to avoid race condition
        canvas_to_cleanup = scroll_indicator_canvas

        def cleanup_scroll_canvas():
            global scroll_indicator_canvas
            # Only close if it's still the same canvas we created
            if scroll_indicator_canvas is canvas_to_cleanup:
                canvas_to_cleanup.close()
                scroll_indicator_canvas = None

        # Convert to milliseconds since cron.after doesn't accept float seconds
        cleanup_delay_ms = int((fade_duration + 0.1) * 1000)
        cron.after(f"{cleanup_delay_ms}ms", cleanup_scroll_canvas)

    def show_ocr_overlay_for_query(
        type: str, query: str = "", persistent: bool = False
    ):
        """Display overlay over primary screen, displaying the query."""
        global debug_canvas
        if debug_canvas:
            debug_canvas.close()
            debug_canvas = None
        contents = gaze_ocr_controller.latest_screen_contents()

        contents_rect = screen_ocr.to_rect(contents.bounding_box)

        # Capture start time for synchronized fading
        start_time = time.perf_counter()

        def on_draw(c):
            light_bg = has_light_background(contents.screenshot)
            debug_color = get_debug_color(light_bg)

            if type == "text":
                # Text overlay needs opaque background with fading to be readable
                # Fade timing configuration
                fade_in_duration = 0.5  # seconds to fade in
                hold_duration = 0.5  # seconds to hold at full opacity
                fade_out_duration = 0.5  # seconds to fade out
                total_cycle_time = (
                    fade_in_duration + hold_duration + fade_out_duration
                )  # 1.5s total

                elapsed_time = time.perf_counter() - start_time
                cycle_time = elapsed_time % total_cycle_time

                # Calculate alpha (0.0 to 1.0) based on cycle position
                timeout = settings.get("user.ocr_debug_display_seconds")
                if not persistent and elapsed_time > timeout:
                    # Ensure the animation does not overrun.
                    alpha = 0.0
                elif cycle_time < fade_in_duration:
                    # Fade in: 0 to 1 over fade_in_duration
                    alpha = cycle_time / fade_in_duration
                elif cycle_time < fade_in_duration + hold_duration:
                    # Hold: stay at 1.0
                    alpha = 1.0
                else:
                    # Fade out: 1 to 0 over fade_out_duration
                    fade_start = fade_in_duration + hold_duration
                    alpha = 1.0 - ((cycle_time - fade_start) / fade_out_duration)

                # Clamp alpha to valid range
                alpha = max(0.0, min(1.0, alpha))

                bg_color = "FFFFFF" if light_bg else "000000"
                alpha_byte = int(alpha * 255)

                # Draw opaque background over the contents area with alpha
                c.paint.style = c.paint.Style.FILL
                c.paint.color = f"{bg_color}{alpha_byte:02X}"
                c.draw_rect(contents_rect)

                # Show bounding box with alpha
                c.paint.style = c.paint.Style.STROKE
                c.paint.color = f"{debug_color}{alpha_byte:02X}"
                c.draw_rect(contents_rect)

                # Draw text with alpha
                for line in contents.result.lines:
                    for word in line.words:
                        c.paint.typeface = ""
                        c.paint.textsize = calculate_optimal_text_size(
                            c.paint, word.text, word.width, word.height
                        )
                        c.paint.style = c.paint.Style.FILL
                        c.paint.color = f"{debug_color}{alpha_byte:02X}"
                        # Position baseline at ~80% down from top of OCR bounding box
                        c.draw_text(
                            word.text, word.left, word.top + (word.height * 0.8)
                        )

            elif type == "boxes":
                # Box outlines don't interfere with text, so no background or fading needed
                # Show bounding box
                c.paint.style = c.paint.Style.STROKE
                c.paint.color = debug_color
                c.draw_rect(contents_rect)

                # Draw word boxes
                for line in contents.result.lines:
                    for word in line.words:
                        c.paint.style = c.paint.Style.STROKE
                        c.paint.color = debug_color
                        c.draw_rect(
                            rect.Rect(
                                x=word.left,
                                y=word.top,
                                width=word.width,
                                height=word.height,
                            )
                        )

            else:
                raise RuntimeError(f"Type not recognized: {type}")
            if debug_canvas and not persistent:
                current_canvas = debug_canvas

                def timeout_debug_canvas():
                    global debug_canvas
                    if debug_canvas and debug_canvas == current_canvas:
                        debug_canvas.close()
                        debug_canvas = None

                cron.after(
                    f"{settings.get('user.ocr_debug_display_seconds')}s",
                    timeout_debug_canvas,
                )

        # Increased size slightly for canvas to ensure everything will be inside canvas
        canvas_rect = contents_rect.copy()
        center = canvas_rect.center
        canvas_rect.height += 100
        canvas_rect.width += 100
        canvas_rect.center = center

        debug_canvas = Canvas.from_rect(canvas_rect)
        debug_canvas.blocks_mouse = True
        debug_canvas.register("draw", on_draw)

        def on_mouse(e: MouseEvent):
            global debug_canvas
            if e.event == "mousedown" and debug_canvas:  # Any mouse button click
                debug_canvas.close()
                debug_canvas = None

        debug_canvas.register("mouse", on_mouse)

    def hide_ocr_overlay():
        """Hide any visible OCR overlay."""
        global debug_canvas
        if debug_canvas:
            debug_canvas.close()
            debug_canvas = None

    def choose_gaze_ocr_option(index: int):
        """Disambiguate with the provided index."""
        global ambiguous_matches, disambiguation_generator, disambiguation_canvas
        if (
            not ambiguous_matches
            or not disambiguation_generator
            or not disambiguation_canvas
        ):
            assert not ambiguous_matches
            assert not disambiguation_generator
            assert not disambiguation_canvas
            raise RuntimeError("Disambiguation not active")
        ctx.tags = []
        disambiguation_canvas.close()
        disambiguation_canvas = None
        # Give the canvas a moment to disappear so it doesn't interfere with subsequent screenshots.
        actions.sleep("10ms")
        match = ambiguous_matches[index - 1]
        try:
            ambiguous_matches = disambiguation_generator.send(match)
            show_disambiguation()
        except StopIteration:
            # Execution completed successfully.
            reset_disambiguation()

    def hide_gaze_ocr_options():
        """Hide the disambiguation UI."""
        reset_disambiguation()

    #
    # Actions operating on the gaze point.
    #

    def move_cursor_to_gaze_point(offset_right: int = 0, offset_down: int = 0):
        """Moves mouse cursor to gaze location."""
        tracker.move_to_gaze_point((offset_right, offset_down))

    def scroll_down_with_visualization(amount: float = 1.0):
        """Scroll down and show visual indicator of content movement.

        Uses a two-phase approach:
        1. Probe scroll: Small scroll to detect viewport size and calibrate scroll ratio
        2. Calibrated scroll: Complete remaining scroll based on detected viewport
        """
        global scroll_indicator_canvas

        if not settings.get("user.ocr_scroll_indicator_enabled"):
            actions.user.mouse_scroll_down(amount)
            return

        # Close any existing canvas to prevent screenshot interference
        if scroll_indicator_canvas:
            scroll_indicator_canvas.close()
            scroll_indicator_canvas = None
            actions.sleep("10ms")

        # Configuration
        screen_rect = screen.main().rect
        wait_ms = settings.get("user.ocr_scroll_wait_ms")
        viewport_fraction = settings.get("user.ocr_scroll_viewport_fraction")
        logging_dir = settings.get("user.ocr_logging_dir")
        PROBE_SCROLL_AMOUNT = 50  # Wheel units (not pixels)

        # Capture initial state
        img_before_pil = screen.capture_rect(screen_rect, retina=False)
        img_before = np.array(img_before_pil)
        actions.sleep("5ms")
        cursor_pos = (actions.mouse_x(), actions.mouse_y())

        # Phase 1: Probe scroll to detect viewport and calibrate
        probe = perform_scroll_and_detect(
            img_before_pil,
            img_before,
            PROBE_SCROLL_AMOUNT,
            cursor_pos,
            screen_rect,
            wait_ms,
            phase_name="probe",
        )

        if not probe.succeeded:
            return

        scroll_ratio = probe.scroll_distance / PROBE_SCROLL_AMOUNT

        # Phase 2: Calculate and execute remaining scroll
        probe_viewport_h = probe.viewport[3]  # height
        total_desired_pixels = probe_viewport_h * viewport_fraction * amount
        remaining_pixels = max(0, total_desired_pixels - probe.scroll_distance)
        phase2 = None

        if remaining_pixels > 0 and scroll_ratio > 0:
            remaining_wheel_units = remaining_pixels / scroll_ratio

            phase2 = perform_scroll_and_detect(
                probe.img_after_pil,
                probe.img_after,
                remaining_wheel_units,
                cursor_pos,
                screen_rect,
                wait_ms,
                phase_name="phase2",
            )

            if phase2.succeeded:
                total_scroll = probe.scroll_distance + phase2.scroll_distance
                vis_x, vis_y, vis_w, vis_h = phase2.viewport
            else:
                # Detection failed (e.g., hit bottom of content) - use estimate
                total_scroll = probe.scroll_distance + remaining_pixels
                vis_x, vis_y, vis_w, vis_h = probe.viewport
        else:
            # No second scroll needed
            total_scroll = probe.scroll_distance
            vis_x, vis_y, vis_w, vis_h = probe.viewport

        # Line shows where content from before the scroll ends
        line_y = vis_y + vis_h - total_scroll

        logging.info(f"Scroll complete: total={total_scroll:.0f}px")

        # Show visualization using the viewport from whichever detection we're using
        debug_mode = settings.get("user.ocr_scroll_debug_mode")
        actions.user.show_scroll_indicator(
            int(line_y),
            vis_x,
            vis_x + vis_w,
            viewport_bbox=(vis_x, vis_y, vis_w, vis_h) if debug_mode else None,
            scroll_distance=int(total_scroll) if debug_mode else None,
        )

        # Save screenshots after visualization is shown (deferred to avoid blocking)
        if logging_dir:
            save_scroll_screenshots(
                img_before_pil,
                probe.img_after_pil,
                probe.detection,
                cursor_pos,
                logging_dir,
            )
            if phase2 is not None:
                save_scroll_screenshots(
                    probe.img_after_pil,
                    phase2.img_after_pil,
                    phase2.detection,
                    cursor_pos,
                    logging_dir,
                )

    #
    # Actions operating on a single point within onscreen text.
    #

    def move_cursor_to_word(text: TimestampedText):
        """Moves cursor to onscreen word."""
        begin_generator(move_cursor_to_word_generator(text))

    def move_text_cursor_to_word(
        text: TimestampedText,
        position: str,
    ):
        """Moves text cursor near onscreen word."""
        begin_generator(move_text_cursor_to_word_generator(text, position))

    def insert_adjacent_to_text(
        find_text: TimestampedText, position: str, insertion_text: str
    ):
        """Insert text adjacent to onscreen text."""

        def run():
            yield from move_text_cursor_to_word_generator(
                find_text,
                position,
            )
            context_sensitive_insert(insertion_text)

        begin_generator(run())

    def move_cursor_to_text_and_do(
        text: TimestampedText, action: Callable[[], None], disambiguate: bool = True
    ) -> None:
        """Moves cursor to onscreen word and performs an action."""

        def run():
            yield from move_cursor_to_word_generator(text, disambiguate)
            action()

        begin_generator(run())

    def click_text(text: TimestampedText):
        """Click on the provided on-screen text."""
        actions.user.move_cursor_to_text_and_do(text, lambda: actions.mouse_click())

    def click_text_without_disambiguation(text: TimestampedText):
        """Click on the provided on-screen text, choosing the best match if multiple are
        found."""
        actions.user.move_cursor_to_text_and_do(
            text, lambda: actions.mouse_click(), disambiguate=False
        )

    def double_click_text(text: TimestampedText):
        """Double-lick on the provided on-screen text."""

        def double_click() -> None:
            actions.mouse_click()
            actions.mouse_click()

        actions.user.move_cursor_to_text_and_do(text, double_click)

    def right_click_text(text: TimestampedText):
        """Right-click on the provided on-screen text."""
        actions.user.move_cursor_to_text_and_do(text, lambda: actions.mouse_click(1))

    def middle_click_text(text: TimestampedText):
        """Middle-click on the provided on-screen text."""
        actions.user.move_cursor_to_text_and_do(text, lambda: actions.mouse_click(2))

    def modifier_click_text(modifier: str, text: TimestampedText):
        """Control-click on the provided on-screen text."""

        def click_with_modifier() -> None:
            actions.key(f"{modifier}:down")
            actions.mouse_click()
            actions.key(f"{modifier}:up")

        actions.user.move_cursor_to_text_and_do(text, click_with_modifier)

    def change_text_homophone(text: TimestampedText):
        """Switch the on-screen text to a different homophone."""

        def change_homophone() -> None:
            actions.mouse_click()
            actions.edit.select_word()
            actions.user.homophones_show_selection()

        actions.user.move_cursor_to_text_and_do(text, change_homophone)

    #
    # Actions operating on a selection of onscreen text.
    #

    def select_text_and_do(
        text_range: TextRange,
        for_deletion: bool,
        ocr_modifier: str,
        action: Callable[[], None],
    ) -> None:
        """Selects text and performs an action."""
        if ocr_modifier not in _OCR_MODIFIERS:
            raise ValueError(f"Modifier not supported: {ocr_modifier}")

        def run():
            yield from select_text_range_generator(text_range, for_deletion)
            _OCR_MODIFIERS[ocr_modifier]()
            action()

        begin_generator(run())

    def perform_ocr_action(
        ocr_action: str,
        ocr_modifier: str,
        text_range: TextRange,
    ) -> None:
        """Selects text and performs a known action by name."""
        if ocr_action not in _OCR_ACTIONS:
            raise ValueError(f"Action not supported: {ocr_action}")

        actions.user.select_text_and_do(
            text_range=text_range,
            for_deletion=(ocr_action in ("cut", "delete_with_whitespace")),
            ocr_modifier=ocr_modifier,
            action=_OCR_ACTIONS[ocr_action],
        )

    def replace_text(ocr_modifier: str, text_range: TextRange, replacement: str):
        """Replaces onscreen text."""
        actions.user.select_text_and_do(
            text_range=text_range,
            for_deletion=settings.get("user.context_sensitive_dictation"),
            ocr_modifier=ocr_modifier,
            action=lambda: context_sensitive_insert(replacement),
        )

    #
    # Actions providing natural text editing.
    #

    def append_text(text: TimestampedText):
        """Finds onscreen text that matches the beginning of the provided text and
        appends the rest to it."""

        def run():
            prefix_length = yield from move_text_cursor_to_longest_prefix_generator(
                text, "after"
            )
            insertion_text = text.text[prefix_length:]
            context_sensitive_insert(insertion_text)

        begin_generator(run())

    def prepend_text(text: TimestampedText):
        """Finds onscreen text that matches the end of the provided text and
        prepends the rest to it."""

        def run():
            suffix_length = yield from move_text_cursor_to_longest_suffix_generator(
                text, "before"
            )
            insertion_text = text.text[:-suffix_length]
            context_sensitive_insert(insertion_text)

        begin_generator(run())

    def insert_text_difference(text: TimestampedText):
        """Finds onscreen text that matches the start and/or end of the provided text
        and inserts the difference."""

        def run():
            start, end = yield from move_text_cursor_to_difference(text)
            insertion_text = text.text[start:end]
            context_sensitive_insert(insertion_text)

        begin_generator(run())

    def revise_text(text: TimestampedText):
        """Finds onscreen text that matches the beginning and end of the provided text
        and replaces it."""

        def run():
            yield from select_matching_text_generator(text)
            insertion_text = text.text
            context_sensitive_insert(insertion_text)

        begin_generator(run())

    def revise_text_starting_with(text: TimestampedText):
        """Finds onscreen text that matches the beginning of the provided text
        and replaces it until the caret."""

        def run():
            try:
                yield from move_text_cursor_to_longest_prefix_generator(
                    text, "before", hold_shift=True
                )
            except RuntimeError as e:
                # Keep going so the user doesn't lose the dictated text.
                print(e)
            insertion_text = text.text
            context_sensitive_insert(insertion_text)

        begin_generator(run())

    def revise_text_ending_with(text: TimestampedText):
        """Finds onscreen text that matches the end of the provided text and
        replaces it from the caret."""

        def run():
            try:
                yield from move_text_cursor_to_longest_suffix_generator(
                    text, "after", hold_shift=True
                )
            except RuntimeError as e:
                # Keep going so the user doesn't lose the dictated text.
                print(e)
            insertion_text = text.text
            context_sensitive_insert(insertion_text)

        begin_generator(run())


def focus_element_window(element) -> bool:
    """Focuses the window containing the accessibility element."""
    try:
        ax_window = element.AXWindow
    except AttributeError:
        # Assume the current element is the window.
        ax_window = element

    try:
        ax_app = ax_window.AXParent
    except AttributeError:
        return False

    if ax_app.AXRole != "AXApplication":
        return False

    # Raise the window to the top.
    try:
        ax_window.perform("AXRaise")
    except Exception:
        return False

    # Focus the application. Check if it is already focused first to avoid
    # unnecessary impact on window ordering.
    if not ax_app.AXFrontmost:
        ax_app.AXFrontmost = True
    return True


# Mac-specific implementation that focuses windows at coordinates
ctx_mac = Context()
ctx_mac.matches = "os: mac"


@ctx_mac.action_class("user")
class MacGazeOcrActions:
    def focus_at(x: int, y: int):
        """Focus the window at the given coordinates on Mac."""
        use_window_at = settings.get("user.ocr_use_window_at_api")
        if use_window_at:
            # Attempt to turn off HUD if talon_hud is installed.
            try:
                actions.user.hud_set_visibility(False, pause_seconds=0)
            except Exception:
                pass
            # Use window_at API (requires beta Talon)
            try:
                window = ui.window_at(x, y)
            except (RuntimeError, AttributeError):
                # No window at this position
                logging.debug(f"No window at position ({x}, {y}); skipping focus.")
                return
            finally:
                # Attempt to turn on HUD if talon_hud is installed.
                try:
                    actions.user.hud_set_visibility(True, pause_seconds=0)
                except Exception:
                    pass

            # Focus the window if not already active
            if ui.active_window() != window:
                if window.title == "Notification Center":
                    app.notify(
                        "Unable to focus window with notifications active. Please dismiss notifications."
                    )
                    return
                window.focus()
                start_time = time.perf_counter()
                while ui.active_window() != window:
                    if time.perf_counter() - start_time > 1:
                        logging.warning(
                            f"Can't focus window: {window.title}. Proceeding anyway."
                        )
                        break
                    actions.sleep(0.1)
        else:
            # Use element_at API (works on older Talon versions)
            try:
                element = ui.element_at(x, y)
            except RuntimeError:
                logging.debug(f"No element at position ({x}, {y}); skipping focus.")
                return

            if not focus_element_window(element):
                # This can happen when clicking on the desktop or menu bar.
                logging.debug(
                    f"Unable to focus window for element {element}; skipping focus."
                )
