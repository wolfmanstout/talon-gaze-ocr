"""Tests comparing debug_scroll.py and scroll_detection.py implementations.

These tests verify that the debug visualization tool produces identical
algorithmic results to the production scroll detection code.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Add paths for imports
sys.path.insert(
    0, str(Path(__file__).parent.parent / ".tools")
)  # .tools (for debug_scroll)
sys.path.insert(
    0, str(Path(__file__).parent.parent)
)  # project root (for scroll_detection)

from debug_scroll import detect_scroll_debug

from scroll_detection import BoundingBox, detect_scroll


def get_test_data_files():
    """Discover all test data JSON files."""
    data_dir = Path(__file__).parent / "data"
    json_files = sorted(data_dir.glob("scroll_success_*.json"))
    return json_files


def load_test_case(json_path: Path):
    """Load a test case from JSON file and associated images."""
    base = str(json_path).rsplit(".json", 1)[0]
    before_path = base + "_before.png"
    after_path = base + "_after.png"

    if not Path(before_path).exists() or not Path(after_path).exists():
        return None

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

    img_before = np.array(Image.open(before_path))
    img_after = np.array(Image.open(after_path))

    return {
        "json_path": json_path,
        "img_before": img_before,
        "img_after": img_after,
        "cursor_pos": cursor_pos,
        "existing_viewport": existing_viewport,
    }


# Collect test cases
TEST_CASES = []
for json_path in get_test_data_files():
    case = load_test_case(json_path)
    if case is not None:
        TEST_CASES.append(case)


@pytest.fixture(params=TEST_CASES, ids=lambda c: c["json_path"].stem)
def test_case(request):
    """Parametrized fixture providing each test case."""
    return request.param


def make_debug_params(existing_viewport=None):
    """Create params dict for debug implementation.

    Only pass existing_viewport if provided; all other parameters use defaults
    matching production scroll_detection.py.
    """
    params = {}
    if existing_viewport is not None:
        params["existing_viewport"] = existing_viewport
    return params


class TestDebugScrollParity:
    """Test that debug and production implementations produce identical results."""

    def test_scroll_distance_matches(self, test_case):
        """Verify scroll distance is identical between implementations."""
        img_before = test_case["img_before"]
        img_after = test_case["img_after"]
        cursor_pos = test_case["cursor_pos"]
        existing_viewport = test_case["existing_viewport"]

        # Run production implementation
        existing_bbox = (
            BoundingBox.from_tuple(existing_viewport) if existing_viewport else None
        )
        prod_result = detect_scroll(
            img_before, img_after, cursor_pos, existing_viewport=existing_bbox
        )

        # Run debug implementation
        debug_result = detect_scroll_debug(
            img_before, img_after, cursor_pos, make_debug_params(existing_viewport)
        )

        # Both should succeed or both should fail
        prod_success = prod_result is not None
        debug_success = debug_result.failure_phase is None

        assert prod_success == debug_success, (
            f"Implementation mismatch: production={'success' if prod_success else 'fail'}, "
            f"debug={'success' if debug_success else f'fail at phase {debug_result.failure_phase}'}"
        )

        if prod_success:
            assert prod_result.scroll_distance == debug_result.scroll_distance, (
                f"Scroll distance mismatch: production={prod_result.scroll_distance}, "
                f"debug={debug_result.scroll_distance}"
            )

    def test_viewport_matches(self, test_case):
        """Verify refined viewport is identical between implementations."""
        img_before = test_case["img_before"]
        img_after = test_case["img_after"]
        cursor_pos = test_case["cursor_pos"]
        existing_viewport = test_case["existing_viewport"]

        existing_bbox = (
            BoundingBox.from_tuple(existing_viewport) if existing_viewport else None
        )
        prod_result = detect_scroll(
            img_before, img_after, cursor_pos, existing_viewport=existing_bbox
        )

        debug_result = detect_scroll_debug(
            img_before, img_after, cursor_pos, make_debug_params(existing_viewport)
        )

        if prod_result is not None and debug_result.refined_viewport is not None:
            prod_vp = prod_result.viewport.as_tuple()
            debug_vp = debug_result.refined_viewport

            assert prod_vp == debug_vp, (
                f"Viewport mismatch: production={prod_vp}, debug={debug_vp}"
            )

    def test_after_bbox_matches(self, test_case):
        """Verify after_bbox is identical between implementations."""
        img_before = test_case["img_before"]
        img_after = test_case["img_after"]
        cursor_pos = test_case["cursor_pos"]
        existing_viewport = test_case["existing_viewport"]

        existing_bbox = (
            BoundingBox.from_tuple(existing_viewport) if existing_viewport else None
        )
        prod_result = detect_scroll(
            img_before, img_after, cursor_pos, existing_viewport=existing_bbox
        )

        debug_result = detect_scroll_debug(
            img_before, img_after, cursor_pos, make_debug_params(existing_viewport)
        )

        if prod_result is not None and debug_result.after_bbox is not None:
            prod_bbox = prod_result.after_bbox.as_tuple()
            debug_bbox = debug_result.after_bbox

            assert prod_bbox == debug_bbox, (
                f"After bbox mismatch: production={prod_bbox}, debug={debug_bbox}"
            )

    def test_initial_viewport_matches(self, test_case):
        """Verify initial viewport estimation matches (when not using existing viewport)."""
        img_before = test_case["img_before"]
        img_after = test_case["img_after"]
        cursor_pos = test_case["cursor_pos"]
        existing_viewport = test_case["existing_viewport"]

        # Skip if using existing viewport (Phase 1 is skipped)
        if existing_viewport is not None:
            pytest.skip("Test case uses existing viewport, Phase 1 skipped")

        from scroll_detection import estimate_initial_viewport

        # Convert to grayscale (same as both implementations do internally)
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

        cx, cy = int(cursor_pos[0]), int(cursor_pos[1])

        # Compute same-position diff (as detect_scroll does)
        same_pos_diff = np.abs(ga - gb)

        # Run production Phase 1
        prod_viewport = estimate_initial_viewport((cx, cy), same_pos_diff)

        # Run debug implementation
        debug_result = detect_scroll_debug(
            img_before, img_after, cursor_pos, make_debug_params(None)
        )

        if prod_viewport is not None and debug_result.initial_viewport is not None:
            assert prod_viewport == debug_result.initial_viewport, (
                f"Initial viewport mismatch: production={prod_viewport}, "
                f"debug={debug_result.initial_viewport}"
            )
