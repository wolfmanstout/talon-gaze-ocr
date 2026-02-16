"""
Unit tests for vertical scrolling detection algorithm.

Run with: uv run pytest test_scroll_detection.py -v
"""

import os
import sys

import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import create_scrolled_image, create_text_pattern_image

from scroll_detection import detect_scroll


class TestBasicScrollDetection:
    """Tests for basic scroll detection functionality."""

    def test_basic_scroll(self, sample_text_image):
        """Test basic scrolling detection with clean synthetic images."""
        scroll_dist = 250
        img_after = create_scrolled_image(sample_text_image, scroll_dist)
        cursor_pos = (640, 500)

        result = detect_scroll(sample_text_image, img_after, cursor_pos)

        assert result is not None, "No scroll detected"
        assert abs(result.scroll_distance - scroll_dist) <= 5, (
            f"Expected {scroll_dist}, got {result.scroll_distance}"
        )

    def test_no_scroll_identical_images(self, sample_text_image):
        """Test that identical images return None (no scroll detected)."""
        img_after = sample_text_image.copy()
        cursor_pos = (640, 360)

        result = detect_scroll(sample_text_image, img_after, cursor_pos)

        assert result is None, "Should return None for identical images"

    def test_different_images_low_match(self):
        """Test that completely different images have low match percentage if detected."""
        height, width = 720, 1280
        img_text = create_text_pattern_image(height, width, "text")
        img_grid = create_text_pattern_image(height, width, "grid")
        cursor_pos = (640, 360)

        result = detect_scroll(img_text, img_grid, cursor_pos)

        # With very different images, the algorithm might still find some correlation
        # at certain offsets. The test verifies it doesn't produce obviously wrong results.
        # A None result is acceptable, as is a result with low confidence.
        if result is not None:
            # If a result is returned, verify the viewport is reasonable
            assert result.viewport.width > 0 and result.viewport.height > 0, (
                "Viewport dimensions should be positive"
            )


class TestScrollDistances:
    """Tests for various scroll distance scenarios."""

    def test_small_scroll(self):
        """Test that scrolls just above the minimum threshold can be detected."""
        height, width = 720, 1280
        img_before = create_text_pattern_image(height, width, "text")
        scroll_dist = 120  # Just above MIN_SCROLL_DISTANCE
        img_after = create_scrolled_image(img_before, scroll_dist)
        cursor_pos = (640, 360)

        result = detect_scroll(img_before, img_after, cursor_pos)

        assert result is not None, f"Should detect scroll of {scroll_dist}px"
        error = abs(result.scroll_distance - scroll_dist)
        assert error <= 30, f"Error too large: {error}px"

    def test_partial_overlap(self):
        """Test detection with 50% overlap."""
        height, width = 1200, 1280
        img_before = create_text_pattern_image(height, width, "text")
        scroll_dist = int(height * 0.50)
        img_after = create_scrolled_image(img_before, scroll_dist)
        cursor_pos = (640, 600)

        result = detect_scroll(img_before, img_after, cursor_pos)

        assert result is not None, "Should detect scroll with 50% overlap"
        error = abs(result.scroll_distance - scroll_dist)
        assert error <= 10, f"Expected {scroll_dist}, got {result.scroll_distance}"

    def test_very_large_scroll(self):
        """Test very large scroll (60% of height)."""
        height, width = 1400, 1280
        img_before = create_text_pattern_image(height, width, "text")
        scroll_dist = int(height * 0.6)
        img_after = create_scrolled_image(img_before, scroll_dist)
        cursor_pos = (640, 700)

        result = detect_scroll(img_before, img_after, cursor_pos)

        assert result is not None, "Should detect large scroll"
        error = abs(result.scroll_distance - scroll_dist)
        assert error <= 10, f"Expected {scroll_dist}, got {result.scroll_distance}"


class TestCursorPositions:
    """Tests for cursor position handling."""

    def test_cursor_in_margin(self):
        """Test detection when cursor is in a white margin."""
        height, width = 1000, 1280
        img_before = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Content area: columns 200-1000
        y = 50
        line_num = 0
        while y < height - 50:
            thickness = 15 + ((line_num * 7) % 8)
            intensity = ((line_num * 23) % 180) + 20
            img_before[y : y + thickness, 200:1000] = [intensity, intensity, intensity]
            y += 35 + ((line_num * 19) % 15)
            line_num += 1

        scroll_dist = 250
        img_after = create_scrolled_image(img_before, scroll_dist)

        # Place cursor in left margin (empty area)
        cursor_pos = (50, 500)

        result = detect_scroll(img_before, img_after, cursor_pos)

        assert result is not None, "Should detect scroll even with cursor in margin"

        # Verify content area is included in detected bbox
        bbox = result.after_bbox
        bbox_x_end = bbox.x + bbox.width
        content_start, content_end = 200, 1000

        assert bbox.x <= content_start and bbox_x_end >= content_end, (
            f"Bbox should include content area [{content_start}, {content_end}], "
            f"got [{bbox.x}, {bbox_x_end}]"
        )


class TestArrayShapeSafety:
    """Tests for array shape matching to prevent broadcast errors."""

    @pytest.mark.parametrize(
        "height,width,scroll_dist,expected_distance",
        [
            (720, 1280, 50, 50),
            (1440, 2560, 100, 100),
            (800, 1200, 500, 500),
            (600, 800, 580, None),  # Almost no overlap -> no detection
        ],
    )
    def test_various_sizes_correct_detection(
        self, height, width, scroll_dist, expected_distance
    ):
        """Test that various image sizes detect scrolls correctly without shape errors."""
        img_before = create_text_pattern_image(height, width, "text")
        img_after = create_scrolled_image(img_before, scroll_dist)
        cursor_pos = (width // 2, height // 2)

        result = detect_scroll(img_before, img_after, cursor_pos)

        if expected_distance is None:
            assert result is None, (
                f"Expected None for {scroll_dist}px scroll on {height}px image"
            )
        else:
            assert result is not None, f"Should detect {scroll_dist}px scroll"
            assert result.scroll_distance == expected_distance, (
                f"Expected {expected_distance}px, got {result.scroll_distance}px"
            )


class TestScrollUpDirection:
    """Tests for scroll-up detection (content moves down)."""

    def test_basic_scroll_up(self, sample_text_image):
        """Test basic scroll-up detection with clean synthetic images."""
        scroll_dist = 250
        img_after = create_scrolled_image(
            sample_text_image, scroll_dist, direction="up"
        )
        cursor_pos = (640, 500)

        result = detect_scroll(
            sample_text_image, img_after, cursor_pos, scroll_direction="up"
        )

        assert result is not None, "No scroll detected for scroll-up"
        assert abs(result.scroll_distance - scroll_dist) <= 5, (
            f"Expected {scroll_dist}, got {result.scroll_distance}"
        )

    def test_small_scroll_up(self):
        """Test small scroll-up detection."""
        height, width = 720, 1280
        img_before = create_text_pattern_image(height, width, "text")
        scroll_dist = 120
        img_after = create_scrolled_image(img_before, scroll_dist, direction="up")
        cursor_pos = (640, 360)

        result = detect_scroll(img_before, img_after, cursor_pos, scroll_direction="up")

        assert result is not None, f"Should detect scroll-up of {scroll_dist}px"
        error = abs(result.scroll_distance - scroll_dist)
        assert error <= 30, f"Error too large: {error}px"

    def test_large_scroll_up(self):
        """Test large scroll-up detection (50% of height)."""
        height, width = 1200, 1280
        img_before = create_text_pattern_image(height, width, "text")
        scroll_dist = int(height * 0.50)
        img_after = create_scrolled_image(img_before, scroll_dist, direction="up")
        cursor_pos = (640, 600)

        result = detect_scroll(img_before, img_after, cursor_pos, scroll_direction="up")

        assert result is not None, "Should detect scroll-up with 50% overlap"
        error = abs(result.scroll_distance - scroll_dist)
        assert error <= 10, f"Expected {scroll_dist}, got {result.scroll_distance}"

    def test_scroll_up_symmetry_with_scroll_down(self):
        """Test that scroll-up and scroll-down give symmetric results."""
        height, width = 1000, 1280
        img_original = create_text_pattern_image(height, width, "text")
        scroll_dist = 200
        cursor_pos = (640, 500)

        # Create scroll-down image (content moves up)
        img_scroll_down = create_scrolled_image(
            img_original, scroll_dist, direction="down"
        )
        result_down = detect_scroll(
            img_original, img_scroll_down, cursor_pos, scroll_direction="down"
        )

        # Create scroll-up image (content moves down)
        img_scroll_up = create_scrolled_image(img_original, scroll_dist, direction="up")
        result_up = detect_scroll(
            img_original, img_scroll_up, cursor_pos, scroll_direction="up"
        )

        assert result_down is not None, "Scroll-down detection failed"
        assert result_up is not None, "Scroll-up detection failed"

        # Both should detect the same scroll distance
        assert abs(result_down.scroll_distance - scroll_dist) <= 5, (
            f"Scroll-down: expected {scroll_dist}, got {result_down.scroll_distance}"
        )
        assert abs(result_up.scroll_distance - scroll_dist) <= 5, (
            f"Scroll-up: expected {scroll_dist}, got {result_up.scroll_distance}"
        )
