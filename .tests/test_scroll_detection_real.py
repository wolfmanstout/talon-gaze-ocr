"""
Tests using real screenshot data from data directory.

Run with: uv run pytest test_scroll_detection_real.py -v
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scroll_detection import detect_scroll


class TestRealImages:
    """Tests using real screenshot data from data directory.

    Note: The scroll_distance_px in JSON files represents the scroll command input,
    not necessarily what the algorithm detects (due to viewport boundaries, etc.).
    These tests verify detection works, with flexible tolerances for real-world data.
    """

    def test_scroll_success_145px(self, load_json_test_case):
        """Test detection on scroll_success_145px case - expects 880px detected."""
        img_before, img_after, cursor_pos, _ = load_json_test_case(
            "scroll_success_145px_1766207805.61.json"
        )

        result = detect_scroll(img_before, img_after, cursor_pos)

        assert result is not None, "Should detect scroll in real images"
        assert result.scroll_distance == 880, (
            f"Expected 880px, got {result.scroll_distance}px"
        )

    def test_scroll_success_268px(self, load_json_test_case):
        """Test detection on scroll_success_268px case."""
        img_before, img_after, cursor_pos, _ = load_json_test_case(
            "scroll_success_268px_1766207411.45.json"
        )

        result = detect_scroll(img_before, img_after, cursor_pos)

        assert result is not None, "Should detect scroll in real images"
        assert result.scroll_distance > 0, "Scroll distance should be positive"

    def test_scroll_success_314px(self, load_json_test_case):
        """Test detection on scroll_success_314px case - expects 880px detected."""
        img_before, img_after, cursor_pos, _ = load_json_test_case(
            "scroll_success_314px_1766207950.50.json"
        )

        result = detect_scroll(img_before, img_after, cursor_pos)

        assert result is not None, "Should detect scroll in real images"
        assert result.scroll_distance == 880, (
            f"Expected 880px, got {result.scroll_distance}px"
        )

    def test_scroll_success_603px(self, load_json_test_case):
        """Test detection on scroll_success_603px case - expects 880px detected."""
        img_before, img_after, cursor_pos, _ = load_json_test_case(
            "scroll_success_603px_1766207524.61.json"
        )

        result = detect_scroll(img_before, img_after, cursor_pos)

        assert result is not None, "Should detect scroll in real images"
        assert result.scroll_distance == 880, (
            f"Expected 880px, got {result.scroll_distance}px"
        )

    def test_scroll_success_266px(self, load_json_test_case):
        """Test detection on scroll_success_266px case."""
        img_before, img_after, cursor_pos, _ = load_json_test_case(
            "scroll_success_266px_1765173462.19.json"
        )

        result = detect_scroll(img_before, img_after, cursor_pos)

        assert result is not None, "Should detect scroll in real images"
        assert result.scroll_distance > 0, "Scroll distance should be positive"
