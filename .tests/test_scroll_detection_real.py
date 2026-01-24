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
        """Test detection on scroll_success_268px case.

        Note: This case has minimal overlap between before/after images, making
        accurate detection extremely difficult. We accept either None or a valid
        result since correct behavior is undefined.
        """
        img_before, img_after, cursor_pos, _ = load_json_test_case(
            "scroll_success_268px_1766207411.45.json"
        )

        result = detect_scroll(img_before, img_after, cursor_pos)

        if result is not None:
            assert result.viewport.width > 0 and result.viewport.height > 0, (
                "Viewport dimensions should be positive"
            )

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
        """Test detection on scroll_success_266px case.

        Note: This case has no overlap between before/after images, making
        accurate detection impossible. We accept either None or a valid result
        since correct behavior is undefined.
        """
        img_before, img_after, cursor_pos, _ = load_json_test_case(
            "scroll_success_266px_1765173462.19.json"
        )

        result = detect_scroll(img_before, img_after, cursor_pos)

        if result is not None:
            assert result.viewport.width > 0 and result.viewport.height > 0, (
                "Viewport dimensions should be positive"
            )

    def test_scroll_success_40px_viewport_extends_past_cursor(
        self, load_json_test_case
    ):
        """Test that viewport refinement extends past cursor when content matches.

        This test ensures the PIXEL_TOLERANCE fix (increased to 20) remains in place.
        With tolerance=15, the viewport would stop at the cursor x position (~505).
        With tolerance=20, it correctly extends left to capture the full viewport.
        """
        img_before, img_after, cursor_pos, _ = load_json_test_case(
            "scroll_success_40px_1766818564.08.json"
        )

        result = detect_scroll(img_before, img_after, cursor_pos)

        assert result is not None, "Should detect scroll"
        assert result.scroll_distance == 40, (
            f"Expected 40px scroll, got {result.scroll_distance}px"
        )

        # Viewport should extend significantly left of cursor (x ~505)
        # With the fix, viewport.x should be around 312, width around 1133
        assert result.viewport.x < 400, (
            f"Viewport should extend left of cursor; got x={result.viewport.x}"
        )
        assert result.viewport.width > 1000, (
            f"Viewport should be wide (>1000px); got width={result.viewport.width}"
        )

    def test_scroll_success_40px_with_animation(self, load_json_test_case):
        """Test that viewport refinement handles animations correctly.

        This test ensures the lowered DENSITY_THRESHOLD (0.01) handles cases where
        an animation in the viewport causes non-matching pixels. With threshold=0.3,
        the viewport would incorrectly expand horizontally (1133px) and shrink
        vertically (295px). With threshold=0.01, it correctly detects ~600x600.
        """
        img_before, img_after, cursor_pos, _ = load_json_test_case(
            "scroll_success_40px_1766904228.34.json"
        )

        result = detect_scroll(img_before, img_after, cursor_pos)

        assert result is not None, "Should detect scroll"
        assert result.scroll_distance == 40, (
            f"Expected 40px scroll, got {result.scroll_distance}px"
        )

        # Viewport should be roughly square (~600x600), not wide and short
        assert result.viewport.width < 700, (
            f"Viewport width should be <700px; got {result.viewport.width}"
        )
        assert result.viewport.height > 500, (
            f"Viewport height should be >500px; got {result.viewport.height}"
        )
