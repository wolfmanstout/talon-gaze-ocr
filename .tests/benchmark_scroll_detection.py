"""
Performance benchmarks for vertical scrolling detection algorithm.

Run with: uv run pytest benchmark_scroll_detection.py -v
For detailed stats: uv run pytest benchmark_scroll_detection.py --benchmark-only
"""

import os
import sys

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import create_scrolled_image, create_text_pattern_image

from scroll_detection import detect_scroll


class TestPerformanceBenchmarks:
    """Performance benchmarks for scroll detection."""

    @pytest.fixture
    def images_720p(self):
        """Create 720p test images."""
        height, width = 720, 1280
        img_before = create_text_pattern_image(height, width, "text")
        scroll_dist = 200
        img_after = create_scrolled_image(img_before, scroll_dist)
        cursor_pos = (width // 2, height // 2)
        return img_before, img_after, cursor_pos

    @pytest.fixture
    def images_1080p(self):
        """Create 1080p test images."""
        height, width = 1080, 1920
        img_before = create_text_pattern_image(height, width, "text")
        scroll_dist = 300
        img_after = create_scrolled_image(img_before, scroll_dist)
        cursor_pos = (width // 2, height // 2)
        return img_before, img_after, cursor_pos

    @pytest.fixture
    def images_1440p(self):
        """Create 1440p test images."""
        height, width = 1440, 2560
        img_before = create_text_pattern_image(height, width, "text")
        scroll_dist = 300
        img_after = create_scrolled_image(img_before, scroll_dist)
        cursor_pos = (width // 2, height // 2)
        return img_before, img_after, cursor_pos

    @pytest.fixture
    def images_4k(self):
        """Create 4K test images."""
        height, width = 2160, 3840
        img_before = create_text_pattern_image(height, width, "text")
        scroll_dist = 400
        img_after = create_scrolled_image(img_before, scroll_dist)
        cursor_pos = (width // 2, height // 2)
        return img_before, img_after, cursor_pos

    def test_benchmark_720p(self, benchmark, images_720p):
        """Benchmark scroll detection on 720p images."""
        img_before, img_after, cursor_pos = images_720p

        result = benchmark(detect_scroll, img_before, img_after, cursor_pos)

        assert result is not None, "Should detect scroll"

    def test_benchmark_1080p(self, benchmark, images_1080p):
        """Benchmark scroll detection on 1080p images."""
        img_before, img_after, cursor_pos = images_1080p

        result = benchmark(detect_scroll, img_before, img_after, cursor_pos)

        assert result is not None, "Should detect scroll"

    def test_benchmark_1440p(self, benchmark, images_1440p):
        """Benchmark scroll detection on 1440p images."""
        img_before, img_after, cursor_pos = images_1440p

        result = benchmark(detect_scroll, img_before, img_after, cursor_pos)

        assert result is not None, "Should detect scroll"

    def test_benchmark_4k(self, benchmark, images_4k):
        """Benchmark scroll detection on 4K images."""
        img_before, img_after, cursor_pos = images_4k

        result = benchmark(detect_scroll, img_before, img_after, cursor_pos)

        assert result is not None, "Should detect scroll"
