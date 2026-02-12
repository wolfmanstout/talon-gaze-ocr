"""Pytest configuration and shared fixtures for scroll detection tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path to import scroll_detection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path to test data directory
DATA_DIR = Path(__file__).parent / "data"


def create_text_pattern_image(
    height: int, width: int, pattern_type: str = "text"
) -> np.ndarray:
    """
    Create a synthetic image with text-like patterns.

    Args:
        height: Image height
        width: Image width
        pattern_type: Type of pattern ('lines', 'grid', 'text')

    Returns:
        Numpy array (H, W, 3) with RGB values
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    if pattern_type == "lines":
        # Horizontal lines simulating text
        for y in range(50, height - 50, 40):
            thickness = 20
            img[y : y + thickness, 100 : width - 100] = [0, 0, 0]

    elif pattern_type == "grid":
        # Grid pattern
        for y in range(0, height, 50):
            img[y : y + 2, :] = [100, 100, 100]
        for x in range(0, width, 50):
            img[:, x : x + 2] = [100, 100, 100]

    elif pattern_type == "text":
        # Non-repeating text-like pattern with position-dependent unique content
        y = 50
        line_num = 0
        while y < height - 50:
            thickness = 15 + ((line_num * 7) % 8)
            left_margin = 100 + ((line_num * 11) % 50)
            right_offset = (line_num * 13) % 30
            base_intensity = ((line_num * 23) % 180) + 20

            img[y : y + thickness, left_margin : width - 100 - right_offset] = [
                base_intensity,
                base_intensity,
                base_intensity,
            ]

            marker_x = 50 + ((line_num * 31) % 30)
            marker_intensity = ((line_num * 41) % 150) + 50
            img[y : y + thickness, marker_x : marker_x + 5] = [
                marker_intensity,
                marker_intensity,
                marker_intensity,
            ]

            y += 35 + ((line_num * 19) % 15)
            line_num += 1

    return img


def create_scrolled_image(
    original_img: np.ndarray, scroll_distance: int, direction: str = "down"
) -> np.ndarray:
    """
    Create a scrolled version of an image.

    Args:
        original_img: Original image array
        scroll_distance: Pixels to scroll (positive value)
        direction: "down" (content moves up) or "up" (content moves down)

    Returns:
        Scrolled image with same dimensions
    """
    height = original_img.shape[0]
    scrolled = np.ones_like(original_img) * 255  # White background

    remaining_height = height - scroll_distance
    if remaining_height > 0:
        if direction == "down":
            # Scroll down: content moves up, new content appears at bottom
            scrolled[:remaining_height] = original_img[scroll_distance:]
        else:
            # Scroll up: content moves down, new content appears at top
            scrolled[scroll_distance:] = original_img[:remaining_height]

    return scrolled


@pytest.fixture
def sample_text_image():
    """Create a sample text pattern image for testing."""
    return create_text_pattern_image(1000, 1280, "text")


@pytest.fixture
def sample_grid_image():
    """Create a sample grid pattern image for testing."""
    return create_text_pattern_image(720, 1280, "grid")


@pytest.fixture
def load_real_image_pair():
    """Factory fixture to load real image pairs from data directory."""
    from PIL import Image

    def _load(before_name: str, after_name: str):
        before_path = DATA_DIR / before_name
        after_path = DATA_DIR / after_name

        if not before_path.exists() or not after_path.exists():
            pytest.skip(f"Image files not found: {before_name}, {after_name}")

        img_before = np.array(Image.open(before_path).convert("RGB"))
        img_after = np.array(Image.open(after_path).convert("RGB"))

        return img_before, img_after

    return _load


@pytest.fixture
def load_json_test_case():
    """Factory fixture to load test cases from JSON files."""
    import json

    from PIL import Image

    def _load(json_name: str):
        json_path = DATA_DIR / json_name

        if not json_path.exists():
            pytest.skip(f"JSON file not found: {json_name}")

        with open(json_path) as f:
            data = json.load(f)

        base = str(json_path).rsplit(".json", 1)[0]
        before_path = base + "_before.png"
        after_path = base + "_after.png"

        if not os.path.exists(before_path) or not os.path.exists(after_path):
            pytest.skip(f"Image files not found for {json_name}")

        img_before = np.array(Image.open(before_path).convert("RGB"))
        img_after = np.array(Image.open(after_path).convert("RGB"))

        cursor_pos = None
        cp = data.get("cursor_position", {})
        if "x" in cp and "y" in cp:
            cursor_pos = (cp["x"], cp["y"])

        return img_before, img_after, cursor_pos, data

    return _load
