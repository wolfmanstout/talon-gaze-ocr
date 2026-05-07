"""Tests for OCR cache behavior."""

from typing import cast

import screen_ocr
from screen_ocr import _base

from gaze_ocr._gaze_ocr import BoundingBox, EyeTrackerFallback, OcrCache


def _contents(
    bounding_box: tuple[int, int, int, int], words: list[_base.OcrWord] | None = None
) -> screen_ocr.ScreenContents:
    return screen_ocr.ScreenContents(
        screen_coordinates=None,
        bounding_box=bounding_box,
        screenshot=None,
        result=_base.OcrResult(lines=[_base.OcrLine(words or [])]),
        confidence_threshold=1,
        homophones={},
        search_radius=None,
    )


def _word_texts(contents: screen_ocr.ScreenContents) -> list[str]:
    return [word.text for line in contents.result.lines for word in line.words]


class FakeReader:
    def __init__(self):
        self.read_screen_calls: list[tuple[int, int, int, int] | None] = []
        self.read_current_window_calls = 0

    def read_screen(self, bounding_box: tuple[int, int, int, int] | None = None):
        self.read_screen_calls.append(bounding_box)
        return _contents(
            bounding_box or (0, 0, 100, 100),
            [
                _base.OcrWord("inside", left=15, top=15, width=2, height=2),
                _base.OcrWord("outside", left=90, top=90, width=2, height=2),
            ],
        )

    def read_current_window(self):
        self.read_current_window_calls += 1
        return _contents((20, 30, 120, 150))


def _cache(reader: FakeReader) -> OcrCache:
    return OcrCache(cast(screen_ocr.Reader, reader))


def test_unbounded_main_screen_read_reuses_cache():
    reader = FakeReader()
    cache = _cache(reader)

    first = cache.read(None, EyeTrackerFallback.MAIN_SCREEN)
    second = cache.read(None, EyeTrackerFallback.MAIN_SCREEN)

    assert second is first
    assert reader.read_screen_calls == [None]


def test_unbounded_active_window_read_reuses_cache():
    reader = FakeReader()
    cache = _cache(reader)

    first = cache.read(None, EyeTrackerFallback.ACTIVE_WINDOW)
    second = cache.read(None, EyeTrackerFallback.ACTIVE_WINDOW)

    assert second is first
    assert reader.read_current_window_calls == 1
    assert reader.read_screen_calls == []


def test_unbounded_read_caches_actual_bounding_box_for_later_crop():
    reader = FakeReader()
    cache = _cache(reader)

    cache.read(None, EyeTrackerFallback.MAIN_SCREEN)
    cropped = cache.read(
        BoundingBox(left=10, top=10, right=20, bottom=20),
        EyeTrackerFallback.MAIN_SCREEN,
    )

    assert cropped.bounding_box == (10, 10, 20, 20)
    assert _word_texts(cropped) == ["inside"]
    assert reader.read_screen_calls == [None]


def test_explicit_bounding_box_read_does_not_satisfy_unbounded_read():
    reader = FakeReader()
    cache = _cache(reader)

    cache.read(
        BoundingBox(left=10, top=20, right=30, bottom=40),
        EyeTrackerFallback.MAIN_SCREEN,
    )
    cache.read(None, EyeTrackerFallback.MAIN_SCREEN)

    assert reader.read_screen_calls == [(10, 20, 30, 40), None]


def test_unbounded_cache_misses_when_fallback_mode_changes():
    reader = FakeReader()
    cache = _cache(reader)

    cache.read(None, EyeTrackerFallback.MAIN_SCREEN)
    cache.read(None, EyeTrackerFallback.ACTIVE_WINDOW)

    assert reader.read_screen_calls == [None]
    assert reader.read_current_window_calls == 1
