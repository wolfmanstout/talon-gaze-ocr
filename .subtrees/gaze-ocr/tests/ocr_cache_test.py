"""Tests for OCR cache behavior."""

import logging
from typing import cast

import screen_ocr
from screen_ocr import _base

from gaze_ocr import _gaze_ocr
from gaze_ocr._gaze_ocr import BoundingBox, Controller, EyeTrackerFallback, OcrCache


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
    SCREEN = (0, 0, 100, 100)

    def __init__(self):
        self.read_screen_calls: list[tuple[int, int, int, int] | None] = []
        self.read_current_window_calls = 0

    def read_screen(self, bounding_box: tuple[int, int, int, int] | None = None):
        self.read_screen_calls.append(bounding_box)
        if bounding_box:
            # Simulate the real reader clamping the request to the screen.
            bounding_box = (
                max(self.SCREEN[0], bounding_box[0]),
                max(self.SCREEN[1], bounding_box[1]),
                min(self.SCREEN[2], bounding_box[2]),
                min(self.SCREEN[3], bounding_box[3]),
            )
        return _contents(
            bounding_box or self.SCREEN,
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


def test_bounded_read_extending_offscreen_reuses_cache():
    # Requests near the screen edge extend past it (e.g. padded gaze bounds).
    # The reader clamps them, but the cache should still treat a subset of the
    # previous request as a hit.
    reader = FakeReader()
    cache = _cache(reader)

    cache.read(
        BoundingBox(left=-50, top=-50, right=60, bottom=60),
        EyeTrackerFallback.MAIN_SCREEN,
    )
    cropped = cache.read(
        BoundingBox(left=-40, top=-40, right=50, bottom=50),
        EyeTrackerFallback.MAIN_SCREEN,
    )

    assert reader.read_screen_calls == [(-50, -50, 60, 60)]
    assert cropped.bounding_box == (-40, -40, 50, 50)
    assert _word_texts(cropped) == ["inside"]


def test_unbounded_cache_misses_when_fallback_mode_changes():
    reader = FakeReader()
    cache = _cache(reader)

    cache.read(None, EyeTrackerFallback.MAIN_SCREEN)
    cache.read(None, EyeTrackerFallback.ACTIVE_WINDOW)

    assert reader.read_screen_calls == [None]
    assert reader.read_current_window_calls == 1


def test_empty_cache_miss_does_not_warn(caplog):
    reader = FakeReader()
    cache = _cache(reader)

    with caplog.at_level(logging.WARNING):
        cache.read(None, EyeTrackerFallback.MAIN_SCREEN)

    assert not caplog.records


def test_cache_hit_does_not_warn(caplog):
    reader = FakeReader()
    cache = _cache(reader)
    cache.read(None, EyeTrackerFallback.MAIN_SCREEN)

    with caplog.at_level(logging.WARNING):
        cache.read(None, EyeTrackerFallback.MAIN_SCREEN)

    assert not caplog.records


def test_populated_cache_miss_warns_with_process_lifetime_rate(caplog, monkeypatch):
    monkeypatch.setattr(_gaze_ocr, "_populated_cache_call_count", 0)
    monkeypatch.setattr(_gaze_ocr, "_populated_cache_miss_count", 0)

    first_reader = FakeReader()
    first_cache = _cache(first_reader)
    first_cache.read(
        BoundingBox(left=10, top=20, right=30, bottom=40),
        EyeTrackerFallback.MAIN_SCREEN,
    )
    first_cache.read(
        BoundingBox(left=10, top=20, right=30, bottom=40),
        EyeTrackerFallback.MAIN_SCREEN,
    )

    second_reader = FakeReader()
    second_cache = _cache(second_reader)
    second_cache.read(None, EyeTrackerFallback.MAIN_SCREEN)

    with caplog.at_level(logging.WARNING):
        first_cache.read(None, EyeTrackerFallback.MAIN_SCREEN)
        second_cache.read(None, EyeTrackerFallback.ACTIVE_WINDOW)

    assert len(caplog.records) == 2
    assert caplog.records[0].message == (
        "OCR cache miss with populated cache: requested_bounds=None, "
        "cached_bounds=BoundingBox(left=10, right=30, top=20, bottom=40), "
        "requested_fallback=MAIN_SCREEN, cached_fallback=None; "
        "misses=50.0% of 2 calls"
    )
    assert caplog.records[1].message == (
        "OCR cache miss with populated cache: requested_bounds=None, "
        "cached_bounds=BoundingBox(left=0, right=100, top=0, bottom=100), "
        "requested_fallback=ACTIVE_WINDOW, cached_fallback=MAIN_SCREEN; "
        "misses=66.7% of 3 calls"
    )


def test_controller_invalidation_forces_unbounded_lookup_to_reread():
    reader = FakeReader()
    controller = Controller(
        ocr_reader=cast(screen_ocr.Reader, reader),
        eye_tracker=None,
        mouse=None,
        keyboard=None,
    )
    try:
        controller.read_nearby()
        prior_contents = controller.latest_screen_contents()
        controller.invalidate_ocr_cache()

        # Rendering for an in-progress disambiguation can continue using the prior
        # result even though it is no longer eligible for a new OCR lookup.
        assert controller.latest_screen_contents() is prior_contents

        generator = controller.move_cursor_to_words_generator(
            "missing", disambiguate=False
        )
        try:
            next(generator)
        except StopIteration:
            pass
        else:
            raise AssertionError("Generator unexpectedly yielded for disambiguation")

        assert reader.read_screen_calls == [None, None]
    finally:
        controller.shutdown()


def test_controller_start_reading_is_noop_and_reads_reuse_ocr_cache():
    reader = FakeReader()
    controller = Controller(
        ocr_reader=cast(screen_ocr.Reader, reader),
        eye_tracker=None,
        mouse=None,
        keyboard=None,
    )
    try:
        controller.start_reading_nearby()
        assert reader.read_screen_calls == []

        controller.read_nearby()
        first = controller.latest_screen_contents()
        controller.read_nearby()

        assert controller.latest_screen_contents() is first
        assert reader.read_screen_calls == [None]
    finally:
        controller.shutdown()
