"""Library for manipulating on-screen text using gaze tracking and OCR.

Supports disambiguation using Python generators to stop and resume computation. The *_generator
functions return a generator which can be started with next(generator), which will return a list of
matches if disambiguation is needed. Resume computation with generator.send(match). When computation
completes, next() or send() will raise StopIteration with the .value set to the return value.
"""

import os.path
import time
from concurrent import futures
from enum import Enum, auto
from typing import Callable, Generator, Optional, Sequence, Tuple

from screen_ocr import Reader, ScreenContents, WordLocation


class Controller:
    """Mediates interaction with gaze tracking and OCR.

    Provide Mouse and Keyboard from gaze_ocr.dragonfly or gaze_ocr.talon. AppActions is optional.
    """

    def __init__(
        self,
        ocr_reader: Reader,
        eye_tracker,
        mouse,
        keyboard,
        app_actions=None,
        save_data_directory: Optional[str] = None,
    ):
        self.ocr_reader = ocr_reader
        self.eye_tracker = eye_tracker
        self.mouse = mouse
        self.keyboard = keyboard
        self.app_actions = app_actions
        self.save_data_directory = save_data_directory
        self._change_radius = 10
        self._executor = futures.ThreadPoolExecutor(max_workers=1)
        self._future = None

    def shutdown(self, wait=True):
        self._executor.shutdown(wait)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)
        return False

    def start_reading_nearby(self) -> None:
        """Start OCR nearby the gaze point in a background thread."""
        gaze_point = (
            self.eye_tracker.get_gaze_point()
            if self.eye_tracker and self.eye_tracker.is_connected
            else None
        )
        # Don't enqueue multiple requests.
        if self._future and not self._future.done():
            self._future.cancel()
        self._future = self._executor.submit(
            (lambda: self.ocr_reader.read_nearby(gaze_point))
            if gaze_point
            else (lambda: self.ocr_reader.read_screen())
        )

    # TODO Use timestamp range instead of a single timestamp.
    def read_nearby(self, timestamp: Optional[float] = None) -> None:
        """Perform OCR nearby the gaze point in the current thread.

        Arguments:
        timestamp: If specified, read nearby the gaze point at the provided timestamp.
        """
        gaze_point = (
            self.eye_tracker.get_gaze_point_at_timestamp(timestamp)
            if self.eye_tracker and self.eye_tracker.is_connected and timestamp
            else self.eye_tracker.get_gaze_point()
            if self.eye_tracker and self.eye_tracker.is_connected
            else None
        )
        self._future = futures.Future()
        if not gaze_point:
            self._future.set_result(self.ocr_reader.read_screen())
            return
        if not timestamp:
            self._future.set_result(self.ocr_reader.read_nearby(gaze_point))
            return
        # TODO Extract constants into optional constructor params.
        bounds = self.eye_tracker.get_gaze_bounds_during_time_range(
            timestamp - 0.2, timestamp + 0.2
        )
        if not bounds:
            self._future.set_result(self.ocr_reader.read_nearby(gaze_point))
            return
        max_radius = max(bounds.right - bounds.left, bounds.bottom - bounds.top) / 2.0

        self._future.set_result(
            self.ocr_reader.read_nearby(
                gaze_point,
                search_radius=max_radius + 100,
                crop_radius=max_radius + 200,
            )
        )

    def latest_screen_contents(self) -> ScreenContents:
        """Return the ScreenContents of the latest call to start_reading_nearby().

        Blocks until available.
        """
        if not self._future:
            raise RuntimeError(
                "Call start_reading_nearby() before latest_screen_contents()"
            )
        return self._future.result()

    def move_cursor_to_words(
        self,
        words: str,
        cursor_position: str = "middle",
        timestamp: Optional[float] = None,
        click_offset_right: int = 0,
    ) -> Optional[Tuple[int, int]]:
        """Move the mouse cursor nearby the specified word or words.

        If successful, returns the new cursor coordinates.

        Arguments:
        words: The word or words to search for.
        cursor_position: "before", "middle", or "after" (relative to the matching word)
        timestamp: If specified, read nearby gaze at the provided timestamp.
        click_offset_right: Adjust the X-coordinate when clicking.
        """
        return self._extract_result(
            self.move_cursor_to_words_generator(
                words,
                disambiguate=False,
                cursor_position=cursor_position,
                timestamp=timestamp,
                click_offset_right=click_offset_right,
            )
        )

    def move_cursor_to_words_generator(
        self,
        words: str,
        disambiguate: bool,
        cursor_position: str = "middle",
        timestamp: Optional[float] = None,
        click_offset_right: int = 0,
    ) -> Generator[
        Sequence[Sequence[WordLocation]],
        Sequence[WordLocation],
        Optional[Tuple[int, int]],
    ]:
        """Same as move_cursor_to_words, except it supports disambiguation through a generator.
        See header comment for details.
        """
        if timestamp:
            self.read_nearby(timestamp)
        screen_contents = self.latest_screen_contents()
        if disambiguate:
            matches = screen_contents.find_matching_words(words)
        else:
            match = screen_contents.find_nearest_words(words)
            matches = [match] if match else []
        self._write_data(screen_contents, words, matches)
        if not matches:
            return None
        if len(matches) > 1:
            # Yield all results to the caller and let them send the chosen match.
            locations = yield matches
        else:
            locations = matches[0]
        if cursor_position == "before":
            coordinates = locations[0].start_coordinates
        elif cursor_position == "middle":
            coordinates = (
                int((locations[0].left + locations[-1].right) / 2),
                int((locations[0].top + locations[-1].bottom) / 2),
            )
        elif cursor_position == "after":
            coordinates = locations[-1].end_coordinates
        else:
            raise ValueError(cursor_position)
        if self._screenshot_changed_near_coordinates(
            screen_contents.screenshot,
            screen_contents.screen_offset,
            coordinates,
        ):
            return None
        self.mouse.move(self._apply_click_offset(coordinates, click_offset_right))
        return coordinates

    move_cursor_to_word = move_cursor_to_words

    def _screenshot_changed_near_coordinates(
        self, old_screenshot, old_screen_offset, coordinates
    ) -> bool:
        return False
        # Disable this functionality for now due to interaction with the cursor
        # during selection. Also, most screen changes happen after moving the
        # cursor.
        # old_bounding_box = (
        #     max(0, coordinates[0] - old_screen_offset[0] - self._change_radius),
        #     max(0, coordinates[1] - old_screen_offset[1] - self._change_radius),
        #     min(
        #         old_screenshot.width,
        #         coordinates[0] - old_screen_offset[0] + self._change_radius,
        #     ),
        #     min(
        #         old_screenshot.height,
        #         coordinates[1] - old_screen_offset[1] + self._change_radius,
        #     ),
        # )
        # old_patch = old_screenshot.crop(old_bounding_box)
        # new_screenshot = ImageGrab.grab()
        # new_bounding_box = (
        #     max(0, coordinates[0] - self._change_radius),
        #     max(0, coordinates[1] - self._change_radius),
        #     min(new_screenshot.width, coordinates[0] + self._change_radius),
        #     min(new_screenshot.height, coordinates[1] + self._change_radius),
        # )
        # new_patch = new_screenshot.crop(new_bounding_box)
        # return ImageChops.difference(new_patch, old_patch).getbbox() is not None

    # Returns true if the location should be included as a candidate.
    FilterLocationCallable = Callable[[Sequence[WordLocation]], bool]

    class SelectionPosition(Enum):
        NONE = auto()
        LEFT = auto()
        RIGHT = auto()

    def move_text_cursor_to_words(
        self,
        words: str,
        cursor_position: str = "middle",
        filter_location_function: Optional[FilterLocationCallable] = None,
        include_whitespace: bool = False,
        timestamp: Optional[float] = None,
        click_offset_right: int = 0,
    ) -> Optional[Sequence[WordLocation]]:
        """Move the text cursor nearby the specified word or phrase.

        If successful, returns list of screen_ocr.WordLocation of the matching words.

        Arguments:
        words: The word or phrase to search for.
        cursor_position: "before", "middle", or "after" (relative to the matching word).
        filter_location_function: Given a sequence of word locations, return whether to proceed with
                                    cursor movement.
        include_whitespace: Include whitespace adjacent to the words.
        timestamp: Use gaze position at the provided timestamp.
        click_offset_right: Adjust the X-coordinate when clicking.
        """
        return self._extract_result(
            self.move_text_cursor_to_words_generator(
                words,
                disambiguate=False,
                cursor_position=cursor_position,
                filter_location_function=filter_location_function,
                include_whitespace=include_whitespace,
                timestamp=timestamp,
                click_offset_right=click_offset_right,
            )
        )

    def move_text_cursor_to_words_generator(
        self,
        words: str,
        disambiguate: bool,
        cursor_position: str = "middle",
        filter_location_function: Optional[FilterLocationCallable] = None,
        include_whitespace: bool = False,
        timestamp: Optional[float] = None,
        click_offset_right: int = 0,
        hold_shift: bool = False,
        selection_position: Optional[SelectionPosition] = None,
    ) -> Generator[
        Sequence[Sequence[WordLocation]],
        Sequence[WordLocation],
        Optional[Sequence[WordLocation]],
    ]:
        """Same as move_text_cursor_to_words, except it supports disambiguation through a generator.
        See header comment for details.
        """
        if timestamp:
            self.read_nearby(timestamp)
        screen_contents = self.latest_screen_contents()
        matches = screen_contents.find_matching_words(words)
        if filter_location_function:
            matches = list(filter(filter_location_function, matches))
        if not disambiguate:
            match = screen_contents.find_nearest_words_within_matches(matches)
            matches = [match] if match else []
        self._write_data(screen_contents, words, matches)
        if not matches:
            return None
        if len(matches) > 1:
            locations = yield matches
        else:
            locations = matches[0]
        if hold_shift:
            self.keyboard.shift_down()
        try:
            if not selection_position:
                # Guess the selection position.
                if not hold_shift:
                    selection_position = self.SelectionPosition.NONE
                elif cursor_position == "before":
                    selection_position = self.SelectionPosition.LEFT
                elif cursor_position == "after":
                    selection_position = self.SelectionPosition.RIGHT
                else:
                    selection_position = self.SelectionPosition.NONE
            if not self._move_text_cursor_to_word_locations(
                locations,
                cursor_position=cursor_position,
                include_whitespace=include_whitespace,
                click_offset_right=click_offset_right,
                selection_position=selection_position,
            ):
                return None
        finally:
            if hold_shift:
                self.keyboard.shift_up()
        return locations

    move_text_cursor_to_word = move_text_cursor_to_words

    def move_text_cursor_to_longest_prefix(
        self,
        words: str,
        cursor_position: str = "middle",
        filter_location_function: Optional[FilterLocationCallable] = None,
        timestamp: Optional[float] = None,
        click_offset_right: int = 0,
        hold_shift: bool = False,
    ) -> Tuple[Optional[Sequence[WordLocation]], int]:
        """Moves the text cursor to the longest prefix of the provided words that
        matches onscreen text. See move_text_cursor_to_words for argument details."""
        return self._extract_result(
            self.move_text_cursor_to_longest_prefix_generator(
                words,
                disambiguate=False,
                cursor_position=cursor_position,
                filter_location_function=filter_location_function,
                timestamp=timestamp,
                click_offset_right=click_offset_right,
                hold_shift=hold_shift,
            )
        )

    def move_text_cursor_to_longest_prefix_generator(
        self,
        words: str,
        disambiguate: bool,
        cursor_position: str = "middle",
        filter_location_function: Optional[FilterLocationCallable] = None,
        timestamp: Optional[float] = None,
        click_offset_right: int = 0,
        hold_shift: bool = False,
    ) -> Generator[
        Sequence[Sequence[WordLocation]],
        Sequence[WordLocation],
        Tuple[Optional[Sequence[WordLocation]], int],
    ]:
        """Same as move_text_cursor_to_longest_prefix, except it supports
        disambiguation through a generator. See header comment for details."""
        if timestamp:
            self.read_nearby(timestamp)
        screen_contents = self.latest_screen_contents()
        matches, prefix_length = screen_contents.find_longest_matching_prefix(words)
        if filter_location_function:
            matches = list(filter(filter_location_function, matches))
        if not disambiguate:
            match = screen_contents.find_nearest_words_within_matches(matches)
            matches = [match] if match else []
        self._write_data(screen_contents, words, matches)
        if not matches:
            return None, 0
        if len(matches) > 1:
            locations = yield matches
        else:
            locations = matches[0]
        if hold_shift:
            self.keyboard.shift_down()
        try:
            # Guess the selection position.
            selection_position = (
                self.SelectionPosition.LEFT
                if hold_shift
                else self.SelectionPosition.NONE
            )
            if not self._move_text_cursor_to_word_locations(
                locations,
                cursor_position=cursor_position,
                include_whitespace=False,
                click_offset_right=click_offset_right,
                selection_position=selection_position,
            ):
                return None, 0
        finally:
            if hold_shift:
                self.keyboard.shift_up()
        return locations, prefix_length

    def move_text_cursor_to_longest_suffix(
        self,
        words: str,
        cursor_position: str = "middle",
        filter_location_function: Optional[FilterLocationCallable] = None,
        timestamp: Optional[float] = None,
        click_offset_right: int = 0,
        hold_shift: bool = False,
    ) -> Tuple[Optional[Sequence[WordLocation]], int]:
        """Moves the text cursor to the longest suffix of the provided words that
        matches onscreen text. See move_text_cursor_to_words for argument details."""
        return self._extract_result(
            self.move_text_cursor_to_longest_suffix_generator(
                words,
                disambiguate=False,
                cursor_position=cursor_position,
                filter_location_function=filter_location_function,
                timestamp=timestamp,
                click_offset_right=click_offset_right,
                hold_shift=hold_shift,
            )
        )

    def move_text_cursor_to_longest_suffix_generator(
        self,
        words: str,
        disambiguate: bool,
        cursor_position: str = "middle",
        filter_location_function: Optional[FilterLocationCallable] = None,
        timestamp: Optional[float] = None,
        click_offset_right: int = 0,
        hold_shift: bool = False,
    ) -> Generator[
        Sequence[Sequence[WordLocation]],
        Sequence[WordLocation],
        Tuple[Optional[Sequence[WordLocation]], int],
    ]:
        """Same as move_text_cursor_to_longest_suffix, except it supports
        disambiguation through a generator. See header comment for details."""
        if timestamp:
            self.read_nearby(timestamp)
        screen_contents = self.latest_screen_contents()
        matches, suffix_length = screen_contents.find_longest_matching_suffix(words)
        if filter_location_function:
            matches = list(filter(filter_location_function, matches))
        if not disambiguate:
            match = screen_contents.find_nearest_words_within_matches(matches)
            matches = [match] if match else []
        self._write_data(screen_contents, words, matches)
        if not matches:
            return None, 0
        if len(matches) > 1:
            locations = yield matches
        else:
            locations = matches[0]
        if hold_shift:
            self.keyboard.shift_down()
        try:
            # Guess the selection position.
            selection_position = (
                self.SelectionPosition.RIGHT
                if hold_shift
                else self.SelectionPosition.NONE
            )
            if not self._move_text_cursor_to_word_locations(
                locations,
                cursor_position=cursor_position,
                include_whitespace=False,
                click_offset_right=click_offset_right,
                selection_position=selection_position,
            ):
                return None, 0
        finally:
            if hold_shift:
                self.keyboard.shift_up()
        return locations, suffix_length

    def move_text_cursor_to_insertion_point_generator(
        self,
        words: str,
        disambiguate: bool,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
        click_offset_right: int = 0,
    ) -> Generator[Sequence[Sequence[WordLocation]], Sequence[WordLocation], str,]:
        """TODO"""
        if start_timestamp:
            self.read_nearby(start_timestamp)
        screen_contents = self.latest_screen_contents()
        prefix_matches, prefix_length = screen_contents.find_longest_matching_prefix(
            words
        )
        if end_timestamp:
            self.read_nearby(end_timestamp)
        screen_contents = self.latest_screen_contents()
        suffix_matches, suffix_length = screen_contents.find_longest_matching_suffix(
            words
        )
        if not prefix_matches and not suffix_matches:
            return ""
        elif prefix_matches and suffix_matches:
            remainder = words[prefix_length:-suffix_length].strip()
            # TODO: Check if locations are misaligned or assume anchored one side
        elif prefix_matches:
            remainder = words[prefix_length:]
        else:
            assert suffix_matches
            remainder = words[:-suffix_length]
        matches = prefix_matches + suffix_matches
        if not disambiguate:
            match = screen_contents.find_nearest_words_within_matches(matches)
            matches = [match] if match else []
        self._write_data(screen_contents, words, matches)
        if not matches:
            return ""
        if len(matches) > 1:
            locations = yield matches
        else:
            locations = matches[0]
        cursor_position = "after" if locations in prefix_matches else "before"
        if not self._move_text_cursor_to_word_locations(
            locations,
            cursor_position=cursor_position,
            include_whitespace=False,
            click_offset_right=click_offset_right,
            selection_position=self.SelectionPosition.NONE,
        ):
            return ""
        return remainder

    def _move_text_cursor_to_word_locations(
        self,
        locations: Sequence[WordLocation],
        cursor_position: str,
        include_whitespace: bool,
        click_offset_right: int,
        selection_position: SelectionPosition,
    ) -> bool:
        if cursor_position == "before":
            distance_from_left = locations[0].left_char_offset
            distance_from_right = locations[0].right_char_offset + len(
                locations[0].text
            )
            if not self._move_text_cursor_to_position(
                start_coordinates=locations[0].start_coordinates,
                end_coordinates=locations[0].end_coordinates,
                click_offset_right=click_offset_right,
                distance_from_left=distance_from_left,
                distance_from_right=distance_from_right,
                selection_position=selection_position,
            ):
                return False
            if (
                include_whitespace
                and not distance_from_left
                and self.app_actions
                and not self.keyboard.is_shift_down()
            ):
                left_chars = self.app_actions.peek_left()
                # Check that there is actually a space adjacent (not a newline). Google docs
                # represents a newline as newline followed by space, so we handle that case as
                # well.
                if (
                    len(left_chars) >= 2
                    and left_chars[-1] == " "
                    and left_chars[-2] != "\n"
                ):
                    self.keyboard.left(1)
        elif cursor_position == "middle":
            # Note: if it's helpful, we could change this to position the cursor
            # in the middle of the word.
            coordinates = self._apply_click_offset(
                (
                    int((locations[0].left + locations[-1].right) / 2),
                    int((locations[0].top + locations[-1].bottom) / 2),
                ),
                click_offset_right,
            )
            if self._screenshot_changed_near_coordinates(
                self.latest_screen_contents().screenshot,
                self.latest_screen_contents().screen_offset,
                coordinates,
            ):
                return False
            self.mouse.move(coordinates)
            self.mouse.click()
        if cursor_position == "after":
            distance_from_right = locations[-1].right_char_offset
            distance_from_left = locations[-1].left_char_offset + len(
                locations[-1].text
            )
            if not self._move_text_cursor_to_position(
                start_coordinates=locations[-1].start_coordinates,
                end_coordinates=locations[-1].end_coordinates,
                click_offset_right=click_offset_right,
                distance_from_left=distance_from_left,
                distance_from_right=distance_from_right,
                selection_position=selection_position,
            ):
                return False
            if (
                include_whitespace
                and not distance_from_right
                and self.app_actions
                and not self.keyboard.is_shift_down()
            ):
                right_chars = self.app_actions.peek_right()
                if right_chars and right_chars[0] == " ":
                    self.keyboard.right(1)
        return True

    def _move_text_cursor_to_position(
        self,
        start_coordinates: Tuple[int, int],
        end_coordinates: Tuple[int, int],
        click_offset_right: int,
        distance_from_left: int,
        distance_from_right: int,
        selection_position: SelectionPosition,
    ):
        # Determine whether to start from the left or the right.
        if not distance_from_left:
            start_from_left = True
        elif not distance_from_right:
            start_from_left = False
        elif selection_position == self.SelectionPosition.RIGHT:
            # Mac selection can only be reliably expanded outward.
            start_from_left = True
        elif selection_position == self.SelectionPosition.LEFT:
            # Mac selection can only be reliably expanded outward.
            start_from_left = False
        elif distance_from_left <= distance_from_right:
            start_from_left = True
        else:
            start_from_left = False
        if start_from_left:
            coordinates = self._apply_click_offset(
                start_coordinates, click_offset_right
            )
            if self._screenshot_changed_near_coordinates(
                self.latest_screen_contents().screenshot,
                self.latest_screen_contents().screen_offset,
                coordinates,
            ):
                return False
            self.mouse.move(coordinates)
            self.mouse.click()
            if distance_from_left:
                self.keyboard.right(distance_from_left)
        else:
            # Start from the right.
            coordinates = self._apply_click_offset(end_coordinates, click_offset_right)
            if self._screenshot_changed_near_coordinates(
                self.latest_screen_contents().screenshot,
                self.latest_screen_contents().screen_offset,
                coordinates,
            ):
                return False
            self.mouse.move(coordinates)
            self.mouse.click()
            if distance_from_right:
                self.keyboard.left(distance_from_right)
        return True

    def select_text(
        self,
        start_words: str,
        end_words: Optional[str] = None,
        for_deletion: bool = False,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
        click_offset_right: int = 0,
        after_start: bool = False,
        before_end: bool = False,
    ) -> Optional[Sequence[WordLocation]]:
        """Select a range of onscreen text.

        If only start_words is provided, the full word or phrase is selected. If
        end_word is provided, a range from the start words to end words will be
        selected.

        Arguments:
        for_deletion: If True, select adjacent whitespace for clean deletion of
                      the selected text.
        start_timestamp: If specified, read start nearby gaze at the provided timestamp.
        end_timestamp: If specified, read end nearby gaze at the provided timestamp.
        click_offset_right: Adjust the X-coordinate when clicking.
        after_start: If true, begin selection after the start word.
        before_end: If true, end selection before the end word.
        """
        return self._extract_result(
            self.select_text_generator(
                start_words,
                disambiguate=False,
                end_words=end_words,
                for_deletion=for_deletion,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                click_offset_right=click_offset_right,
                after_start=after_start,
                before_end=before_end,
            )
        )

    def select_text_generator(
        self,
        start_words: str,
        disambiguate: bool,
        end_words: Optional[str] = None,
        for_deletion: bool = False,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
        click_offset_right: int = 0,
        after_start: bool = False,
        before_end: bool = False,
        select_pause_seconds: float = 0.01,
    ) -> Generator[
        Sequence[Sequence[WordLocation]],
        Sequence[WordLocation],
        Optional[Sequence[WordLocation]],
    ]:
        """Same as select_text, except it supports disambiguation through a generator.
        See header comment for details.
        """
        if start_timestamp:
            self.read_nearby(start_timestamp)
        start_locations = yield from self.move_text_cursor_to_words_generator(
            start_words,
            disambiguate=disambiguate,
            cursor_position="after" if after_start else "before",
            include_whitespace=for_deletion and not after_start,
            click_offset_right=click_offset_right,
            selection_position=self.SelectionPosition.LEFT,
        )
        if not start_locations:
            return None
        time.sleep(select_pause_seconds)
        if end_words:
            if end_timestamp:
                self.read_nearby(end_timestamp)
            else:
                self._read_nearby_if_gaze_moved()
            filter_function = lambda location: self._is_valid_selection(
                start_locations[0].start_coordinates, location[-1].end_coordinates
            )
            return (
                yield from self.move_text_cursor_to_words_generator(
                    end_words,
                    disambiguate=disambiguate,
                    cursor_position="before" if before_end else "after",
                    filter_location_function=filter_function,
                    include_whitespace=False,
                    click_offset_right=click_offset_right,
                    hold_shift=True,
                    selection_position=self.SelectionPosition.RIGHT,
                )
            )
        else:
            self.keyboard.shift_down()
            try:
                if not self._move_text_cursor_to_word_locations(
                    start_locations,
                    cursor_position="before" if before_end else "after",
                    include_whitespace=False,
                    click_offset_right=click_offset_right,
                    selection_position=self.SelectionPosition.RIGHT,
                ):
                    return None
            finally:
                self.keyboard.shift_up()
            return start_locations

    def select_matching_text(
        self,
        words: str,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
        click_offset_right: int = 0,
    ) -> Optional[Tuple[int, int]]:
        """Selects onscreen text that matches the beginning and end of the provided
        text. Returns the start and end indices corresponding to the changed text, if
        found. See select_text for argument details."""
        return self._extract_result(
            self.select_matching_text_generator(
                words,
                disambiguate=False,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                click_offset_right=click_offset_right,
            )
        )

    def select_matching_text_generator(
        self,
        words: str,
        disambiguate: bool,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
        click_offset_right: int = 0,
        select_pause_seconds: float = 0.01,
    ) -> Generator[
        Sequence[Sequence[WordLocation]],
        Sequence[WordLocation],
        Optional[Tuple[int, int]],
    ]:
        """Same as select_matching_text, except it supports disambiguation through a
        generator. See header comment for details."""
        if start_timestamp:
            self.read_nearby(start_timestamp)
        (
            start_locations,
            prefix_length,
        ) = yield from self.move_text_cursor_to_longest_prefix_generator(
            words,
            cursor_position="before",
            disambiguate=disambiguate,
            click_offset_right=click_offset_right,
        )
        if not start_locations:
            return None
        remaining_words = words[prefix_length:]
        time.sleep(select_pause_seconds)
        if end_timestamp:
            self.read_nearby(end_timestamp)
        else:
            self._read_nearby_if_gaze_moved()
        filter_function = lambda location: self._is_valid_selection(
            start_locations[0].start_coordinates, location[-1].end_coordinates
        )
        (
            end_locations,
            suffix_length,
        ) = yield from self.move_text_cursor_to_longest_suffix_generator(
            remaining_words,
            cursor_position="after",
            disambiguate=disambiguate,
            filter_location_function=filter_function,
            click_offset_right=click_offset_right,
            hold_shift=True,
        )
        if not end_locations:
            return None
        return prefix_length, len(words) - suffix_length

    def _read_nearby_if_gaze_moved(self):
        current_gaze = (
            self.eye_tracker.get_gaze_point()
            if self.eye_tracker and self.eye_tracker.is_connected
            else None
        )
        latest_screen_contents = self.latest_screen_contents()
        previous_gaze = latest_screen_contents.screen_coordinates
        threshold_squared = (
            _squared(latest_screen_contents.search_radius / 2.0)
            if latest_screen_contents.search_radius
            else 0.0
        )
        if (
            current_gaze
            and previous_gaze
            and _distance_squared(current_gaze, previous_gaze) > threshold_squared
        ):
            self.read_nearby()

    def move_cursor_to_word_action(self):
        raise RuntimeError(
            "controller.move_cursor_to_word_action no longer supported. "
            "Use gaze_ocr.dragonfly.MoveCursorToWordAction instead."
        )

    def move_text_cursor_action(self, word, cursor_position="middle"):
        """Return a dragonfly action for moving the text cursor nearby a word."""
        raise RuntimeError(
            "controller.move_text_cursor_action no longer supported. "
            "Use gaze_ocr.dragonfly.MoveTextCursorAction instead."
        )

    def select_text_action(self, start_word, end_word=None, for_deletion=False):
        """Return a Dragonfly action for selecting text."""
        raise RuntimeError(
            "controller.select_text_action no longer supported. "
            "Use gaze_ocr.dragonfly.SelectTextAction instead."
        )

    @staticmethod
    def _extract_result(generator):
        """Extracts final return value from generator, assuming no values are generated."""
        try:
            next(generator)
            assert False
        except StopIteration as e:
            return e.value

    @staticmethod
    def _apply_click_offset(coordinates, offset_right):
        return (coordinates[0] + offset_right, coordinates[1])

    def _write_data(self, screen_contents, word, word_locations):
        if not self.save_data_directory:
            return
        if word_locations:
            result = "multiple" if len(word_locations) > 1 else "success"
        else:
            result = "failure"
        file_name_prefix = f"{result}_{time.time():.2f}"
        file_path_prefix = os.path.join(self.save_data_directory, file_name_prefix)
        if hasattr(screen_contents.screenshot, "save"):
            screen_contents.screenshot.save(file_path_prefix + ".png")
        else:
            screen_contents.screenshot.write_file(file_path_prefix + ".png")
        with open(file_path_prefix + ".txt", "w") as file:
            file.write(word)

    def _is_valid_selection(self, start_coordinates, end_coordinates):
        epsilon = 5  # pixels
        (start_x, start_y) = start_coordinates
        (end_x, end_y) = end_coordinates
        # Selection goes to previous line.
        if end_y - start_y < -epsilon:
            return False
        # Selection stays on same line.
        elif end_y - start_y < epsilon:
            return end_x > start_x
        # Selection moves to following line.
        else:
            return True


def _squared(x):
    return x * x


def _distance_squared(coordinate1, coordinate2):
    x_diff = coordinate1[0] - coordinate2[0]
    y_diff = coordinate1[1] - coordinate2[1]
    return _squared(x_diff) + _squared(y_diff)
