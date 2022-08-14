"""Library for manipulating on-screen text using gaze tracking and OCR.

Supports disambiguation using Python generators to stop and resume computation. The *_generator
functions return a generator which can be started with next(generator), which will return a list of
matches if disambiguation is needed. Resume computation with generator.send(match). When computation
completes, next() or send() will raise StopIteration with the .value set to the return value.
"""

import os.path
import time
from concurrent import futures


class Controller(object):
    """Mediates interaction with gaze tracking and OCR."""

    def __init__(
        self,
        ocr_reader,
        eye_tracker,
        save_data_directory=None,
        mouse=None,
        keyboard=None,
    ):
        if not mouse or not keyboard:
            raise RuntimeError(
                "Must provide keyboard and mouse implementation. "
                "Import gaze_ocr.dragonfly or gaze_ocr.talon and use Mouse() and Keyboard()"
            )
        self.ocr_reader = ocr_reader
        self.eye_tracker = eye_tracker
        self.save_data_directory = save_data_directory
        self.mouse = mouse
        self.keyboard = keyboard
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

    def start_reading_nearby(self):
        """Start OCR nearby the gaze point in a background thread."""
        gaze_point = self.eye_tracker.get_gaze_point() if self.eye_tracker else None
        # Don't enqueue multiple requests.
        if self._future and not self._future.done():
            self._future.cancel()
        self._future = self._executor.submit(
            lambda: self.ocr_reader.read_nearby(gaze_point)
            if gaze_point
            else lambda: self.ocr_reader.read_screen()
        )

    # TODO Use timestamp range instead of a single timestamp.
    def read_nearby(self, timestamp=None):
        """Perform OCR nearby the gaze point in the current thread.

        Arguments:
        timestamp: If specified, read nearby the gaze point at the provided timestamp.
        """
        gaze_point = (
            self.eye_tracker.get_gaze_point_at_timestamp(timestamp)
            if self.eye_tracker and timestamp
            else self.eye_tracker.get_gaze_point()
            if self.eye_tracker
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

    def latest_screen_contents(self):
        """Return the ScreenContents of the latest call to start_reading_nearby().

        Blocks until available.
        """
        if not self._future:
            raise RuntimeError(
                "Call start_reading_nearby() before latest_screen_contents()"
            )
        return self._future.result()

    def move_cursor_to_words(
        self, words, cursor_position="middle", timestamp=None, click_offset_right=0
    ):
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
        words,
        disambiguate,
        cursor_position="middle",
        timestamp=None,
        click_offset_right=0,
    ):
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
            return False
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
            return False
        self.mouse.move(self._apply_click_offset(coordinates, click_offset_right))
        return coordinates

    move_cursor_to_word = move_cursor_to_words

    def _screenshot_changed_near_coordinates(
        self, old_screenshot, old_screen_offset, coordinates
    ):
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

    def move_text_cursor_to_words(
        self,
        words,
        cursor_position="middle",
        validate_location_function=None,
        include_whitespace=False,
        timestamp=None,
        click_offset_right=0,
    ):
        """Move the text cursor nearby the specified word or phrase.

        If successful, returns list of screen_ocr.WordLocation of the matching words.

        Arguments:
        words: The word or phrase to search for.
        cursor_position: "before", "middle", or "after" (relative to the matching word).
        validate_location_function: Given a sequence of word locations, return whether to proceed with
                                    cursor movement.
        include_whitespace: Include whitespace to the left of the words.
        timestamp: Use gaze position at the provided timestamp.
        click_offset_right: Adjust the X-coordinate when clicking.
        """
        return self._extract_result(
            self.move_text_cursor_to_words_generator(
                words,
                disambiguate=False,
                cursor_position=cursor_position,
                validate_location_function=validate_location_function,
                include_whitespace=include_whitespace,
                timestamp=timestamp,
                click_offset_right=click_offset_right,
            )
        )

    def move_text_cursor_to_words_generator(
        self,
        words,
        disambiguate,
        cursor_position="middle",
        validate_location_function=None,
        include_whitespace=False,
        timestamp=None,
        click_offset_right=0,
        hold_shift=False,
    ):
        """Same as move_text_cursor_to_words, except it supports disambiguation through a generator.
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
            return False
        if len(matches) > 1:
            locations = yield matches
        else:
            locations = matches[0]
        if validate_location_function and not validate_location_function(locations):
            return False
        if hold_shift:
            self.keyboard.shift_down()
        try:
            if not self._move_text_cursor_to_word_locations(
                locations,
                cursor_position=cursor_position,
                include_whitespace=include_whitespace,
                click_offset_right=click_offset_right,
            ):
                return False
        finally:
            if hold_shift:
                self.keyboard.shift_up()
        return locations

    move_text_cursor_to_word = move_text_cursor_to_words

    def _move_text_cursor_to_word_locations(
        self,
        locations,
        cursor_position="middle",
        include_whitespace=False,
        click_offset_right=0,
    ):
        if cursor_position == "before":
            distance_from_left = locations[0].left_char_offset
            distance_from_right = locations[0].right_char_offset + len(
                locations[0].text
            )
            if distance_from_left <= distance_from_right:
                coordinates = self._apply_click_offset(
                    locations[0].start_coordinates, click_offset_right
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
                coordinates = self._apply_click_offset(
                    locations[0].end_coordinates, click_offset_right
                )
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
            if not distance_from_left and include_whitespace:
                # Assume that there is whitespace adjacent to the word. This
                # will gracefully fail if the word is the first in the
                # editable text area.
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
            if distance_from_right <= distance_from_left:
                coordinates = self._apply_click_offset(
                    locations[-1].end_coordinates, click_offset_right
                )
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
            else:
                coordinates = self._apply_click_offset(
                    locations[-1].start_coordinates, click_offset_right
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
            if not distance_from_right and include_whitespace:
                # Assume that there is whitespace adjacent to the word. This
                # will gracefully fail if the word is the last in the
                # editable text area.
                self.keyboard.right(1)
        return True

    def select_text(
        self,
        start_words,
        end_words=None,
        for_deletion=False,
        start_timestamp=None,
        end_timestamp=None,
        click_offset_right=0,
        after_start=False,
        before_end=False,
    ):
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
        start_words,
        disambiguate,
        end_words=None,
        for_deletion=False,
        start_timestamp=None,
        end_timestamp=None,
        click_offset_right=0,
        after_start=False,
        before_end=False,
        select_pause_seconds=0.01,
    ):
        """Same as select_text, except it supports disambiguation through a generator.
        See header comment for details.
        """
        # Automatically split up start word if multiple words are provided.
        if start_timestamp:
            self.read_nearby(start_timestamp)
        # Always click before the word to avoid subword selection issues on Windows.
        start_locations = yield from self.move_text_cursor_to_words_generator(
            start_words,
            disambiguate=disambiguate,
            cursor_position="after" if after_start else "before",
            include_whitespace=for_deletion,
            click_offset_right=click_offset_right,
        )
        if not start_locations:
            return False
        # Emacs requires a small sleep in between mouse clicks.
        time.sleep(select_pause_seconds)
        if end_words:
            if end_timestamp:
                self.read_nearby(end_timestamp)
            else:
                # If gaze has significantly moved, look for the end word at the final gaze coordinates.
                current_gaze = (
                    self.eye_tracker.get_gaze_point() if self.eye_tracker else None
                )
                previous_gaze = self.latest_screen_contents().screen_coordinates
                threshold_squared = _squared(
                    self.latest_screen_contents().search_radius / 2.0
                )
                if (
                    current_gaze
                    and previous_gaze
                    and _distance_squared(current_gaze, previous_gaze)
                    > threshold_squared
                ):
                    self.read_nearby()
            validate_function = lambda location: self._is_valid_selection(
                start_locations[0].start_coordinates, location[-1].end_coordinates
            )
            # Always click after the word to avoid subword selection issues on Windows.
            return (
                yield from self.move_text_cursor_to_words_generator(
                    end_words,
                    disambiguate=disambiguate,
                    cursor_position="before" if before_end else "after",
                    validate_location_function=validate_function,
                    include_whitespace=False,
                    click_offset_right=click_offset_right,
                    hold_shift=True,
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
                ):
                    return False
            finally:
                self.keyboard.shift_up()
            return start_locations

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
