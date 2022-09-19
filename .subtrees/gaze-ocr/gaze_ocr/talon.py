import bisect
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from talon import actions, app, tracking_system, ui
from talon.track import tobii
from talon.types import Point2d


class Mouse:
    def move(self, coordinates):
        actions.mouse_move(*coordinates)

    def click(self):
        actions.mouse_click()

    def click_down(self):
        actions.mouse_drag()

    def click_up(self):
        actions.mouse_release()

    def scroll_down(self, n=1):
        for _ in range(n):
            actions.user.mouse_scroll_down()

    def scroll_up(self, n=1):
        for _ in range(n):
            actions.user.mouse_scroll_up()


class Keyboard:
    def __init__(self):
        # shift:down won't affect future keystrokes on Mac, so we track it ourselves.
        self._shift = False

    def type(self, text):
        actions.insert(text)

    def shift_down(self):
        actions.key("shift:down")
        self._shift = True

    def shift_up(self):
        actions.key("shift:up")
        self._shift = False

    def is_shift_down(self):
        return self._shift

    def left(self, n=1):
        # HACK: When adjusting selected text, Mac does not perform the
        # adjustments wherever the cursor was last placed. Instead, it bases
        # this on the first move of the cursor. We start with shift-right to
        # make it possible to shrink the selection from the right. This works in
        # most cases, except when the text has nothing to the right.
        if self._shift:
            if app.platform == "mac":
                actions.key("shift-right")
                actions.key("shift-left")
        for _ in range(n):
            if self._shift:
                actions.key("shift-left")
            else:
                actions.key("left")

    def right(self, n=1):
        for _ in range(n):
            if self._shift:
                actions.key("shift-right")
            else:
                actions.key("right")


class AppActions:
    def peek_left(self) -> Optional[str]:
        try:
            return actions.user.dictation_peek_left()
        # If action is unavailable (e.g. no knausj).
        except KeyError:
            logging.warning("Action user.dictation_peek_left is unavailable.")
            return None

    def peek_right(self) -> Optional[str]:
        try:
            return actions.user.dictation_peek_right()
        # If action is unavailable (e.g. no knausj).
        except KeyError:
            logging.warning("Action user.dictation_peek_right is unavailable.")
            return None


@dataclass
class BoundingBox:
    left: int
    right: int
    top: int
    bottom: int


class TalonEyeTracker:
    STALE_GAZE_THRESHOLD_SECONDS = 0.1

    def __init__(self):
        # !!! Using unstable private API that may break at any time !!!
        tracking_system.register("gaze", self._on_gaze)
        self.is_connected = True
        # Keep approximately 10 seconds of frames on Tobii 5
        self._queue = deque(maxlen=1000)
        # TODO: Remove once Talon is upgraded to Python 3.10 and bisect supports key arg.
        self._ts_queue = deque(maxlen=1000)

    def _on_gaze(self, frame: tobii.GazeFrame):
        if not frame or not frame.gaze:
            return
        self._queue.append(frame)
        self._ts_queue.append(frame.ts)

    def has_gaze_point(self):
        if not self._queue:
            return False
        return (
            self._queue[-1].ts > time.perf_counter() - self.STALE_GAZE_THRESHOLD_SECONDS
        )

    def get_gaze_point(self):
        if not self.has_gaze_point():
            return None
        return self._gaze_to_pixels(self._queue[-1].gaze)

    def get_gaze_point_or_default(self):
        return self.get_gaze_point() or tuple(ui.active_window().rect.center)

    def get_gaze_point_at_timestamp(self, timestamp):
        if not self._queue:
            print("No gaze history available")
            return None
        frame_index = bisect.bisect_left(self._ts_queue, timestamp)
        if frame_index == len(self._queue):
            frame_index -= 1
        frame = self._queue[frame_index]
        if abs(frame.ts - timestamp) > self.STALE_GAZE_THRESHOLD_SECONDS:
            print(
                "No gaze history available at that time: {}. Range: [{}, {}]".format(
                    timestamp, self._ts_queue[0], self._ts_queue[-1]
                )
            )
            return None
        return self._gaze_to_pixels(frame.gaze)

    def get_gaze_bounds_during_time_range(self, start_timestamp, end_timestamp):
        if not self._queue:
            print("No gaze history available")
            return None
        start_index = bisect.bisect_left(self._ts_queue, start_timestamp)
        if start_index == len(self._queue):
            start_index -= 1
        end_index = bisect.bisect_left(self._ts_queue, end_timestamp)
        if end_index == len(self._queue):
            end_index -= 1
        left = right = top = bottom = None
        for i in range(start_index, end_index + 1):
            frame = self._queue[i]
            if frame.ts < start_timestamp - 0.1 or frame.ts > end_timestamp + 0.1:
                continue
            left = min(frame.gaze.x, left) if left is not None else frame.gaze.x
            top = min(frame.gaze.y, top) if top is not None else frame.gaze.y
            right = max(frame.gaze.x, right) if right is not None else frame.gaze.x
            bottom = max(frame.gaze.y, bottom) if bottom is not None else frame.gaze.y
        if left is None or right is None or top is None or bottom is None:
            assert left is None
            assert right is None
            assert top is None
            assert bottom is None
            return None
        top_left = self._gaze_to_pixels(Point2d(x=left, y=top))
        bottom_right = self._gaze_to_pixels(Point2d(x=right, y=bottom))
        return BoundingBox(
            left=top_left[0],
            top=top_left[1],
            right=bottom_right[0],
            bottom=bottom_right[1],
        )

    @staticmethod
    def _gaze_to_pixels(gaze):
        rect = ui.main_screen().rect
        pos = rect.pos + gaze * rect.size
        pos = rect.clamp(pos)
        return (pos.x, pos.y)

    def move_to_gaze_point(self, offset=(0, 0)):
        gaze = self.get_gaze_point_or_default()
        x = gaze[0] + offset[0]
        y = gaze[1] + offset[1]
        actions.mouse_move(x, y)
