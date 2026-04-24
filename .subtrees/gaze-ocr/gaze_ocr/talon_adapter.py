import logging
from dataclasses import dataclass
from typing import Optional

from talon import actions, ui


class Mouse:
    def move(self, coordinates):
        actions.mouse_move(*coordinates)

    def click(self):
        actions.mouse_click()


class Keyboard:
    def __init__(self):
        # shift:down won't affect future keystrokes on Mac, so we track it ourselves.
        self._shift = False

    def shift_down(self):
        actions.key("shift:down")
        self._shift = True

    def shift_up(self):
        actions.key("shift:up")
        self._shift = False

    def is_shift_down(self):
        return self._shift

    def left(self, n=1):
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
    def focus_at(self, x: int, y: int):
        """Focus the window at the given coordinates."""
        actions.user.focus_at(x, y)

    def peek_left(self) -> Optional[str]:
        try:
            return actions.user.dictation_peek(True, False)[0]
        except KeyError:
            try:
                return actions.user.dictation_peek_left()
            # If action is unavailable (e.g. no knausj).
            except KeyError:
                logging.warning("Action user.dictation_peek is unavailable.")
                return None

    def peek_right(self) -> Optional[str]:
        try:
            return actions.user.dictation_peek(False, True)[1]
        except KeyError:
            try:
                return actions.user.dictation_peek_right()
            # If action is unavailable (e.g. no knausj).
            except KeyError:
                logging.warning("Action user.dictation_peek is unavailable.")
                return None


@dataclass
class BoundingBox:
    left: int
    right: int
    top: int
    bottom: int


def rect_to_pixel_bounding_box(rect) -> Optional[BoundingBox]:
    """Convert a normalized skia.Rect returned by actions.word.gaze_bounds
    into an absolute-pixel BoundingBox on the main screen."""
    if rect is None:
        return None
    screen = ui.main_screen().rect
    left = int(screen.x + rect.x * screen.width)
    top = int(screen.y + rect.y * screen.height)
    right = int(screen.x + (rect.x + rect.width) * screen.width)
    bottom = int(screen.y + (rect.y + rect.height) * screen.height)
    return BoundingBox(left=left, right=right, top=top, bottom=bottom)


class TalonEyeTracker:
    """Tracks whether gaze-based lookups are desired. Gaze bounds are resolved
    at capture time via actions.word.gaze_bounds, so this adapter no longer
    subscribes to a gaze stream. Talon's "Always On" tracking option is
    controlled by the user via the tray UI (no public action exposes it)."""

    def __init__(self):
        self.is_connected = False
        self.connect()

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    # Stubs preserved so the generic Controller can fall through to its
    # screen/window OCR path when no gaze_bounds are supplied. Talon callers
    # always pass gaze_bounds resolved at capture time.
    def get_gaze_point(self):
        return None

    def get_gaze_bounds_during_time_range(self, *_args):
        return None
