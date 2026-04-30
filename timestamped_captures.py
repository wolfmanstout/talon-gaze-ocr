from dataclasses import dataclass
from typing import Optional

from talon import Module, actions, ui
from talon.grammar import Phrase


@dataclass
class BoundingBox:
    left: int
    right: int
    top: int
    bottom: int


@dataclass
class GazePoint:
    x: int
    y: int


def rect_to_pixel_bounding_box(rect) -> Optional[BoundingBox]:
    """Convert a normalized skia.Rect from actions.word.gaze_bounds into an
    absolute-pixel BoundingBox on the main screen."""
    if rect is None:
        return None
    screen = ui.main_screen().rect
    left = int(screen.x + rect.x * screen.width)
    top = int(screen.y + rect.y * screen.height)
    right = int(screen.x + (rect.x + rect.width) * screen.width)
    bottom = int(screen.y + (rect.y + rect.height) * screen.height)
    return BoundingBox(left=left, right=right, top=top, bottom=bottom)


def point_to_pixel_gaze(point) -> Optional[GazePoint]:
    """Convert a normalized skia.Point from actions.word.gaze into an
    absolute-pixel GazePoint on the main screen."""
    if point is None:
        return None
    screen = ui.main_screen().rect
    x = int(screen.x + point.x * screen.width)
    y = int(screen.y + point.y * screen.height)
    return GazePoint(x=x, y=y)


mod = Module()


@dataclass
class SeenText:
    text: str
    gaze_bounds: Optional[BoundingBox]


@dataclass
class TextRange:
    start: Optional[SeenText]
    after_start: bool
    end: Optional[SeenText]
    before_end: bool


@dataclass
class TextPosition:
    text: SeenText
    position: str


def _gaze_bounds_for(subcapture, padding: float = 0.5) -> Optional[BoundingBox]:
    """Resolve the gaze bounding box for a capture or word subcapture.

    Must be called from within a capture function so Talon can look up the
    spoken word metadata. Returns None in environments that don't supply
    gaze data (e.g. mimic()). Only works with literals, lists, and
    built-in captures like <phrase> — not arbitrary user captures."""
    rect = actions.word.gaze_bounds(subcapture, padding=padding)
    return rect_to_pixel_bounding_box(rect)


def _gaze_point_for(subcapture) -> Optional[GazePoint]:
    """Resolve the gaze point for a capture or word subcapture."""
    point = actions.word.gaze(subcapture)
    return point_to_pixel_gaze(point)


def _merge_bounds(
    bounds: list[Optional[BoundingBox]],
) -> Optional[BoundingBox]:
    """Compute the union rectangle of a list of BoundingBoxes."""
    valid = [b for b in bounds if b is not None]
    if not valid:
        return None
    return BoundingBox(
        left=min(b.left for b in valid),
        right=max(b.right for b in valid),
        top=min(b.top for b in valid),
        bottom=max(b.bottom for b in valid),
    )


def phrase_gaze_bounds(phrase) -> Optional[BoundingBox]:
    """Resolve merged gaze bounds for each item in a phrase-like iterable."""
    return _merge_bounds([_gaze_bounds_for(item) for item in phrase])


@mod.capture(rule="eye | i")
def eye_gaze_point(m) -> Optional[GazePoint]:
    """Gaze point for direct-gaze commands (e.g. "eye touch", "eye hover").

    Resolved at capture time so the trigger word's gaze metadata is
    available to actions.word.gaze."""
    return _gaze_point_for(m[0])


# "edit" is frequently misrecognized as "at it", and is common in UIs.
@mod.capture(
    rule="<phrase> | {user.vocabulary} | {user.punctuation} | {user.prose_snippets} | edit"
)
def timestamped_phrase_default(m) -> SeenText:
    """Dictated phrase appearing onscreen (default capture).

    Uses only primitives that actions.word.gaze_bounds supports (built-in
    captures and lists)."""
    item = m[0]
    if isinstance(item, Phrase):
        text = " ".join(actions.dictate.replace_words(item))
    else:
        text = str(item)
    return SeenText(text=text, gaze_bounds=_gaze_bounds_for(item))


# Forward to enable easy extension via a Context.
@mod.capture(rule="<user.timestamped_phrase_default>")
def timestamped_phrase(m) -> SeenText:
    """Dictated phrase appearing onscreen."""
    return m[0]


@mod.capture(rule="<user.timestamped_phrase>+")
def timestamped_prose_only(m) -> SeenText:
    """Dictated text appearing onscreen, with merged gaze bounds."""
    items = list(m)
    return SeenText(
        text=" ".join(item.text for item in items),
        gaze_bounds=_merge_bounds([item.gaze_bounds for item in items]),
    )


@mod.capture(rule="{user.onscreen_ocr_text}")
def onscreen_text(m) -> SeenText:
    """Onscreen text match with gaze bounds resolved at capture time."""
    return SeenText(
        text=m[0],
        gaze_bounds=_gaze_bounds_for(m[0]),
    )


@mod.capture(rule="<self.timestamped_prose_only> | <self.onscreen_text>")
def timestamped_prose(m) -> SeenText:
    """Timestamped prose or onscreen text."""
    return m[0]


@mod.capture(rule="[before | after] <self.timestamped_prose>")
def prose_position(m) -> TextPosition:
    """Position relative to prose."""
    return TextPosition(
        text=m.timestamped_prose,
        position=m[0] if m[0] in ("before", "after") else "",
    )


@mod.capture(
    rule="<self.one_ended_prose_range> | <self.prose_position> through <self.prose_position>"
)
def prose_range(m) -> TextRange:
    """A range of onscreen text."""
    if hasattr(m, "one_ended_prose_range"):
        return m.one_ended_prose_range
    return TextRange(
        start=m.prose_position_1.text,
        after_start=m.prose_position_1.position == "after",
        end=m.prose_position_2.text,
        before_end=m.prose_position_2.position == "before",
    )


@mod.capture(rule="[through | from] <self.prose_position>")
def one_ended_prose_range(m) -> TextRange:
    """A range of onscreen text with only start or end specified."""
    has_through = m[0] == "through"
    has_from = m[0] == "from"
    # As a convenience, allow dropping "through" or "from" if position is provided.
    if m.prose_position.position:
        return TextRange(
            start=None,
            after_start=False,
            end=m.prose_position.text,
            before_end=m.prose_position.position == "before",
        )
    elif has_through:
        # Select current cursor point through the text after the cursor.
        return TextRange(
            start=None,
            after_start=False,
            end=m.prose_position.text,
            before_end=False,
        )
    elif has_from:
        # Select from the text before the cursor up to the current cursor point.
        return TextRange(
            start=None,
            after_start=False,
            end=m.prose_position.text,
            before_end=True,
        )
    else:
        # Select the phrase itself.
        return TextRange(
            start=m.prose_position.text,
            after_start=m.prose_position.position == "after",
            end=None,
            before_end=False,
        )
