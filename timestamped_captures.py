import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from talon import Module, actions
from talon.grammar import Phrase

_subtree_dir = Path(__file__).parent / ".subtrees"
_package_paths = [
    str(_subtree_dir / "gaze-ocr"),
    str(_subtree_dir / "screen-ocr"),
    str(_subtree_dir / "rapidfuzz/src"),
]
_saved_path = sys.path.copy()
try:
    sys.path.extend([p for p in _package_paths if p not in sys.path])
    from gaze_ocr.talon_adapter import BoundingBox, rect_to_pixel_bounding_box
finally:
    sys.path = _saved_path.copy()

mod = Module()


@dataclass
class TimestampedText:
    text: str
    gaze_bounds: Optional[BoundingBox]


@dataclass
class TextRange:
    start: Optional[TimestampedText]
    after_start: bool
    end: Optional[TimestampedText]
    before_end: bool


@dataclass
class TextPosition:
    text: TimestampedText
    position: str


def _gaze_bounds_for(subcapture) -> Optional[BoundingBox]:
    """Resolve the gaze bounding box for a capture or word subcapture.

    Must be called from within a capture function so Talon can look up the
    spoken word metadata. Returns None in environments that don't supply
    gaze data (e.g. mimic()). Only works with literals, lists, and
    built-in captures like <phrase> — not arbitrary user captures."""
    try:
        rect = actions.word.gaze_bounds(subcapture, padding=0.5)
    except (TypeError, KeyError, AttributeError):
        return None
    return rect_to_pixel_bounding_box(rect)


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


@mod.capture(rule="eye | i")
def eye_gaze_bounds(m) -> Optional[BoundingBox]:
    """Gaze bounds for direct-gaze commands (e.g. "eye touch", "eye hover").

    Resolved at capture time so the trigger word's gaze metadata is
    available to actions.word.gaze_bounds."""
    return _gaze_bounds_for(m[0])


# "edit" is frequently misrecognized as "at it", and is common in UIs.
@mod.capture(
    rule="<phrase> | {user.vocabulary} | {user.punctuation} | {user.prose_snippets} | edit"
)
def timestamped_phrase_default(m) -> TimestampedText:
    """Dictated phrase appearing onscreen (default capture).

    Uses only primitives that actions.word.gaze_bounds supports (built-in
    captures and lists)."""
    item = m[0]
    if isinstance(item, Phrase):
        text = " ".join(actions.dictate.replace_words(item))
    else:
        text = str(item)
    return TimestampedText(text=text, gaze_bounds=_gaze_bounds_for(item))


# Forward to enable easy extension via a Context.
@mod.capture(rule="<user.timestamped_phrase_default>")
def timestamped_phrase(m) -> TimestampedText:
    """Dictated phrase appearing onscreen."""
    return m[0]


@mod.capture(rule="<user.timestamped_phrase>+")
def timestamped_prose_only(m) -> TimestampedText:
    """Dictated text appearing onscreen, with merged gaze bounds."""
    items = list(m)
    return TimestampedText(
        text=" ".join(item.text for item in items),
        gaze_bounds=_merge_bounds([item.gaze_bounds for item in items]),
    )


@mod.capture(rule="{user.onscreen_ocr_text}")
def onscreen_text(m) -> TimestampedText:
    """Onscreen text match with gaze bounds resolved at capture time."""
    return TimestampedText(
        text=m[0],
        gaze_bounds=_gaze_bounds_for(m[0]),
    )


@mod.capture(rule="<self.timestamped_prose_only> | <self.onscreen_text>")
def timestamped_prose(m) -> TimestampedText:
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
