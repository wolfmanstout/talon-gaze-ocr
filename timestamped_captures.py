from dataclasses import dataclass
from typing import Optional

from talon import Module, actions
from talon.grammar import Phrase

mod = Module()


@dataclass
class TimestampedText:
    text: str
    start: float
    end: float


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


@mod.capture(rule="<user.timestamped_phrase>+")
def timestamped_prose(m) -> TimestampedText:
    """Dictated text appearing onscreen."""
    return TimestampedText(
        text=" ".join([item.text for item in m]), start=m[0].start, end=m[-1].end
    )


# Forward to enable easy extension via a Context.
@mod.capture(rule="<user.timestamped_phrase_default>")
def timestamped_phrase(m) -> TimestampedText:
    """Dictated phrase appearing onscreen."""
    return m[0]


# "edit" is frequently misrecognized as "at it", and is common in UIs.
@mod.capture(
    rule="<phrase> | {user.vocabulary} | {user.punctuation} | {user.prose_snippets} | edit"
)
def timestamped_phrase_default(m) -> TimestampedText:
    """Dictated phrase appearing onscreen (default capture)."""
    item = m[0]
    if isinstance(item, Phrase):
        text = " ".join(actions.dictate.replace_words(item))
        # TODO: Remove fallbacks once Phrase changes have been pushed to stable.
        try:
            start = item[0].start
        except AttributeError:
            start = item.words[0].start
        try:
            end = item[-1].end
        except AttributeError:
            end = item.words[-1].end
    else:
        text = str(item)
        start = item.start
        end = item.end
    return TimestampedText(text=text, start=start, end=end)


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


@mod.capture(rule="[through] <self.prose_position>")
def one_ended_prose_range(m) -> TextRange:
    """A range of onscreen text with only start or end specified."""
    has_through = m[0] == "through"
    # As a convenience, allow dropping "through" if position is provided.
    if has_through or m.prose_position.position:
        if not m.prose_position.position:
            actions.app.notify(
                'Try "[through] before <phrase>" or "[through] after <phrase>" instead'
                ' of "through <phrase>". The cursor position is unknown to '
                "talon-gaze-ocr."
            )
            raise ValueError(
                'Text range "through <phrase>" not supported because cursor position is unknown.'
            )
        return TextRange(
            start=None,
            after_start=False,
            end=m.prose_position.text,
            before_end=m.prose_position.position == "before",
        )
    else:
        return TextRange(
            start=m.prose_position.text,
            after_start=m.prose_position.position == "after",
            end=None,
            before_end=False,
        )
